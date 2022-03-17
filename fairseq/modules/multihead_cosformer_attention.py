import math
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter

from torch.nn.modules.module import _addindent

from fairseq import utils
import fairseq.init as init

from .linear_super import LinearSuper, Linear

from fast_transformers.attention.causal_linear_attention import causal_linear

# cosformer
@with_incremental_state
class MultiheadCosformerAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, is_encoder, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, out_dim=None, qkv_dim=None, is_fixed=False,
                 causal=False, seq_len=4096,): # add
        super().__init__()

        # the configs of super arch
        self.super_q_embed_dim = embed_dim
        self.super_kv_embed_dim = None
        self.fixed = is_fixed

        # the configs of current sampled arch
        self.sample_q_embed_dim = None
        self.sample_kv_embed_dim = None

        if kdim is not None:
            assert kdim == vdim
            self.super_kv_embed_dim = kdim
        else:
            self.super_kv_embed_dim = self.super_q_embed_dim

        if qkv_dim is None:
            self.qkv_dim = self.super_q_embed_dim
        else:
            self.qkv_dim = qkv_dim

        # this qkv same dim means the input dim for qkv are the same, not the output dim
        # self.qkv_same_dim = self.kdim == self.super_embed_dim and self.vdim == self.super_embed_dim
        self.qkv_same_dim = self.super_kv_embed_dim == self.super_q_embed_dim
        self.encoder = is_encoder

        # Caution! these actually are the sampled num_heads, head_dim and scaling
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.qkv_dim // num_heads
        assert self.head_dim * num_heads == self.qkv_dim, "qkv must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * self.qkv_dim, self.super_q_embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(self.qkv_dim, self.super_kv_embed_dim))
            self.v_proj_weight = Parameter(torch.Tensor(self.qkv_dim, self.super_kv_embed_dim))
            self.q_proj_weight = Parameter(torch.Tensor(self.qkv_dim, self.super_q_embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.qkv_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        if out_dim is None:
            out_dim = self.super_q_embed_dim

        if not is_fixed:
            self.out_proj = LinearSuper(super_in_dim=self.qkv_dim, super_out_dim=out_dim, bias=bias)
        else:
            self.out_proj = Linear(self.qkv_dim, out_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.super_q_embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.super_q_embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # add begin
        # causal
        self.causal = causal
        # weight index
        self.weight_index = self.get_index(seq_len)
        # add end

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.enable_torch_version = False

    # add begin
    def get_index(self, seq_len):
        a = np.pi / 2
        index = a * torch.arange(1, seq_len + 1).reshape(1, -1, 1, 1)

        return nn.Parameter(index, requires_grad=False)
    # add end

    def calc_sampled_param_num(self):
        assert self.in_proj_weight is not None and self.in_proj_bias is not None
        in_proj_q_weight_numel = self.sample_q_embed_dim * self.qkv_dim
        in_proj_v_weight_numel = in_proj_k_weight_numel = self.sample_kv_embed_dim * self.qkv_dim
        in_proj_bias_numel = self.in_proj_bias.numel()

        # does not count in the output proj because it will be counted in LinearSuper layer
        # out_proj_weight_numel = self.qkv_dim * self.sample_q_embed_dim
        # out_proj_bias_numel = self.

        return in_proj_q_weight_numel + in_proj_k_weight_numel + in_proj_v_weight_numel + in_proj_bias_numel

    def set_sample_config(self, sample_q_embed_dim, sample_attention_heads, sample_kv_embed_dim=None):
        self.sample_q_embed_dim = sample_q_embed_dim
        if sample_kv_embed_dim is None:
            self.sample_kv_embed_dim = sample_q_embed_dim
        else:
            self.sample_kv_embed_dim = sample_kv_embed_dim

        self.num_heads = sample_attention_heads
        self.head_dim = self.qkv_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.qkv_dim, "qkv_dim must be divisible by sampled num_heads"
        self.scaling = self.head_dim ** -0.5

        if not self.fixed:
            self.out_proj.set_sample_config(sample_in_dim=self.qkv_dim, sample_out_dim=self.sample_q_embed_dim)


    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)


    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        #print('super!!!!!!!!!!!!!!!!!')
        #num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        #head_dim = embed_dim // num_heads

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        
        #q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        # multihead
        # (N, L, h, d)
        #print(q.shape)
        
        q = q.contiguous().view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # (N, S, h, d)
        k = k.contiguous().view(-1, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # (N, S, h, d)
        v = v.contiguous().view(-1, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        # relu
        q = F.relu(q)
        k = F.relu(k)

        # transformation
        # (N, L, h, 2 * d)
        m = max(tgt_len,src_len)
        q_ = torch.cat([q * torch.sin(self.weight_index[:, :tgt_len, :, :] / tgt_len), q * torch.cos(self.weight_index[:, :tgt_len, :, :] / tgt_len)], dim=-1)
        # (N, S, h, 2 * d)
        #print(k.shape)
        #print((torch.sin(self.weight_index[:, :src_len, :, :]).shape))
        
        k_ = torch.cat([k * torch.sin(self.weight_index[:, :src_len, :, :] / src_len), k * torch.cos(self.weight_index[:, :src_len, :, :] / src_len)], dim=-1)
        eps = 1e-3

        if self.causal:
            # (N, L, h, 2 * d), (N, S, h, 2 * d), (N, S, h, d) -> (N, L, h, d)
            qkv_cos_sin = causal_linear(q_, k_, v)
            # 分母
            # (N, L, h)
            z_cos_sin = 1 / torch.clamp_min(torch.einsum('nlhi,nlhi->nlh', q_, torch.cumsum(k_, dim=1)), eps)
            # (N, L, h, d) -> (L, N, h, d) -> (L, N, E)
            attn_output = (qkv_cos_sin * z_cos_sin.unsqueeze(-1)).transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N, S, h, 2 * d) (N, S, h, d) -> (N, h, d, 2 * d)
            kv_ = torch.einsum('nshd,nshm->nhmd', k_, v)
            # (N, L, h, 2 * d) (N, h, 2 * d) -> (N, L, h)
            z_ = 1 / torch.clamp_min(torch.einsum('nlhd,nhd->nlh', q_, torch.sum(k_, axis=1)), eps)
            # (N, L, h, 2 * d) (N, h, d, 2 * d) (N, L, h) -> (N, L, h, d)
            attn_output = torch.einsum('nlhd,nhmd,nlh->nlhm', q_, kv_, z_)

            # (N, L, h, d) -> (L, N, h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # output
        # (L, N, E) -> (L, N, E)
        attn_output = self.out_proj(attn_output)

        attn_weights = None

        return attn_output, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query, sample_dim=self.sample_q_embed_dim).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.qkv_dim, sample_dim=self.sample_q_embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.qkv_dim]
            return F.linear(query, self.q_proj_weight[..., :self.sample_q_embed_dim], bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.qkv_dim, end=2 * self.qkv_dim, sample_dim=self.sample_kv_embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.qkv_dim:2 * self.qkv_dim]
            return F.linear(key, weight[..., :self.sample_kv_embed_dim], bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.qkv_dim, sample_dim=self.sample_kv_embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.qkv_dim:]
            return F.linear(value, weight[..., :self.sample_kv_embed_dim], bias)

    def _in_proj(self, input, sample_dim, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :sample_dim]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '\tnum_heads:' + str(self.num_heads) + '\t qkv_dim:' + str(self.qkv_dim)
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

