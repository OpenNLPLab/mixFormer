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
from fairseq.modules  import LayerNormSuper
from torch.nn.modules.module import _addindent

from fairseq import utils
import fairseq.init as init

from .linear_super import LinearSuper, Linear
from .conv_super import Conv2dSuper
from fast_transformers.attention.causal_linear_attention import causal_linear


# cosformer
@with_incremental_state
class MultiheadCosformerAttention2dSuper(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, is_encoder, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, out_dim=None, qkv_dim=None, is_fixed=False,
                 causal=False, seq_len=4096,
                 dropout_rate=0.0,use_sum=True, sr_ratio=2, fr_ratio=1, linear=False, se_reduction=2):  # add
        super().__init__()

        # the configs of super arch
        self.super_q_embed_dim = embed_dim
        self.super_kv_embed_dim = None
        self.fixed = is_fixed
        self.fr = fr_ratio
        self.sr_ratio = sr_ratio
        # the configs of current sampled arch
        self.sample_q_embed_dim = None
        self.sample_kv_embed_dim = None
        self.linear = linear
        self.dropout_rate = dropout_rate

        if not linear:
            if sr_ratio > 1:
                self.sr = Conv2dSuper(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = LayerNormSuper(embed_dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = Conv2dSuper(embed_dim, embed_dim, kernel_size=1, stride=1)
            self.norm = LayerNormSuper(embed_dim)
            self.act = nn.GELU()

        self.apply(self._init_weights)

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
        #self.weight_index = self.get_index(seq_len)
        # add end
        self.use_sum = use_sum
        if self.use_sum:
            print('using 2d cosformer!', 'use sum')
            print('linear:', linear, 'sr_ratio:', sr_ratio, 'fr_ratio:', fr_ratio, 'se_ratio:', se_reduction)
        else:
            print('using 2d cosformer!', 'use peoduction')
            print('linear:', linear, 'sr_ratio:', sr_ratio, 'fr_ratio:', fr_ratio, 'se_ratio:', se_reduction)
        self.reset_parameters()
        ### se block
        self.reduction = se_reduction
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc1 = LinearSuper(super_in_dim=embed_dim, super_out_dim=embed_dim // self.reduction, bias=False)
        self.se_relu = nn.ReLU(inplace=True)
        self.se_fc2 = LinearSuper(super_in_dim=embed_dim // self.reduction, super_out_dim=embed_dim, bias=False)
        self.se_sigmoid = nn.Sigmoid()
        # self.se_fc = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(embed_dim // reduction, embed_dim, bias=False),
        #     nn.Sigmoid()
        # )

        self.clip = True

        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.enable_torch_version = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # add begin
    def get_index(self, m, n):
        """
        m = width, n = highth
        """
        c = np.pi / 2
        seq_len = m * n
        index = torch.arange(seq_len).reshape(1, -1, 1, 1)
        a = c * (index // m) / n
        b = c * (index % m) / m
        # a = a.half()
        # b = b.half()

        seq_len = (m/self.sr_ratio) * (n/self.sr_ratio)
        index = torch.arange(seq_len).reshape(1, -1, 1, 1)
        a_sr = c * (index // (m/self.sr_ratio) ) / (n/self.sr_ratio)
        b_sr = c * (index % (m/self.sr_ratio)) / (m/self.sr_ratio)

        return nn.Parameter(a, requires_grad=False), nn.Parameter(b, requires_grad=False), \
                nn.Parameter(a_sr, requires_grad=False), nn.Parameter(b_sr, requires_grad=False)

    def abs_clamp(self, t):
        min_mag = 1e-4
        max_mag = 10000
        sign = t.sign()
        return t.abs_().clamp_(min_mag, max_mag)*sign

    def calc_sampled_param_num(self):
        assert self.in_proj_weight is not None and self.in_proj_bias is not None

        in_proj_q_weight_numel = self.sample_q_embed_dim * self.sample_embed_dim
        in_proj_v_weight_numel = in_proj_k_weight_numel = self.sample_kv_embed_dim * self.sample_embed_dim
        in_proj_bias_numel = self.in_proj_bias.numel()
        # does not count in the output proj because it will be counted in LinearSuper layer
        # out_proj_weight_numel = self.qkv_dim * self.sample_q_embed_dim
        # out_proj_bias_numel = self.

        return in_proj_q_weight_numel + in_proj_k_weight_numel + in_proj_v_weight_numel + in_proj_bias_numel + sr_numel

    def set_sample_config(self, sample_embed_dim, sample_q_embed_dim, sample_attention_heads, sample_kv_embed_dim=None):
        self.sample_embed_dim = sample_embed_dim
        self.sample_q_embed_dim = sample_q_embed_dim
        if sample_kv_embed_dim is None:
            self.sample_kv_embed_dim = sample_q_embed_dim
        else:
            self.sample_kv_embed_dim = sample_kv_embed_dim

        self.num_heads = sample_attention_heads
        self.head_dim = self.sample_q_embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.sample_q_embed_dim, "qkv_dim must be divisible by sampled num_heads"
        self.scaling = self.head_dim ** -0.5

        if not self.fixed:
            self.out_proj.set_sample_config(sample_in_dim=sample_q_embed_dim, sample_out_dim=self.sample_embed_dim)

        self.sr.set_sample_config(sample_in_dim=sample_embed_dim, sample_out_dim=self.sample_embed_dim)
        self.norm.set_sample_config(
                    sample_embed_dim=self.sample_embed_dim)
        self.se_fc1.set_sample_config(sample_in_dim=sample_embed_dim, sample_out_dim=self.sample_embed_dim // self.reduction)

        self.se_fc2.set_sample_config(sample_in_dim=sample_embed_dim // self.reduction,
                                      sample_out_dim=self.sample_embed_dim)
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

    def forward(self, query, H, W, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        # print('super!!!!!!!!!!!!!!!!!')
        # num_heads = self.num_heads
        query = query.permute(1, 0, 2)
        num_heads = self.num_heads
        B, N, C = query.shape
        query_se = query.permute(0, 2, 1)
        query_se = self.se_pool(query_se).view(B, C)
        query_se = self.se_fc1(query_se)
        query_se = self.se_relu(query_se)
        query_se = self.se_fc2(query_se)
        query_se = self.se_sigmoid(query_se).view(B, C, 1)
        # query_se = self.se_fc(query_se).view(B, C, 1)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = query.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = x_.permute(1, 0, 2)
                k = self.in_proj_k(x_)
                v = self.in_proj_v(x_)
            else:
                k = self.in_proj_k(query.permute(1, 0, 2))
                v = self.in_proj_v(query.permute(1, 0, 2))

        else:
            x_ = query.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            k = self.in_proj_k(x_.permute(1, 0, 2))
            v = self.in_proj_v(x_.permute(1, 0, 2))

        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        query = query.permute(1, 0, 2)
        q = self.in_proj_q(query)
        tgt_len, bsz, embed_dim = query.size()
        head_dim = self.head_dim

        src_len = key.size(0)
        # head_dim = embed_dim // num_heads

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

        # q *= self.scaling

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
        # print(q.shape)

        q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim // self.fr).transpose(0, 1)
        # (N, S, h, d)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim // self.fr).transpose(0, 1)
        # (N, S, h, d)
        v = v.contiguous().view(-1, bsz, num_heads, head_dim // self.fr).transpose(0, 1)

        # relu
        q = F.relu(q)
        k = F.relu(k)

        a, b, a_sr, b_sr = self.get_index(W, H)
        a = a.to(q)
        b = b.to(q)
        a_sr = a_sr.to(q)
        b_sr = b_sr.to(q)

        if self.use_sum:
            # sum
            q_ = torch.cat([q * torch.cos(a), \
                            q * torch.sin(a), \
                            q * torch.cos(b), \
                            q * torch.sin(b)], \
                            dim=-1)
            # (N, S, h, 2 * d)
            k_ = torch.cat([k * torch.cos(a_sr), \
                            k * torch.sin(a_sr), \
                            k * torch.cos(b_sr), \
                            k * torch.sin(b_sr)], \
                            dim=-1)
            #print('q_k_:', q_.dtype, k_.dtype)
        else:
            q_ = torch.cat([q * torch.cos(a) * torch.cos(b), \
                            q * torch.cos(a) * torch.sin(b), \
                            q * torch.sin(a) * torch.cos(b), \
                            q * torch.sin(a) * torch.sin(b)], \
                            dim=-1)
            # (N, S, h, 4 * d)
            k_ = torch.cat([k * torch.cos(a_sr) * torch.cos(b_sr), \
                            k * torch.cos(a_sr) * torch.sin(b_sr), \
                            k * torch.sin(a_sr) * torch.cos(b_sr), \
                            k * torch.sin(a_sr) * torch.sin(b_sr)], \
                            dim=-1)
        eps = 1e-4
        kv_ = torch.matmul(k_.permute(0, 2, 3, 1), v.permute(0, 2, 1, 3))  # no einsum
        if self.clip:
            kv_ = self.abs_clamp(kv_)
        # ---------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        k_sum = torch.sum(k_, axis=1, keepdim=True)  # no einsum
        z_ = 1 / (torch.sum(torch.mul(q_, k_sum), axis=-1) + eps)  # no einsum
        if self.clip:
            z_ = self.abs_clamp(z_)
        # --------------------------------------------------------------------------------

        # no einsum---------------------------------------------------------------------
        attn_output = torch.matmul(q_.transpose(1, 2), kv_).transpose(1, 2)
        if self.clip:
            attn_output = self.abs_clamp(attn_output)
        # print('attn_output ', attn_output.shape)
        # nlhm,nlh -> nlhm
        attn_output = torch.mul(attn_output, z_.unsqueeze(-1))
        if self.clip:
            attn_output = self.abs_clamp(attn_output)
        # --------------------------------------------------------------------------------
        # (N, L, h, d) -> (L, N, h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.sample_q_embed_dim // self.fr)

        attn_output = self.out_proj(attn_output)
        if self.clip:
            attn_output = self.abs_clamp(attn_output)

        # ------------------------------------- se block
        attn_output = attn_output.permute(1, 2, 0)
        attn_output = attn_output + attn_output * query_se.expand_as(attn_output)
        if self.clip:
            attn_output = self.abs_clamp(attn_output)
        attn_output = attn_output.permute(2, 0, 1)
        # -------------------------------------------------

        # attn_output = attn_output.permute(1, 0, 2)

        attn_weights = None

        return attn_output, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query, sample_dim=self.sample_embed_dim, end=self.sample_q_embed_dim*3).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.sample_q_embed_dim, sample_dim=self.sample_embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.qkv_dim]
            return F.linear(query, self.q_proj_weight[..., :self.sample_q_embed_dim], bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.sample_q_embed_dim, end=2 * self.sample_q_embed_dim, sample_dim=self.sample_embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.qkv_dim:2 * self.qkv_dim]
            return F.linear(key, weight[..., :self.sample_kv_embed_dim], bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.sample_q_embed_dim, end=3 * self.sample_q_embed_dim, sample_dim=self.sample_embed_dim)
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

