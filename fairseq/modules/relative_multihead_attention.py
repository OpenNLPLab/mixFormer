# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import _addindent

from fairseq import utils
import fairseq.init as init
from .linear_super import LinearSuper, Linear


class RelativeMultiheadAttention(nn.Module):
    """Relative Multi-headed attention.

    See "Self-Attention with Relative Position Representations" for more details.
    """

    def __init__(self, super_embed_dim, num_heads, is_encoder, super_kdim=None, super_vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, out_dim=None, qkv_dim=None,
                 max_relative_length=-1, k_only=True, is_fixed=False):
        super().__init__()

        # the configs of super arch
        self.super_q_embed_dim = super_embed_dim
        self.super_kv_embed_dim = None
        self.k_only = k_only
        self.fixed = is_fixed

        # the configs of current sampled arch
        self.sample_q_embed_dim = None
        self.sample_kv_embed_dim = None

        if super_kdim is not None:
            assert super_kdim == super_vdim
            self.super_kv_embed_dim = super_kdim
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

        self.num_heads = num_heads
        self.max_relative_length = max_relative_length
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

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.qkv_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        if out_dim is None:
            out_dim = self.super_q_embed_dim

        if is_fixed:
            self.out_proj = Linear(self.qkv_dim, out_dim, bias=bias)
        else:
            self.out_proj = LinearSuper(super_in_dim=self.qkv_dim, super_out_dim=out_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.super_q_embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.super_q_embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.relative_position_keys = Parameter(torch.Tensor(2 * self.max_relative_length + 1, self.super_q_embed_dim))
        if not self.k_only:
            self.relative_position_values = Parameter(torch.Tensor(2 * self.max_relative_length + 1, self.super_q_embed_dim))

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.enable_torch_version = False

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
        nn.init.xavier_uniform_(self.relative_position_keys)
        if not self.k_only:
            nn.init.xavier_uniform_(self.relative_position_values)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        tgt_len, bsz, embed_dim = query.size()
        # assert embed_dim == self.super_q_embed_dim
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        q, k, v = self.in_proj_qkv(query)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
 
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        relative_positions_matrix = self._generate_relative_positions_matrix(
            src_len, self.max_relative_length, incremental_state
        )

        if self.k_only:
            relation_keys = F.embedding(relative_positions_matrix.long().cuda(), self.relative_position_keys)
        else:
            relation_keys = F.embedding(relative_positions_matrix.long().cuda(), self.relative_position_keys)
            relation_values = F.embedding(relative_positions_matrix.long().cuda(), self.relative_position_values)
        relation_keys = relation_keys[:,:,:self.head_dim]
        relative_attn_weights = self._relative_attention_inner(q, k, relation_keys, transpose=True)
        assert list(relative_attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(relative_attn_weights.size(0), 1, 1)
            relative_attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            relative_attn_weights = relative_attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                relative_attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    relative_attn_weights.float()
                ).type_as(relative_attn_weights)
            else:
                relative_attn_weights = relative_attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(relative_attn_weights)  # FP16 support: cast to float and back
                relative_attn_weights = relative_attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        relative_attn_weights = utils.softmax(
            relative_attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(relative_attn_weights)
        relative_attn_weights = F.dropout(relative_attn_weights, p=self.dropout, training=self.training)
        # key only mode
        if self.k_only:
            attn = torch.bmm(relative_attn_weights, v)
        # original implementation
        else:
            attn = self._relative_attention_inner(relative_attn_weights, v, relation_values, transpose=False)

        # attn = torch.bmm(relative_attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.qkv_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.qkv_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            relative_attn_weights = relative_attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            relative_attn_weights = relative_attn_weights.sum(dim=1) / self.num_heads
        else:
            relative_attn_weights = None

        return attn, relative_attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query, sample_dim=self.sample_q_embed_dim).chunk(3, dim=-1)

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

    def _generate_relative_positions_matrix(self, length, max_relative_length, incremental_state):
        if not incremental_state:
            # training process
            range_vec = torch.arange(length).cuda()
            range_mat = range_vec.repeat(length, 1)
            distance_mat = range_mat - range_mat.transpose(0, 1)
        else:
            distance_mat = torch.arange(-length + 1, 1).cuda().view(1, -1)

        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_length, max_relative_length)

        # position difference.
        final_mat = distance_mat_clipped + max_relative_length

        return final_mat

    def _relative_attention_inner(self, x, y, z, transpose=True):
        """Relative position-aware dot-product attention inner calculation.

        This batches matrix multiply calculations to avoid unnecessary broadcasting.

        Args:
          x: Tensor with shape [batch_size*heads, length, length or depth].
          y: Tensor with shape [batch_size*heads, length, depth].
          z: Tensor with shape [length, length, depth].
          transpose: Whether to tranpose inner matrices of y and z. Should be true if
              last dimension of x is depth, not length.

        Returns:
          A Tensor with shape [batch_size*heads, length, length or depth].

          wq: this function actually does 'X(Y+Z)', where Z is vector,
          but factor above formular as: 'XY + XZ'
        """
        # print(x.size())
        # print(y.size())
        # print(z.size())
        # exit(0)

        batch_size_mul_head = x.size()[0]
        length = z.size()[0]
        # print(batch_size_mul_head, length)
        # xy_matmul is [batch_size*heads, length, length or depth]
        if transpose:
            y = y.transpose(1, 2)
        xy_matmul = torch.bmm(x, y)
        # x_t is [length, batch_size * heads, length or depth]
        x_t = x.transpose(0, 1)
        # x_tz_matmul is [length, batch_size * heads, length or depth]
        if transpose:
            z = z.transpose(1, 2)

        x_tz_matmul = torch.bmm(x_t, z).transpose(0, 1).view(batch_size_mul_head, length, -1)
        attn = xy_matmul + x_tz_matmul

        return attn

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
