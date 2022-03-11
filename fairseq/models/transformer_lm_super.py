# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    SuperFairseqLanguageModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)

from fairseq.modules import (
    AdaptiveInput,
    AdaptiveSoftmax,
    CharacterTokenEmbedder,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiheadAttentionSuper,
    EmbeddingSuper,
    LinearSuper,
    LayerNormSuper,
    RelativeMultiheadAttention
)

import fairseq.init as init

DEFAULT_MAX_TARGET_POSITIONS = 1024

class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, FFN) is postprocessed with: 
    `dropout -> add residual -> layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, layer_idx, add_bias_kv=False, add_zero_attn=False, config=None):
        super().__init__()

        # the configs of super arch
        self.fixed = (config is not None)
        if config is None:
            self.super_embed_dim = args.decoder_embed_dim
            self.super_ffn_embed_dim_this_layer = args.decoder_ffn_embed_dim
            self.super_self_attention_heads_this_layer = args.decoder_attention_heads
        else:
            self.super_embed_dim = config['decoder']['emb_dim']
            self.super_ffn_embed_dim_this_layer = config['decoder']['ffn_emb_dim'][layer_idx]
            self.super_self_attention_heads_this_layer = config['decoder']['self_attn_heads'][layer_idx]

        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_self_attention_heads_this_layer = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.is_identity_layer = None

        self.qkv_dim = args.qkv_dim

        self.layer_idx = layer_idx

        if args.max_relative_length == -1:
            self.self_attn = MultiheadAttentionSuper(
                is_encoder=False,
                super_embed_dim=self.super_embed_dim,
                num_heads=self.super_self_attention_heads_this_layer,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
                qkv_dim=self.qkv_dim,
                is_fixed=self.fixed
            )
        else:
            self.self_attn = RelativeMultiheadAttention(
                is_encoder=False,
                super_embed_dim=self.super_embed_dim,
                num_heads=self.super_self_attention_heads_this_layer,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
                qkv_dim=self.qkv_dim,
                max_relative_length=args.max_relative_length,
                is_fixed=self.fixed
            )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)

        if self.fixed:
            self.self_attn_layer_norm = LayerNorm(self.super_embed_dim)
        else:
            self.self_attn_layer_norm = LayerNormSuper(self.super_embed_dim)

        if self.fixed:
            self.fc1 = Linear(self.super_embed_dim, self.super_ffn_embed_dim_this_layer,
                              uniform_=init.uniform_, non_linear='relu')
            self.fc2 = Linear(self.super_ffn_embed_dim_this_layer, self.super_embed_dim,
                              uniform_=init.uniform_, non_linear='linear')
        else:
            self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer,
                                   uniform_=init.uniform_, non_linear='relu')
            self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim,
                                   uniform_=init.uniform_, non_linear='linear')

        if self.fixed:
            self.final_layer_norm = LayerNorm(self.super_embed_dim)
        else:
            self.final_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.need_attn = False
        self.onnx_trace = False

    def set_sample_config(self, is_identity_layer, sample_embed_dim=None,
                          sample_ffn_embed_dim_this_layer=None, sample_self_attention_heads_this_layer=None,
                          sample_dropout=None,
                          sample_activation_dropout=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_ffn_embed_dim_this_layer = sample_ffn_embed_dim_this_layer
        self.sample_self_attention_heads_this_layer = sample_self_attention_heads_this_layer

        self.sample_dropout = sample_dropout
        self.sample_activation_dropout = sample_activation_dropout

        if not self.fixed:
            self.self_attn_layer_norm.set_sample_config(
                sample_embed_dim=self.sample_embed_dim)

        self.self_attn.set_sample_config(sample_q_embed_dim=self.sample_embed_dim,
                                         sample_attention_heads=self.sample_self_attention_heads_this_layer)

        if not self.fixed:
            self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim,
                                       sample_out_dim=self.sample_ffn_embed_dim_this_layer)
            self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer,
                                       sample_out_dim=self.sample_embed_dim)

            self.final_layer_norm.set_sample_config(
                sample_embed_dim=self.sample_embed_dim)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
            self,
            x,
            incremental_state=None,
            prev_self_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if self.is_identity_layer:
            return x, None

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_activation_dropout,
                      training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


@register_model('transformer_lm_super')
class SuperTransformerLanguageModel(SuperFairseqLanguageModel):

    @classmethod
    def hub_models(cls):
        return {
            'transformer_lm.gbw.adaptive_huge': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2',
            'transformer_lm.wiki103.adaptive': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.tar.bz2',
            'transformer_lm.wmt19.en': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2',
            'transformer_lm.wmt19.de': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2',
            'transformer_lm.wmt19.ru': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2',
        }

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', default=4, type=int, metavar='N',
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', default=2, type=int, metavar='N',
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # fmt: on

        
        parser.add_argument('--get-attn', action='store_true', default=False)

        # SuperTransformer
        # embedding dim
        parser.add_argument('--decoder-embed-choice',
                            nargs='+', default=[512, 768, 1024], type=int)

        # number of layers
        parser.add_argument('--decoder-layer-num-choice',
                            nargs='+', default=[8, 10, 12, 14], type=int)

        # FFN inner size
        parser.add_argument('--decoder-ffn-embed-dim-choice',
                            nargs='+', default=[2048, 3072, 4096, 5120], type=int)

        # number of heads
        parser.add_argument('--decoder-self-attention-heads-choice',
                            nargs='+', default=[16, 8, 4, 2, 1], type=int)

        # qkv dim
        parser.add_argument('--qkv-dim', type=int, default=None)

        parser.add_argument('--vocab-original-scaling',
                            action='store_true', default=False)

        # for SubTransformer
        parser.add_argument('--decoder-embed-dim-subtransformer', type=int, help='subtransformer decoder embedding dimension',
                            default=None)

        parser.add_argument(
            '--decoder-ffn-embed-dim-all-subtransformer', nargs='+', default=None, type=int)

        parser.add_argument('--decoder-layer-num-subtransformer',
                            type=int, help='subtransformer num decoder layers')

        parser.add_argument(
            '--decoder-self-attention-heads-all-subtransformer', nargs='+', default=None, type=int)

        # rpr attention
        parser.add_argument('--max-relative-length', type=int,
                            help='the maximum relative length for RPR attention',
                            default=-1)

    

    def profile(self, mode=True):
        for module in self.modules():
            if hasattr(module, 'profile') and self != module:
                module.profile(mode)

    
    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                # a hacky way to skip the layers that exceed decoder-layer-num
                if name == 'decoder.embed_tokens':
                    numels.append(module.calc_sampled_param_num())
                    continue

                if name.split('.')[0] == 'decoder' and eval(name.split('.')[2]) >= config['decoder']['decoder_layer_num']:
                    continue

                numels.append(module.calc_sampled_param_num())
        return sum(numels)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = getattr(
                args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS)

        
        init.build_init(args)

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary, eval(args.character_filters),
                args.character_embedding_dim, args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary), task.source_dictionary.pad(
                ), args.decoder_input_dim,
                args.adaptive_input_factor, args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
            )
        else:
            embed_tokens = Embedding(
                len(task.source_dictionary), args.decoder_input_dim, task.source_dictionary.pad())

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = SuperTransformerDecoder(
            args, task.target_dictionary, embed_tokens
        )
        return SuperTransformerLanguageModel(decoder)


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


def Embedding(num_embeddings, embedding_dim, padding_idx):
    return EmbeddingSuper(num_embeddings, embedding_dim, padding_idx=padding_idx)


def Linear(in_features, out_features, bias=True, uniform_=None, non_linear='linear'):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight) if uniform_ is None else uniform_(
        m.weight, non_linear=non_linear)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class SuperTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)

        # the configs of super arch
        self.super_embed_dim = args.decoder_embed_dim
        self.super_ffn_embed_dim = [
            args.decoder_ffn_embed_dim] * args.decoder_layers
        self.super_layer_num = args.decoder_layers
        self.super_self_attention_heads = [
            args.decoder_attention_heads] * args.decoder_layers

        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        self.super_embed_scale = math.sqrt(self.super_embed_dim)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim = None
        self.sample_layer_num = None
        self.sample_self_attention_heads = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.sample_embed_scale = None

        # the configs of current sampled arch
        self.register_buffer('version', torch.Tensor([3]))

        self.share_input_output_embed = args.share_decoder_input_output_embed

        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, self.super_embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(
                args, layer_idx=i)
            for i in range(self.super_layer_num)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(self.super_embed_dim, self.output_embed_dim, bias=False) \
            if self.super_embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(
                len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0,
                            std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNormSuper(self.super_embed_dim)
        else:
            self.layer_norm = None
        self.get_attn = args.get_attn

        self.vocab_original_scaling = args.vocab_original_scaling

    def set_sample_config(self, config: dict):

        self.sample_embed_dim = config['decoder']['decoder_embed_dim']

        # Caution: this is a list for all layers
        self.sample_ffn_embed_dim = config['decoder']['decoder_ffn_embed_dim']

        # Caution: this is a list for all layers
        self.sample_self_attention_heads = config['decoder']['decoder_self_attention_heads']

        self.sample_layer_num = config['decoder']['decoder_layer_num']

        self.sample_dropout = calc_dropout(
            self.super_dropout, self.sample_embed_dim, self.super_embed_dim)
        self.sample_activation_dropout = calc_dropout(
            self.super_activation_dropout, self.sample_embed_dim, self.super_embed_dim)

        self.sample_embed_scale = math.sqrt(
            self.sample_embed_dim) if not self.vocab_original_scaling else self.super_embed_scale

        self.embed_tokens.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)

        if self.layer_norm is not None:
            self.layer_norm.set_sample_config(
                sample_embed_dim=self.sample_embed_dim)

        for i, layer in enumerate(self.layers):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                layer.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim,
                                        sample_ffn_embed_dim_this_layer=self.sample_ffn_embed_dim[i],
                                        sample_self_attention_heads_this_layer=self.sample_self_attention_heads[
                                            i],
                                        sample_dropout=self.sample_dropout,
                                        sample_activation_dropout=self.sample_activation_dropout)
            # exceeds sample layer number
            else:
                layer.set_sample_config(is_identity_layer=True)

    def forward(self, prev_output_tokens, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, incremental_state)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if positions is not None:
            positions = positions[..., :self.sample_embed_dim]

        if incremental_state is not None:
            # only take the last token in to the decoder
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.sample_embed_scale * self.embed_tokens(prev_output_tokens)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        attns = []
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            x, attn = layer(
                x,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(
                    x) if incremental_state is None else None,
            )
            inner_states.append(x)
            attns.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        if not self.get_attn:
            attns = attns[-1]
        return x, {'attn': attns, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.sampled_weight('decoder'))
            else:
                return F.linear(features, self.embed_out[:, :self.sample_embed_dim])
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(
                name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(
                        name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(
                            name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


@register_model_architecture('transformer_lm_super', 'transformer_lm_super')
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, 'no_tie_adaptive_proj'):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, 'decoder_final_norm'):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.adaptive_softmax_cutoff = getattr(
        args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(
        args, 'adaptive_softmax_dropout', 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')

    args.add_bos_token = getattr(args, 'add_bos_token', False)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', False)
    args.character_embeddings = getattr(args, 'character_embeddings', False)

    args.decoder_output_dim = getattr(
        args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(
        args, 'decoder_input_dim', args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)

    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)

    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)


@register_model_architecture('transformer_lm_super', 'transformer_lm_big_super')
def transformer_lm_big(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_lm_architecture(args)


@register_model_architecture('transformer_lm_super', 'transformer_lm_wiki103_super')
@register_model_architecture('transformer_lm_super', 'transformer_lm_baevski_wiki103_super')
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 16)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.dropout = getattr(args, 'dropout', 0.3)
    args.adaptive_input = getattr(args, 'adaptive_input', True)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', True)
    args.adaptive_input_cutoff = getattr(
        args, 'adaptive_input_cutoff', '20000,60000')
    args.adaptive_softmax_cutoff = getattr(
        args, 'adaptive_softmax_cutoff', '20000,60000')
    args.adaptive_softmax_dropout = getattr(
        args, 'adaptive_softmax_dropout', 0.2)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', True)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    transformer_lm_big(args)


@register_model_architecture('transformer_lm_super', 'transformer_lm_gbw')
@register_model_architecture('transformer_lm_super', 'transformer_lm_baevski_gbw')
def transformer_lm_baevski_gbw(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', True)
    transformer_lm_big(args)
