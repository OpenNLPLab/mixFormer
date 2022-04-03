# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.cifar_embedding_super import PatchembedSuper, trunc_normal_
from fairseq.modules.multihead_cosformer_attention import MultiheadCosformerAttention
from fairseq.modules.multihead_attention_super_qkv import MultiheadAttentionSuperqkv
from fairseq.modules.multihead_cosformer_attention_super import MultiheadCosformerAttentionSuper
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    SuperFairseqEncoderDecoderModel,
    SuperFairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiheadAttentionSuper,
    EmbeddingSuper,
    LinearSuper,
    LayerNormSuper, RelativeMultiheadAttention, LayerNorm, Linear, RelativeMultiheadAttentionSuper

)
from fairseq.modules.multihead_cosformer_attention_2d_super import MultiheadCosformerAttention2dSuper
import fairseq.init as init

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model('transformersuper_cifar10')
class VisionTransformerSuperModel(SuperFairseqEncoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off
        return {
            'transformer.wmt14.en-fr': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2',
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz',
            'transformer.wmt19.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.en-ru': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz',
            'transformer.wmt19.de-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.ru-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz',
            'transformer.wmt19.en-de.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz',
            'transformer.wmt19.en-ru.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz',
            'transformer.wmt19.de-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz',
            'transformer.wmt19.ru-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz',
        }
        # fmt: on

    def __init__(self, encoder,):
        super().__init__(encoder)
        self.encoder = encoder
    def forward(self, src_tokens, **kwargs):
        return self.encoder(src_tokens, **kwargs)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}
    
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
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--get-attn', action='store_true', default=False)
        # fmt: on
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')

        # rpr attention
        parser.add_argument('--max-relative-length', type=int,
                            help='the maximum relative length for RPR attention',
                            default=-1)
        parser.add_argument('--attn-cal-choice',
                            nargs='+', default=[1, 2, 3], type=int)
        parser.add_argument('--attn_cal', type=int, metavar='N',
                            help='way to calculate attention')
        # SuperTransformer
        # embedding dim
        parser.add_argument('--encoder-rpr-choice', nargs='+',
                            default=[16, 12, 8], type=int)
        parser.add_argument('--decoder-rpr-choice', nargs='+',
                            default=[16, 12, 8], type=int)
        parser.add_argument('--encoder-embed-choice',
                            nargs='+', default=[512, 256, 128], type=int)
        parser.add_argument('--decoder-embed-choice',
                            nargs='+', default=[512, 256, 128], type=int)

        # number of layers
        parser.add_argument('--encoder-layer-num-choice',
                            nargs='+', default=[7, 6, 5, 4, 3, 2], type=int)
        parser.add_argument('--decoder-layer-num-choice',
                            nargs='+', default=[7, 6, 5, 4, 3, 2], type=int)

        # FFN inner size
        parser.add_argument('--encoder-ffn-embed-dim-choice',
                            nargs='+', default=[4096, 3072, 2048, 1024], type=int)
        parser.add_argument('--decoder-ffn-embed-dim-choice',
                            nargs='+', default=[4096, 3072, 2048, 1024], type=int)

        # number of heads
        parser.add_argument('--encoder-self-attention-heads-choice',
                            nargs='+', default=[16, 8, 4, 2, 1], type=int)
        parser.add_argument('--decoder-self-attention-heads-choice',
                            nargs='+', default=[16, 8, 4, 2, 1], type=int)
        parser.add_argument('--decoder-ende-attention-heads-choice',
                            nargs='+', default=[16, 8, 4, 2, 1], type=int)

        # qkv dim
        parser.add_argument('--qkv-dim', type=int, default=None)

        # arbitrary-ende-attn
        parser.add_argument('--decoder-arbitrary-ende-attn-choice', nargs='+', default=[-1, 1, 2], type=int,
                            help='-1 means only attend to the last layer; 1 means attend to last two layers, 2 means attend to last three layers')

        parser.add_argument('--vocab-original-scaling',
                            action='store_true', default=False)

        # for SubTransformer
        parser.add_argument('--encoder-embed-dim-subtransformer', type=int,
                            help='subtransformer encoder embedding dimension',
                            default=None)
        parser.add_argument('--decoder-embed-dim-subtransformer', type=int,
                            help='subtransformer decoder embedding dimension',
                            default=None)

        parser.add_argument(
            '--encoder-ffn-embed-dim-all-subtransformer', nargs='+', default=None, type=int)
        parser.add_argument(
            '--decoder-ffn-embed-dim-all-subtransformer', nargs='+', default=None, type=int)

        parser.add_argument('--encoder-layer-num-subtransformer',
                            type=int, help='subtransformer num encoder layers')
        parser.add_argument('--decoder-layer-num-subtransformer',
                            type=int, help='subtransformer num decoder layers')

        parser.add_argument(
            '--encoder-self-attention-heads-all-subtransformer', nargs='+', default=None, type=int)
        parser.add_argument(
            '--encoder-attention-choices-all-subtransformer', nargs='+', default=None, type=int)
        parser.add_argument(
            '--decoder-self-attention-heads-all-subtransformer', nargs='+', default=None, type=int)
        parser.add_argument(
            '--decoder-ende-attention-heads-all-subtransformer', nargs='+', default=None, type=int)

        parser.add_argument(
            '--decoder-arbitrary-ende-attn-all-subtransformer', nargs='+', default=None, type=int)

    def profile(self, mode=True):
        for module in self.modules():
            if hasattr(module, 'profile') and self != module:
                module.profile(mode)

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):

                if name in ['encoder.layer_norm', 'decoder.layer_norm']:
                    numels.append(module.calc_sampled_param_num())
                    continue

                # a hacky way to skip the layers that exceed encoder-layer-num or decoder-layer-num
                if name == 'encoder.head':
                    numels.append(module.calc_sampled_param_num())
                    continue
                if name == 'encoder.embed_tokens':
                    numels.append(module.calc_sampled_param_num())
                    continue
                if name.split('.')[0] == 'encoder' and eval(name.split('.')[2]) >= config['encoder'][
                        'encoder_layer_num']:
                    continue
                if name.split('.')[0] == 'decoder' and eval(name.split('.')[2]) >= config['decoder'][
                        'decoder_layer_num']:
                    continue
                if name.split('.')[3] in ['self_attn_cosformer', 'self_attn_multihead', 'self_attn_cosformer2d']:
                    continue
                if name.split('.')[3] == 'self_attn_relative':
                    name_ = name.split('.')
                    if len(name_) == 4:
                        module = self.encoder.layers[int(name_[2])].self_attn
                        numels.append(module.calc_sampled_param_num())
                        continue
                    elif len(name_) == 5:
                        module = self.encoder.layers[int(name_[2])].self_attn.out_proj
                        numels.append(module.calc_sampled_param_num())
                        continue
                numels.append(module.calc_sampled_param_num())
        return sum(numels)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        fixed = args.fixed and args.train_subtransformer
        config = None
        # if fixed:
        #     config = {
        #         'encoder': {
        #             'layer': args.encoder_layer_num_subtransformer,
        #             'self_attn_heads': args.encoder_self_attention_heads_all_subtransformer,
        #             'emb_dim': args.encoder_embed_dim_subtransformer,
        #             'ffn_emb_dim': args.encoder_ffn_embed_dim_all_subtransformer
        #         },
        #     }
        #
        #     args.encoder_embed_dim = args.encoder_embed_dim_subtransformer
        #
        #     if args.share_all_embeddings:
        #         max_emb_dim = min(
        #             config['encoder']['emb_dim'])
        #         config['encoder']['emb_dim'] = max_emb_dim
        #         args.qkv_dim = max_emb_dim
        #         args.encoder_embed_dim = max_emb_dim

                # make sure all arguments are present in older models
        base_architecture(args)
        patch_embed_super = PatchembedSuper(img_size=args.input_size, patch_size=args.patch_size,
                                                 in_chans=args.in_chans, embed_dim=args.encoder_embed_dim)
        # if not hasattr(args, 'max_source_positions'):
        #     args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        # if not hasattr(args, 'max_target_positions'):
        #     args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        #src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        init.build_init(args)

        # if args.share_all_embeddings:
        #     if src_dict != tgt_dict:
        #         raise ValueError(
        #             '--share-all-embeddings requires a joined dictionary')
        #     if args.encoder_embed_dim != args.decoder_embed_dim:
        #         raise ValueError(
        #             '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
        #     if args.decoder_embed_path and (
        #             args.decoder_embed_path != args.encoder_embed_path):
        #         raise ValueError(
        #             '--share-all-embeddings not compatible with --decoder-embed-path')
        #     if fixed:
        #         encoder_embed_tokens = build_embedding(
        #             src_dict, config['encoder']['emb_dim'], args.encoder_embed_path
        #         )
        #     else:
        #         encoder_embed_tokens = build_embedding(
        #             src_dict, args.encoder_embed_dim, args.encoder_embed_path
        #         )
        #     decoder_embed_tokens = encoder_embed_tokens
        #     args.share_decoder_input_output_embed = True
        # else:
        #     if fixed:
        #         encoder_embed_tokens = build_embedding(
        #             src_dict, config['encoder']['emb_dim'], args.encoder_embed_path
        #         )
        #         decoder_embed_tokens = build_embedding(
        #             tgt_dict, config['decoder']['emb_dim'], args.decoder_embed_path
        #         )
        #     else:
        #         encoder_embed_tokens = build_embedding(
        #             src_dict, args.encoder_embed_dim, args.encoder_embed_path
        #         )
        #         decoder_embed_tokens = build_embedding(
        #             tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        #         )
        encoder = cls.build_encoder(args, patch_embed_super, config)
        return VisionTransformerSuperModel(encoder)

    @classmethod
    def build_encoder(cls, args, patch_embed_super, config):
        return TransformerEncoder(args, patch_embed_super, config=config)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, embed_tokens, config=None, dictionary=None):
        super().__init__(dictionary)
        # the configs of super arch
        self.fixed = (config is not None)
        if config is None:
            self.super_embed_dim = args.encoder_embed_dim
            self.super_ffn_embed_dim = [
                args.encoder_ffn_embed_dim] * args.encoder_layers
            self.super_layer_num = args.encoder_layers
            self.super_self_attention_heads = [
                args.encoder_attention_heads] * args.encoder_layers
        else:
            self.super_embed_dim = config['encoder']['emb_dim']
            self.super_ffn_embed_dim = config['encoder']['ffn_emb_dim']
            self.super_layer_num = config['encoder']['layer']
            self.super_self_attention_heads = config['encoder']['self_attn_heads']

        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        self.super_embed_scale = math.sqrt(self.super_embed_dim)
        dpr = [x.item() for x in torch.linspace(0, args.drop_path, self.super_layer_num)]
        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim = None
        self.sample_layer_num = None
        self.sample_self_attention_heads = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.sample_embed_scale = None

        self.register_buffer('version', torch.Tensor([3]))

        # self.dropout = args.dropout

        # embed_dim = embed_tokens.embedding_dim
        #self.padding_idx = embed_tokens.padding_idx
        self.padding_idx = None
        #self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        # self.embed_scale = math.sqrt(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_tokens.num_patches + 1, self.super_embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        # self.embed_positions = PositionalEmbedding(
        #     embed_tokens.num_patches+1, self.super_embed_dim, self.padding_idx,
        #     learned=args.encoder_learned_pos,
        # ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args, layer_idx=i, config=config, drop_path=dpr[i])
            for i in range(self.super_layer_num)
        ])

        if args.encoder_normalize_before:
            if self.fixed:
                self.layer_norm = LayerNorm(self.super_embed_dim)
            else:
                self.layer_norm = LayerNormSuper(self.super_embed_dim)
        else:
            self.layer_norm = None

        self.vocab_original_scaling = args.vocab_original_scaling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.super_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        # self.sample_scale = self.embed_scale
        self.num_classes = args.num_classes
        self.head = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.num_classes)
    def set_sample_config(self, config: dict):

        self.sample_embed_dim = config['encoder']['encoder_embed_dim']

        # Caution: this is a list for all layers
        self.sample_ffn_embed_dim = config['encoder']['encoder_ffn_embed_dim']

        self.sample_layer_num = config['encoder']['encoder_layer_num']

        # Caution: this is a list for all layers
        self.sample_self_attention_heads = config['encoder']['encoder_self_attention_heads']
        self.attn_choices = config['encoder']['encoder_attention_choices']
        self.sample_dropout = calc_dropout(
            self.super_dropout, self.sample_embed_dim, self.super_embed_dim)
        self.sample_activation_dropout = calc_dropout(self.super_activation_dropout, self.sample_embed_dim,
                                                      self.super_embed_dim)

        self.sample_embed_scale = math.sqrt(
            self.sample_embed_dim) if not self.vocab_original_scaling else self.super_embed_scale

        if not self.fixed:
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
                                        sample_activation_dropout=self.sample_activation_dropout,
                                        attn_choice_this_layer=self.attn_choices[i])
            # exceeds sample layer number
            else:
                layer.set_sample_config(is_identity_layer=True)

        self.head.set_sample_config(self.sample_embed_dim, self.num_classes)

    def forward(self, src_tokens):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        B = src_tokens.shape[0]
        x, H, W = self.embed_tokens(src_tokens)
        cls_tokens = self.cls_token[..., :self.sample_embed_dim].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.sample_embed_scale * x
        #if self.embed_positions is not None:
            # sample the positional embedding and add
        x += self.pos_embed[..., :self.sample_embed_dim]

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        # encoder_padding_mask = src_tokens.eq(self.padding_idx)
        # if not encoder_padding_mask.any():
        #     encoder_padding_mask = None
        encoder_padding_mask = None
        all_x = []
        # encoder layers
        for layer in self.layers:
            # print(x.shape)
            x = layer(x, H, W, encoder_padding_mask)
            all_x.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)
        x = torch.mean(x[:, 1:], dim=1)
        x = self.head(x)
        #return x[:, 0]
        return {
            'encoder_out': x,
            'encoder_out_all': all_x,
            'encoder_padding_mask': encoder_padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)

        # need to reorder each layer of output
        if 'encoder_out_all' in encoder_out.keys():
            new_encoder_out_all = []
            for encoder_out_one_layer in encoder_out['encoder_out_all']:
                new_encoder_out_all.append(
                    encoder_out_one_layer.index_select(1, new_order))
            encoder_out['encoder_out_all'] = new_encoder_out_all

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        # if self.embed_positions is None:
        #     return self.max_source_positions
        # return min(self.max_source_positions, self.embed_positions.max_positions())
        return 0

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.pos_embed, SinusoidalPositionalEmbedding):
            weights_key = '{}.pos_embed.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.pos_embed._float_tensor'.format(
                name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, layer_idx, add_bias_kv=False, add_zero_attn=False, config=None, drop_path=0.):
        super().__init__()

        # the configs of super arch
        self.fixed = (config is not None)
        if config is None:
            self.super_embed_dim = args.encoder_embed_dim
            self.super_ffn_embed_dim_this_layer = args.encoder_ffn_embed_dim
            self.super_self_attention_heads_this_layer = args.encoder_attention_heads
            self.attn_choice_this_layer = args.attn_cal
        else:
            self.super_embed_dim = config['encoder']['emb_dim']
            self.super_ffn_embed_dim_this_layer = config['encoder']['ffn_emb_dim'][layer_idx]
            self.super_self_attention_heads_this_layer = config['encoder']['self_attn_heads'][layer_idx]
            self.attn_choice_this_layer = config['encoder']['encoder_attention_choices'][layer_idx]
        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, 'activation_dropout', 0)
        self.change_qkv = args.change_qkv
        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_self_attention_heads_this_layer = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.is_identity_layer = None

        self.qkv_dim = args.qkv_dim
        #if args.max_relative_length == -1:

        if self.change_qkv:
            self.self_attn_cosformer = MultiheadCosformerAttentionSuper(
                    embed_dim=self.super_embed_dim,
                    num_heads=self.super_self_attention_heads_this_layer,
                    is_encoder=True,
                    dropout=args.attention_dropout,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    self_attention=True,
                    qkv_dim=self.qkv_dim,
                    is_fixed=self.fixed,
                    causal=True
                )
            self.self_attn_multihead = MultiheadAttentionSuperqkv(
                    is_encoder=True,
                    super_embed_dim=self.super_embed_dim,
                    num_heads=self.super_self_attention_heads_this_layer,
                    dropout=args.attention_dropout,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    self_attention=True,
                    qkv_dim=self.qkv_dim,
                    is_fixed=self.fixed
                )
            self.self_attn_relative = RelativeMultiheadAttentionSuper(
                    super_embed_dim=self.super_embed_dim, num_heads=self.super_self_attention_heads_this_layer,
                    is_encoder=True,
                    dropout=args.attention_dropout, self_attention=True, qkv_dim=self.qkv_dim,
                    max_relative_length=args.max_relative_length, is_fixed=self.fixed
                )
            # self.self_attn_cosformer2d = MultiheadCosformerAttention2dSuper(
            #     embed_dim=self.super_embed_dim,
            #     num_heads=self.super_self_attention_heads_this_layer,
            #     is_encoder=True,
            #     dropout=args.attention_dropout,
            #     add_bias_kv=add_bias_kv,
            #     add_zero_attn=add_zero_attn,
            #     self_attention=True,
            #     qkv_dim=self.qkv_dim,
            #     is_fixed=self.fixed,
            #     causal=True
            # )
        self.self_attn_super = {1: self.self_attn_cosformer, 2: self.self_attn_multihead, 3: self.self_attn_relative}
        self.self_attn = self.self_attn_super[self.attn_choice_this_layer]
        # if self.attn_cal_this_layer == 1:
        #     self.self_attn = MultiheadCosformerAttention(
        #         embed_dim=self.super_embed_dim,
        #         num_heads=self.super_self_attention_heads_this_layer,
        #         is_encoder=True,
        #         dropout=args.attention_dropout,
        #         add_bias_kv=add_bias_kv,
        #         add_zero_attn=add_zero_attn,
        #         self_attention=True,
        #         qkv_dim=self.qkv_dim,
        #         is_fixed=self.fixed,
        #         causal=True
        #     )
        #     if self.change_qkv:
        #         self.self_attn = MultiheadCosformerAttentionSuper(
        #             embed_dim=self.super_embed_dim,
        #             num_heads=self.super_self_attention_heads_this_layer,
        #             is_encoder=True,
        #             dropout=args.attention_dropout,
        #             add_bias_kv=add_bias_kv,
        #             add_zero_attn=add_zero_attn,
        #             self_attention=True,
        #             qkv_dim=self.qkv_dim,
        #             is_fixed=self.fixed,
        #             causal=True
        #         )
        # elif self.attn_cal_this_layer == 2:
        #     self.self_attn = MultiheadAttentionSuper(
        #         is_encoder=True,
        #         super_embed_dim=self.super_embed_dim,
        #         num_heads=self.super_self_attention_heads_this_layer,
        #         dropout=args.attention_dropout,
        #         add_bias_kv=add_bias_kv,
        #         add_zero_attn=add_zero_attn,
        #         self_attention=True,
        #         qkv_dim=self.qkv_dim,
        #         is_fixed=self.fixed
        #     )
        #     if self.change_qkv:
        #         self.self_attn = MultiheadAttentionSuperqkv(
        #             is_encoder=True,
        #             super_embed_dim=self.super_embed_dim,
        #             num_heads=self.super_self_attention_heads_this_layer,
        #             dropout=args.attention_dropout,
        #             add_bias_kv=add_bias_kv,
        #             add_zero_attn=add_zero_attn,
        #             self_attention=True,
        #             qkv_dim=self.qkv_dim,
        #             is_fixed=self.fixed
        #         )
        # elif self.attn_cal_this_layer == 3:
        #     self.self_attn = RelativeMultiheadAttention(
        #         super_embed_dim=self.super_embed_dim, num_heads=self.super_self_attention_heads_this_layer,
        #         is_encoder=True,
        #         dropout=args.attention_dropout, self_attention=True, qkv_dim=self.qkv_dim,
        #         max_relative_length=args.max_relative_length, is_fixed=self.fixed
        #     )
        #     if self.change_qkv:
        #         self.self_attn = RelativeMultiheadAttentionSuper(
        #             super_embed_dim=self.super_embed_dim, num_heads=self.super_self_attention_heads_this_layer,
        #             is_encoder=True,
        #             dropout=args.attention_dropout, self_attention=True, qkv_dim=self.qkv_dim,
        #             max_relative_length=args.max_relative_length, is_fixed=self.fixed
        #         )
        if self.fixed:
            self.self_attn_layer_norm = LayerNorm(self.super_embed_dim)
        else:
            self.self_attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.normalize_before = args.encoder_normalize_before

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

    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_ffn_embed_dim_this_layer=None,
                          sample_self_attention_heads_this_layer=None, sample_dropout=None, attn_choice_this_layer=1,
                          sample_activation_dropout=None):
        self.attn_choice_this_layer = attn_choice_this_layer
        self.self_attn = self.self_attn_super[self.attn_choice_this_layer]
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
        if self.change_qkv:
            self.self_attn.set_sample_config(sample_embed_dim=self.sample_embed_dim, sample_q_embed_dim=self.sample_self_attention_heads_this_layer*64,
                                             sample_attention_heads=self.sample_self_attention_heads_this_layer)
        else:
            self.self_attn.set_sample_config(sample_q_embed_dim=self.sample_embed_dim,
                                             sample_attention_heads=self.sample_self_attention_heads_this_layer)

        if not self.fixed:
            self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim,
                                       sample_out_dim=self.sample_ffn_embed_dim_this_layer)
            self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer,
                                       sample_out_dim=self.sample_embed_dim)

            self.final_layer_norm.set_sample_config(
                sample_embed_dim=self.sample_embed_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, H, W, encoder_padding_mask, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if self.is_identity_layer:
            return x
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.byte(), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        x, _ = self.self_attn(query=x, key=x, value=x,
                              key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.drop_path(x)
        x[:residual.size(0), :, :] = residual + x[:residual.size(0), :, :]
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_activation_dropout,
                      training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


def Embedding(num_embeddings, embedding_dim, padding_idx, fixed):
    if fixed:
        m = nn.Embedding(num_embeddings, embedding_dim,
                         padding_idx=padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
        return m
    else:
        return EmbeddingSuper(num_embeddings, embedding_dim, padding_idx=padding_idx)


@register_model_architecture('transformersuper_cifar10', 'transformersuper_cifar10')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(
        args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)

    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(
        args, 'adaptive_softmax_dropout', 0)

    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.fixed = getattr(args, 'fixed', False)
    args.max_relative_length = getattr(args, 'max_relative_length', -1)
    args.attn_cal = getattr(args, 'attn_cal', 1)


@register_model_architecture('transformersuper_cifar10', 'transformersuper_cifar10_small')
def transformer_cifar10(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture('transformersuper_cifar10', 'transformersuper_cifar10_big')
def transformer_cifar10_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(
        args, 'encoder_normalize_before', False)

    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture('transformersuper_cifar10', 'transformersuper_cifar10_big_t2t')
def transformer_cifar10_big_t2t(args):
    args.encoder_normalize_before = getattr(
        args, 'encoder_normalize_before', True)

    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    transformer_cifar10_big(args)
