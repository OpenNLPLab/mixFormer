# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os

import numpy as np
import torch

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    AppendTokenDataset,
    TruncateDataset,
    StripTokenDataset,
)

from . import FairseqTask, register_task


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions,
    truncate_source=False
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
    )


@register_task('translation')
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on

        # options for task-specific data augmentation
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--augmentation-schema', default='cut_off', type=str,
                            help='augmentation schema: e.g. `cut_off`, `src_cut_off`, `trg_cut_off`')
        parser.add_argument('--augmentation-merge-sample', action='store_true', default=False,
                            help='merge original and augmented samples together')
        parser.add_argument('--augmentation-masking-schema', default='word', type=str,
                            help='augmentation masking schema: e.g. `word`, `span`')
        parser.add_argument('--augmentation-masking-probability', default=0.15, type=float,
                            help='augmentation masking probability')
        parser.add_argument('--augmentation-replacing-schema', default=None, type=str,
                            help='augmentation replacing schema: e.g. `mask`, `random`, `mixed`')
        parser.add_argument("--augmentation-span-type", type=str, default='sample',
                            help="augmentation span type e.g. sample, w_sample, ws_sample, etc.")
        parser.add_argument("--augmentation-span-len-dist", default='geometric', type=str,
                            help="augmentation span length distribution e.g. geometric, poisson, etc.")
        parser.add_argument("--augmentation-max-span-len", type=int, default=10,
                            help="augmentation maximum span length")
        parser.add_argument("--augmentation-min-num-spans", type=int, default=5,
                            help="augmentation minimum number of spans")
        parser.add_argument("--augmentation-geometric-prob", type=float, default=0.2,
                            help="augmentation probability of geometric distribution.")
        parser.add_argument("--augmentation-poisson-lambda", type=float, default=5.0,
                            help="augmentation lambda of poisson distribution.")

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            truncate_source=self.args.truncate_source,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def augment_sample(self, sample):
        augmented_sample = {
            'id': sample['id'].clone(),
            'nsentences': sample['nsentences'],
            'ntokens': sample['ntokens'],
            'net_input': {
                'src_tokens': None,
                'src_lengths': sample['net_input']['src_lengths'].clone(),
                'prev_output_tokens': None,
            },
            'target': sample['target'].clone()
        }

        if self.args.augmentation_schema == 'cut_off':
            augmented_sample['net_input']['src_tokens'] = self._mask_tokens(sample['net_input']['src_tokens'],
                                                                            self.src_dict)
            augmented_sample['net_input']['prev_output_tokens'] = self._mask_tokens(
                sample['net_input']['prev_output_tokens'], self.tgt_dict)
        elif self.args.augmentation_schema == 'src_cut_off':
            augmented_sample['net_input']['src_tokens'] = self._mask_tokens(sample['net_input']['src_tokens'],
                                                                            self.src_dict)
            augmented_sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].clone()
        elif self.args.augmentation_schema == 'trg_cut_off':
            augmented_sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].clone()
            augmented_sample['net_input']['prev_output_tokens'] = self._mask_tokens(
                sample['net_input']['prev_output_tokens'], self.tgt_dict)
        else:
            raise ValueError("Augmentation schema {0} is not supported".format(self.args.augmentation_schema))

        if self.args.augmentation_merge_sample:
            sample = {
                'id': torch.cat((sample['id'], augmented_sample['id']), dim=0),
                'nsentences': sample['nsentences'] + augmented_sample['nsentences'],
                'ntokens': sample['ntokens'] + augmented_sample['ntokens'],
                'net_input': {
                    'src_tokens': torch.cat(
                        (sample['net_input']['src_tokens'], augmented_sample['net_input']['src_tokens']), dim=0),
                    'src_lengths': torch.cat(
                        (sample['net_input']['src_lengths'], augmented_sample['net_input']['src_lengths']), dim=0),
                    'prev_output_tokens': torch.cat((sample['net_input']['prev_output_tokens'],
                                                     augmented_sample['net_input']['prev_output_tokens']), dim=0),
                },
                'target': torch.cat((sample['target'], augmented_sample['target']), dim=0)
            }
        else:
            sample = {
                'primary': sample,
                'secondary': augmented_sample,
            }

        return sample

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _mask_tokens(self, inputs, vocab_dict):
        if self.args.augmentation_masking_schema == 'word':
            masked_inputs = self._mask_tokens_by_word(inputs, vocab_dict)
        elif self.args.augmentation_masking_schema == 'span':
            masked_inputs = self._mask_tokens_by_span(inputs, vocab_dict)
        else:
            raise ValueError("The masking schema {0} is not supported".format(self.args.augmentation_masking_schema))

        return masked_inputs

    def _mask_tokens_by_word(self, inputs, vocab_dict):
        vocab_size = len(vocab_dict)
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()

        available_token_indices = (inputs != bos_index) & (inputs != eos_index) & (inputs != pad_index) & (
                    inputs != unk_index)
        random_masking_indices = torch.bernoulli(
            torch.full(inputs.shape, self.args.augmentation_masking_probability, device=inputs.device)).bool()

        masked_inputs = inputs.clone()
        masking_indices = random_masking_indices & available_token_indices
        self._replace_token(masked_inputs, masking_indices, unk_index, vocab_size)

        return masked_inputs

    def _mask_tokens_by_span(self, inputs, vocab_dict):
        vocab_size = len(vocab_dict)
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()

        span_info_list = self._generate_spans(inputs)

        num_spans = len(span_info_list)
        masked_span_list = np.random.binomial(1, self.args.augmentation_masking_probability, size=num_spans).astype(
            bool)
        masked_span_list = [span_info_list[i] for i, masked in enumerate(masked_span_list) if masked]

        available_token_indices = (inputs != bos_index) & (inputs != eos_index) & (inputs != pad_index) & (
                    inputs != unk_index)
        random_masking_indices = torch.zeros_like(inputs)
        for batch_index, seq_index, span_length in masked_span_list:
            random_masking_indices[batch_index, seq_index:seq_index + span_length] = 1

        masked_inputs = inputs.clone()
        masking_indices = random_masking_indices.bool() & available_token_indices
        self._replace_token(masked_inputs, masking_indices, unk_index, vocab_size)

        return masked_inputs

    def _sample_span_length(self, span_len_dist, max_span_len, geometric_prob=0.2, poisson_lambda=5.0):
        if span_len_dist == 'geometric':
            span_length = min(np.random.geometric(geometric_prob) + 1, max_span_len)
        elif span_len_dist == 'poisson':
            span_length = min(np.random.poisson(poisson_lambda) + 1, max_span_len)
        else:
            span_length = np.random.randint(max_span_len) + 1

        return span_length

    def _get_default_spans(self, batch_index, seq_length, num_spans):
        span_length = int((seq_length - 2) / num_spans)
        last_span_length = seq_length - 2 - (num_spans - 1) * span_length
        span_infos = []
        for i in range(num_spans):
            span_info = (batch_index, 1 + i * span_length, span_length if i < num_spans - 1 else last_span_length)
            span_infos.append(span_info)

        return span_infos

    def _generate_spans(self, inputs):
        if self.args.augmentation_span_type == 'sample':
            span_info_list = self._generate_spans_by_sample(inputs)
        elif self.args.augmentation_span_type == 'w_sample':
            span_info_list = self._generate_spans_by_w_sample(inputs)
        elif self.args.augmentation_span_type == 'ws_sample':
            span_info_list = self._generate_spans_by_ws_sample(inputs)
        else:
            raise ValueError("Span type {0} is not supported".format(self.args.augmentation_span_type))

        return span_info_list

    def _generate_spans_by_sample(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]

        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index <= max_index:
                span_length = self._sample_span_length(self.args.augmentation_span_len_dist,
                                                       self.args.augmentation_max_span_len,
                                                       self.args.augmentation_geometric_prob,
                                                       self.args.augmentation_poisson_lambda)
                span_length = min(span_length, max_index - seq_index + 1)

                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.args.augmentation_min_num_spans:
                span_infos = self._get_default_spans(batch_index, seq_length, self.args.augmentation_min_num_spans)

            span_info_list.extend(span_infos)

        return span_info_list

    def _generate_spans_by_w_sample(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]
        input_words = ((inputs & ((1 << 25) - 1)) >> 16) - 1

        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index < max_index:
                span_length = self._sample_span_length(self.args.augmentation_span_len_dist,
                                                       self.args.augmentation_max_span_len,
                                                       self.args.augmentation_geometric_prob,
                                                       self.args.augmentation_poisson_lambda)
                span_length = min(span_length, max_index - seq_index + 1)

                word_id = input_words[batch_index, seq_index + span_length - 1]
                if word_id >= 0:
                    word_index = (input_words[batch_index, :] == word_id + 1).nonzero().squeeze(-1)
                    span_length = (word_index[
                                       0] - seq_index).item() if word_index.nelement() > 0 else max_index - seq_index + 1

                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.args.augmentation_min_num_spans:
                span_infos = self._get_default_spans(batch_index, seq_length, self.args.augmentation_min_num_spans)

            span_info_list.extend(span_infos)

        return span_info_list

    def _generate_spans_by_ws_sample(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]
        input_segments = (inputs >> 25) - 1
        input_words = ((inputs & ((1 << 25) - 1)) >> 16) - 1

        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index < max_index:
                span_length = self._sample_span_length(self.args.augmentation_span_len_dist,
                                                       self.args.augmentation_max_span_len,
                                                       self.args.augmentation_geometric_prob,
                                                       self.args.augmentation_poisson_lambda)
                span_length = min(span_length, max_index - seq_index + 1)

                segment_start_id = input_segments[batch_index, seq_index]
                segment_end_id = input_segments[batch_index, seq_index + span_length - 1]
                if segment_start_id != segment_end_id:
                    segment_index = (input_segments[batch_index, :] == segment_start_id).nonzero().squeeze(-1)
                    span_length = (segment_index[-1] - seq_index + 1).item()

                word_id = input_words[batch_index, seq_index + span_length - 1]
                if word_id >= 0:
                    word_index = (input_words[batch_index, :] == word_id + 1).nonzero().squeeze(-1)
                    span_length = (word_index[
                                       0] - seq_index).item() if word_index.nelement() > 0 else max_index - seq_index + 1

                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.args.augmentation_min_num_spans:
                span_infos = self._get_default_spans(batch_index, seq_length, self.args.augmentation_min_num_spans)

            span_info_list.extend(span_infos)

        return span_info_list

    def _replace_token(self, inputs, masking_indices, mask_index, vocab_size):
        if self.args.augmentation_replacing_schema == 'mask':
            inputs[masking_indices] = mask_index
        elif self.args.augmentation_replacing_schema == 'random':
            random_words = torch.randint(vocab_size, inputs.shape, device=inputs.device, dtype=torch.long)
            inputs[masking_indices] = random_words[masking_indices]
        elif self.args.augmentation_replacing_schema == 'mixed':
            # 80% of the time, we replace masked input tokens with <unk> token
            mask_token_indices = torch.bernoulli(
                torch.full(inputs.shape, 0.8, device=inputs.device)).bool() & masking_indices
            inputs[mask_token_indices] = mask_index

            # 10% of the time, we replace masked input tokens with random word
            random_token_indices = torch.bernoulli(
                torch.full(inputs.shape, 0.5, device=inputs.device)).bool() & masking_indices & ~mask_token_indices
            random_words = torch.randint(vocab_size, inputs.shape, device=inputs.device, dtype=torch.long)
            inputs[random_token_indices] = random_words[random_token_indices]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        else:
            raise ValueError(
                "The replacing schema: {0} is not supported. Only support ['mask', 'random', 'mixed']".format(
                    self.args.augmentation_replacing_schema))