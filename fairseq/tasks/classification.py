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
from fairseq.data import data_utils, FairseqDataset, iterators, Dictionary
from . import FairseqTask, register_task
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from fairseq.data.cifar_dataset import CifarDataset


@register_task('classification')
class ClassificationTask(FairseqTask):
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
        # custom parameters
        parser.add_argument('--relative_position', action='store_true')
        parser.add_argument('--gp', action='store_true')
        parser.add_argument('--change-qkv', type=bool, default=False)
        parser.add_argument('--max_relative_position', type=int, default=14,
                            help='max distance in relative position embedding')

        # Model parameters
        parser.add_argument('--input-size', default=224, type=int)
        parser.add_argument('--nb-classes', default=0, type=int)
        parser.add_argument('--in_chans', default=3, type=int)
        parser.add_argument('--patch_size', default=16, type=int)

        parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                            help='Dropout rate (default: 0.)')
        parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                            help='Drop path rate (default: 0.1)')
        parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                            help='Drop block rate (default: None)')
        # parser.set_defaults(model_ema=True)
        parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
        parser.add_argument('--post_norm', action='store_true')
        parser.add_argument('--no_abs_pos', action='store_true')
        # Augmentation parameters
        parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                            help='Color jitter factor (default: 0.4)')
        parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + \
                                     "(default: rand-m9-mstd0.5-inc1)'),
        parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
        parser.add_argument('--train-interpolation', type=str, default='bicubic',
                            help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

        parser.add_argument('--repeated-aug', action='store_true')
        parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')

        parser.set_defaults(repeated_aug=True)

        # * Random Erase params
        parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
        parser.add_argument('--remode', type=str, default='pixel',
                            help='Random erase mode (default: "pixel")')
        parser.add_argument('--recount', type=int, default=1,
                            help='Random erase count (default: 1)')
        parser.add_argument('--resplit', action='store_true', default=False,
                            help='Do not random erase first (clean) augmentation split')

        # * Mixup params
        parser.add_argument('--mixup', type=float, default=0.8,
                            help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
        parser.add_argument('--cutmix', type=float, default=1.0,
                            help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
        parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                            help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
        parser.add_argument('--mixup-prob', type=float, default=1.0,
                            help='Probability of performing mixup or cutmix when either/both is enabled')
        parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                            help='Probability of switching to cutmix when both mixup and cutmix enabled')
        parser.add_argument('--mixup-mode', type=str, default='batch',
                            help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

        # Dataset parameters
        parser.add_argument('--data', default='./data/', type=str,
                            help='dataset path')
        parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                            type=str, help='Image Net dataset path')
        parser.add_argument('--inat-category', default='name',
                            choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                            type=str, help='semantic granularity')

        parser.add_argument('--output_dir', default='./',
                            help='path where to save, empty for no saving')
        parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
        parser.add_argument('--num_workers', default=10, type=int)
        parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
        parser.add_argument('--pin-mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
        parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                            help='')
        parser.set_defaults(pin_mem=True)

        # distributed training parameters
        parser.add_argument('--amp', action='store_true')
        parser.add_argument('--no-amp', action='store_false', dest='amp')
        parser.set_defaults(amp=True)

    def __init__(self, args):
        super().__init__(args)
        self.tgt_dict = None

    def build_generator(self, args):
        return None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)

    def build_transform(self, is_train, args):
        resize_im = args.input_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)
            return transform

        t = []
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    def get_batch_sampler(self, indices, max_sentences):
        batch = []
        for idx in indices:
            if len(batch) == max_sentences:
                yield batch[:max_sentences]
                batch = batch[max_sentences:]
            batch.append(idx)

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
             indices = dataset.ordered_indices()
        # global_rank = torch.distributed.get_rank()
        # batch_sampler = torch.utils.data.DistributedSampler(
        #     dataset, num_replicas=num_shards, rank=global_rank, shuffle=True
        # )
        batch_sampler = self.get_batch_sampler(indices, max_sentences)
        # filter examples that are too large
        # if max_positions is not None:
        #     indices = data_utils.filter_by_size(
        #         indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
        #     )
        #     indices = np.fromiter(indices, dtype=np.int64, count=-1)
        #
        # # create mini-batches with given size constraints
        # batch_sampler = data_utils.batch_by_size(
        #     indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
        #     required_batch_size_multiple=required_batch_size_multiple,
        # )

        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )


    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        args = self.args
        is_train = True if split == 'train' else False
        transform = self.build_transform(is_train, args)
        if args.data_set == 'CIFAR10':
            dataset = datasets.CIFAR10(args.data, train=is_train, transform=transform, download=True)
            nb_classes = 10
        elif args.data_set == 'CIFAR100':
            dataset = datasets.CIFAR100(args.data, train=is_train, transform=transform, download=True)
            nb_classes = 100
        elif args.data_set == 'IMNET':
            root = os.path.join(args.data, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
            nb_classes = 1000
        elif args.data_set == 'EVO_IMNET':
            root = os.path.join(args.data, folder_name)
            dataset = datasets.ImageFolder(root, transform=transform)
            nb_classes = 1000

        self.datasets[split] = CifarDataset(dataset, nb_classes)

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        #return (self.args.max_source_positions, self.args.max_target_positions)
        return (0)

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