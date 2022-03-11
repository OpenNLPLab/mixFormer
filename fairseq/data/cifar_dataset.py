# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import data_utils, FairseqDataset


def collate(
    samples
):
    if len(samples) == 0:
        return {}

    # def merge(key, left_pad, move_eos_to_beginning=False):
    #     return data_utils.collate_tokens(
    #         [s[key] for s in samples],
    #         pad_idx, eos_idx, left_pad, move_eos_to_beginning,
    #     )
    def merge(key):
        src_tokens = [s[key] for s in samples]
        C, H, W = src_tokens[0].shape
        return torch.cat(src_tokens, dim=0).reshape(-1, C, H, W)
    id = torch.LongTensor([s['id'] for s in samples])
    target = torch.LongTensor([s['target'] for s in samples])
    src_tokens = merge('source')
    # sort by descending source length
    # src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    # src_lengths, sort_order = src_lengths.sort(descending=True)
    # id = id.index_select(0, sort_order)
    # src_tokens = src_tokens.index_select(0, sort_order)

    # prev_output_tokens = None
    # target = None
    # if samples[0].get('target', None) is not None:
    #     target = merge('target', left_pad=left_pad_target)
    #     target = target.index_select(0, sort_order)
    #     ntokens = sum(len(s['target']) for s in samples)
    #
    #     if input_feeding:
    #         # we create a shifted version of targets for feeding the
    #         # previous output token(s) into the next decoder step
    #         prev_output_tokens = merge(
    #             'target',
    #             left_pad=left_pad_target,
    #             move_eos_to_beginning=True,
    #         )
    #         prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    # else:
    #     ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': len(samples),
        'net_input': {
            'src_tokens': src_tokens,
        },
        'target': target,
    }
    # if prev_output_tokens is not None:
    #     batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class CifarDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, dataset, nb_classes,
        shuffle=True,
    ):
        self.src = dataset
        self.shuffle = shuffle
        self.nb_classes = nb_classes
    def __getitem__(self, index):
        src_item, tgt_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(samples)

    # def num_tokens(self, index):
    #     """Return the number of tokens in a sample. This value is used to
    #     enforce ``--max-tokens`` during batching."""
    #     return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    # def size(self, index):
    #     """Return an example's size as a float or tuple. This value is used when
    #     filtering a dataset with ``--max-positions``."""
    #     return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
