#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import glob

import torch
import os
import re


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(
                    s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    new_state['model'] = averaged_params
    return new_state


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    # fmt: off
    parser.add_argument('-path', required=True, type=str,
                        help='Input checkpoints path.')
    parser.add_argument('-number', required=True, type=int,
                        help='Number of checkpoints to ensemble')
    # fmt: on
    args = parser.parse_args()
    print(args)

    files = glob.glob(args.path + '/checkpoint[!_]*.pt')

    numbers = [int(os.path.basename(f)[10:-3]) for f in files]
    numbers.sort(reverse=False)
    numbers = numbers[-args.number:]

    files = [args.path + '/checkpoint{}.pt'.format(n) for n in numbers]

    save_path = args.path + '/' + 'averaged.pt'

    print('input files:')
    for f in files:
        print(f)

    new_state = average_checkpoints(files)
    torch.save(new_state, save_path)
    print('Finished writing averaged checkpoint to {}'.format(save_path))


if __name__ == '__main__':
    main()
