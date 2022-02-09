import os

os.environ['MKL_NUM_THREADS'] = '6'

import pdb
import random
import time
from tqdm import tqdm
import numpy as np
import torch

from fairseq import options, tasks, utils


def main(args):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task
    task = tasks.setup_task(args)

    # Build model
    model = task.build_model(args)

    with open(args.arch_path, 'r') as fi:
        if args.latgpu:
            model.cuda()
        if args.latcpu:
            model.cpu()

        fo = open(args.flops_path, 'w')
        fi = fi.readlines()
        for l in tqdm(fi):
            config = eval(l[:-1])
            model.set_sample_config(config)

            dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
            if 'iwslt' in args.arch:
                dummy_sentence_length = dummy_sentence_length_dict['iwslt']
            elif 'wmt' in args.arch:
                dummy_sentence_length = dummy_sentence_length_dict['wmt']
            else:
                raise NotImplementedError

            dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
            dummy_prev = [7] * (dummy_sentence_length - 1) + [2]
            import torchprofile
            model.profile(mode=True)
            if args.latcpu:
                macs = torchprofile.profile_macs(model, args=(
                    torch.tensor([dummy_src_tokens], dtype=torch.long), torch.tensor([30]),
                    torch.tensor([dummy_prev], dtype=torch.long)))
            elif args.latgpu:
                macs = torchprofile.profile_macs(model, args=(
                    torch.tensor([dummy_src_tokens], dtype=torch.long).cuda(), torch.tensor([30]).cuda(),
                    torch.tensor([dummy_prev], dtype=torch.long).cuda()))
            model.profile(mode=False)
            last_layer_macs = config['decoder']['decoder_embed_dim'] * dummy_sentence_length * len(
                task.tgt_dict)

            flops = macs * 2
            last_layer_flops = last_layer_macs * 2

            fo.write(str(flops) + '\n')


def cli_main():
    parser = options.get_training_parser()

    parser.add_argument('--flops', action='store_true',
                        help='measure the FLOPs of SubTransformers')
    parser.add_argument('--measure-loss', action='store_true',
                        help='measure the loss of SubTransformers')
    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer latency on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer latency on CPU')
    parser.add_argument('--latiter', type=int, default=20, help='how many iterations to run when measure the latency')
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure latency')

    parser.add_argument('--lat-dataset-path', type=str, default='./latency_dataset/lat.tmp',
                        help='the path to write latency dataset')
    parser.add_argument('--lat-dataset-size', type=int, default=10000, help='number of data points for the dataset')
    parser.add_argument('--valid-iter', type=int, default=20, help='max number of sentences to use in validation set')
    parser.add_argument('--arch-path', type=str, default='./latency_dataset/lat.tmp',
                        help='the path to write latency dataset')
    parser.add_argument('--flops-path', type=str, default='./latency_dataset/lat.tmp',
                        help='the path to write latency dataset')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    if args.latcpu:
        args.cpu = True
        args.fp16 = False
        args.valid_iter = 20
    else:
        args.valid_iter = 20

    if args.pdb:
        pdb.set_trace()

    main(args)


if __name__ == '__main__':
    cli_main()
