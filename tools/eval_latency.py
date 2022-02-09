import os
from glob import glob

import yaml

os.environ['MKL_NUM_THREADS'] = '12'

import time
import numpy as np
import torch
import torchprofile


from fairseq import options, tasks, utils

def get_flops(args, task, model, config):
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

    return flops

def load_arch_from_file(f):
    arch_flatten = yaml.safe_load(f.read())
    arch = dict()
    arch['encoder'] = dict()
    arch['decoder'] = dict()
    for k in arch_flatten.keys():
        new_k = k.replace('-', '_')
        new_k = new_k.replace('_all_subtransformer', '')
        new_k = new_k.replace('_subtransformer', '')
        if 'encoder' in k:
            arch['encoder'][new_k] = arch_flatten[k]
        elif 'decoder' in k:
            arch['decoder'][new_k] = arch_flatten[k]
    return arch

def main(args):
    for p in (args.lat_dataset_path, args.arch_path):
        data_path = os.path.dirname(p)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    utils.import_user_module(args)

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(1)

    # Setup task
    task = tasks.setup_task(args)

    # Build model
    model = task.build_model(args)

    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
    src_lengths_test = torch.tensor([dummy_sentence_length])
    prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)
    if args.latcpu:
        model.cpu()
        print('Measuring model latency on CPU for dataset generation...')
    elif args.latgpu:
        model.cuda()
        src_tokens_test = src_tokens_test.cuda()
        src_lengths_test = src_lengths_test.cuda()
        prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam.cuda()

        src_tokens_test.get_device()
        print('Measuring model latency on GPU for dataset generation...')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    all_archs = []
    files = []
    for f in glob('configs/yml/*/*/*.yml'):
        with open(f, 'r') as fi:
            arch = load_arch_from_file(fi)
            all_archs.append(arch)
        files.append(f)


    for config_sam, file in zip(all_archs, files):
        print('-'*30)
        print(file)
        print(config_sam)

        model.set_sample_config(config_sam)

        flops = get_flops(args, task, model, config_sam)
        print(flops)

        for _ in range(5):
            encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

        encoder_latencies = []
        for _ in (range(args.latiter)):
            if args.latgpu:
                start.record()
            elif args.latcpu:
                start = time.time()

            model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

            if args.latgpu:
                end.record()
                torch.cuda.synchronize()
                encoder_latencies.append(start.elapsed_time(end))
                if not args.latsilent:
                    print('Encoder one run on GPU (for dataset generation): ', start.elapsed_time(end))

            elif args.latcpu:
                end = time.time()
                encoder_latencies.append((end - start) * 1000)
                if not args.latsilent:
                    print('Encoder one run on CPU (for dataset generation): ', (end - start) * 1000)

        encoder_latencies.sort()
        encoder_latencies = encoder_latencies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]

        bsz = 1
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
        if args.latgpu:
            new_order = new_order.cuda()

        encoder_out_test_with_beam = model.encoder.reorder_encoder_out(encoder_out_test, new_order)

        for _ in range(5):
            model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                          encoder_out=encoder_out_test_with_beam)

        decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}
        if 'iwslt' in args.arch:
            decoder_iterations = decoder_iterations_dict['iwslt']
        elif 'wmt' in args.arch:
            decoder_iterations = decoder_iterations_dict['wmt']

        decoder_latencies = []

        for _ in (range(args.latiter)):
            if args.latgpu:
                start.record()
            elif args.latcpu:
                start = time.time()
            incre_states = {}
            for k_regressive in range(decoder_iterations):
                model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam[:, :k_regressive + 1],
                              encoder_out=encoder_out_test_with_beam, incremental_state=incre_states)
            if args.latgpu:
                end.record()
                torch.cuda.synchronize()
                decoder_latencies.append(start.elapsed_time(end))
                if not args.latsilent:
                    print('Decoder one run on GPU (for dataset generation): ', start.elapsed_time(end))

            elif args.latcpu:
                end = time.time()
                decoder_latencies.append((end - start) * 1000)
                if not args.latsilent:
                    print('Decoder one run on CPU (for dataset generation): ', (end - start) * 1000)

        decoder_latencies.sort()
        decoder_latencies = decoder_latencies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]

        lats = [np.mean(encoder_latencies), np.mean(decoder_latencies), np.std(encoder_latencies),
                np.std(decoder_latencies)]
        print(lats[0] + lats[1])


def cli_main():
    parser = options.get_training_parser()

    parser.add_argument('--flops', action='store_true', help='measure SubTransformer latency on GPU')
    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer latency on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer latency on CPU')
    parser.add_argument('--latiter', type=int, default=300, help='how many iterations to run when measure the latency')
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure latency')
    parser.add_argument('--arch-path', type=str, default='./latency_dataset/arch.txt',
                        help='the path to write sampled architectures')
    parser.add_argument('--lat-dataset-path', type=str, default='./latency_dataset/lat.tmp',
                        help='the path to write latency dataset')
    parser.add_argument('--lat-dataset-size', type=int, default=200, help='number of data points for the dataset')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    if args.latcpu:
        args.cpu = True
        args.fp16 = False

    main(args)


if __name__ == '__main__':
    cli_main()
