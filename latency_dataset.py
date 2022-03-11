import os
import random

import pdb
import time
from tqdm import tqdm
import numpy as np
import torch
import torchprofile

# random.seed(1)
from copy import deepcopy
from fairseq import options, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.sequence_scorer import SequenceScorer


def get_flops(args, task, model, config):
    model.set_sample_config(config)

    if args.task == 'translation':
        dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
        if 'iwslt' in args.arch:
            dummy_sentence_length = dummy_sentence_length_dict['iwslt']
        elif 'wmt' in args.arch:
            dummy_sentence_length = dummy_sentence_length_dict['wmt']
        else:
            raise NotImplementedError
        dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
        dummy_prev = [7] * (dummy_sentence_length - 1) + [2]
    elif args.task == 'language_modeling':
        dummy_sentence_length = args.max_tokens
        dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
        dummy_prev = [7] * (dummy_sentence_length - 1) + [2]
    if args.task == 'classification':
        dummy_src_tokens = torch.randn(1, 3, 224, 224)
        model.profile(mode=True)
        if args.latcpu:
            macs = torchprofile.profile_macs(model, args=(
                dummy_src_tokens))
        elif args.latgpu:
            macs = torchprofile.profile_macs(model, args=(
                dummy_src_tokens.cuda()
            ))
        model.profile(mode=False)
        flops = macs * 2
        return flops
    model.profile(mode=True)
    if args.latcpu:
        macs = torchprofile.profile_macs(model, args=(
            torch.tensor([dummy_src_tokens],
                         dtype=torch.long), torch.tensor([30]),
            torch.tensor([dummy_prev], dtype=torch.long)))
    elif args.latgpu:
        macs = torchprofile.profile_macs(model, args=(
            torch.tensor([dummy_src_tokens], dtype=torch.long).cuda(
            ), torch.tensor([30]).cuda(),
            torch.tensor([dummy_prev], dtype=torch.long).cuda()))
    model.profile(mode=False)
    last_layer_macs = config['decoder']['decoder_embed_dim'] * dummy_sentence_length * len(
        task.tgt_dict)

    flops = macs * 2
    last_layer_flops = last_layer_macs * 2

    return flops


def main(args):
    for p in (args.lat_dataset_path, args.arch_path):
        data_path = os.path.dirname(p)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    utils.import_user_module(args)

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Print args
    print(args)

    # Setup task
    task = tasks.setup_task(args)

    # Build model
    model = task.build_model(args)
    model.eval()
    print(model)

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    if args.task == 'translation':
        dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
        if 'iwslt' in args.arch:
            dummy_sentence_length = dummy_sentence_length_dict['iwslt']
        elif 'wmt' in args.arch:
            dummy_sentence_length = dummy_sentence_length_dict['wmt']
        else:
            raise NotImplementedError
    elif args.task == 'language_modeling':
        dummy_sentence_length = args.max_tokens
    if args.task != 'classification':
        dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
        dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    # for latency predictor: latency dataset generation
    # we store the sampled architectures to another file
    f_arch = open(args.arch_path, 'w')
    with open(args.lat_dataset_path, 'w') as fid:
        if args.task != 'classification':
            src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
            src_lengths_test = torch.tensor([dummy_sentence_length])
        if args.task == 'translation':
            prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)
        if args.task == 'classification':
            task = tasks.setup_task(args)
            task.load_dataset('valid')
            dataset = task.dataset('valid')
            itr = task.get_batch_iterator(
                dataset=dataset,
                max_tokens=args.max_tokens,
                max_sentences=args.max_sentences,
                max_positions=model.max_positions(),
            ).next_epoch_itr(shuffle=True)
            for s in itr:
                cls_inputs = deepcopy(s)
                break
            cls_inputs = utils.move_to_cuda(cls_inputs) if args.latgpu else cls_inputs
        if args.task == 'language_modeling':
            task = tasks.setup_task(args)

            # Load dataset splits
            task.load_dataset('test')
            dataset = task.dataset('test')
            if args.context_window > 0:
                dataset = LMContextWindowDataset(
                    dataset=dataset,
                    tokens_per_sample=args.tokens_per_sample,
                    context_window=args.context_window,
                    pad_idx=task.source_dictionary.pad(),
                )
            print('| {} {} {} examples'.format(args.data, 'test', len(dataset)))

            itr = task.get_batch_iterator(
                dataset=dataset,
                max_tokens=args.max_tokens or 2048,
                max_sentences=1,
                max_positions=model.max_positions(),
                ignore_invalid_inputs=True,
                num_workers=1,
            ).next_epoch_itr(shuffle=True)

            for s in itr:
                lm_inputs = deepcopy(s)
                break

            lm_inputs = utils.move_to_cuda(lm_inputs) if args.latgpu else lm_inputs

            scorer = SequenceScorer(task.target_dictionary, 1)

        if args.latcpu:
            model.cpu()
            print('Measuring model latency on CPU for dataset generation...')
        elif args.latgpu:
            model.cuda()
            if args.task == 'translation':
                src_tokens_test = src_tokens_test.cuda()
                src_lengths_test = src_lengths_test.cuda()
                prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam.cuda()

                src_tokens_test.get_device()
            print('Measuring model latency on GPU for dataset generation...')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

        if args.task == 'translation':
            feature_info = utils.get_feature_info()
        elif args.task == 'language_modeling':
            feature_info = utils.get_feature_info_lm()
        elif args.task == 'classification':
            feature_info = utils.get_feature_info_classification()
        fid.write(','.join(feature_info) + ',')

        if args.task == 'translation':
            latency_info = ['latency_mean_encoder', 'latency_mean_decoder',
                            'latency_std_encoder', 'latency_std_decoder']
        elif args.task == 'language_modeling':
            latency_info = ['latency_mean_decoder', 'latency_std_decoder']
        elif args.task == 'classification':
            latency_info = ['latency_mean_encoder', 'latency_std_encoder']
        fid.write(','.join(latency_info))
        if args.flops:
            fid.write(',flops')
        fid.write('\n')

        hash_table = []
        for i in tqdm(range(args.lat_dataset_size)):

            model_size = 0

            while True:
                if args.task == 'translation':
                    config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False,
                                                      super_decoder_num_layer=args.decoder_layers)
                    # store the architecture
                    f_arch.write(str(config_sam) + '\n')
                    features = utils.get_config_features(config_sam)
                    fid.write(','.join(map(str, features)) + ',')

                    if config_sam not in hash_table:
                        hash_table.append(hash_table)
                        model.set_sample_config(config_sam)
                        model_size = model.get_sampled_params_numel(config_sam)
                        break

                elif args.task == 'language_modeling':
                    config_sam = utils.sample_configs_lm(utils.get_all_choices(args), reset_rand_seed=False,
                                                         super_decoder_num_layer=args.decoder_layers)
                    # store the architecture
                    f_arch.write(str(config_sam) + '\n')
                    features = utils.get_config_features(config_sam)
                    fid.write(','.join(map(str, features)) + ',')
                    model.set_sample_config(config_sam)
                    model_size = model.get_sampled_params_numel(config_sam)

                    if config_sam not in hash_table and model_size > 235000000 and model_size < 265000000:
                        hash_table.append(hash_table)
                        break
                elif args.task == 'classification':
                    config_sam = utils.sample_configs_classification(utils.get_all_choices(args), reset_rand_seed=False,
                                                                     super_decoder_num_layer=args.decoder_layers)
                    f_arch.write(str(config_sam) + '\n')
                    features = utils.get_config_features(config_sam)
                    fid.write(','.join(map(str, features)) + ',')
                    if config_sam not in hash_table:
                        hash_table.append(hash_table)
                        model.set_sample_config(config_sam)
                        model_size = model.get_sampled_params_numel(config_sam)
                        break
            print('Sample:', config_sam)
            print('#Parameters:', model_size)

            flops = 0

            if args.flops:
                flops = get_flops(args, task, model, config_sam)

            # dry runs
            if args.task == 'classification':
                for _ in range(5):
                    encoder_out = model.forward(**cls_inputs['net_input'])

                encoder_latencies = []
                # print('Measuring encoder for dataset generation...')
                for _ in (range(args.latiter)):
                    if args.latgpu:
                        start.record()
                    elif args.latcpu:
                        start = time.time()

                    model.forward(**cls_inputs['net_input'])

                    if args.latgpu:
                        end.record()
                        torch.cuda.synchronize()
                        encoder_latencies.append(start.elapsed_time(end))
                        if not args.latsilent:
                            print(
                                'Encoder one run on GPU (for dataset generation): ', start.elapsed_time(end))
                encoder_latencies.sort()
                encoder_latencies = encoder_latencies[int(
                    args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]
                lats = [np.mean(encoder_latencies), np.std(encoder_latencies)]
                fid.write(','.join(map(str, lats)))
                if args.flops:
                    fid.write(',' + str(flops))
                fid.write('\n')
                print(features, lats)
                continue
            if args.task == 'translation':
                for _ in range(5):
                    encoder_out_test = model.encoder(
                        src_tokens=src_tokens_test, src_lengths=src_lengths_test)

                encoder_latencies = []
                # print('Measuring encoder for dataset generation...')
                for _ in (range(args.latiter)):
                    if args.latgpu:
                        start.record()
                    elif args.latcpu:
                        start = time.time()

                    model.encoder(src_tokens=src_tokens_test,
                                  src_lengths=src_lengths_test)

                    if args.latgpu:
                        end.record()
                        torch.cuda.synchronize()
                        encoder_latencies.append(start.elapsed_time(end))
                        if not args.latsilent:
                            print(
                                'Encoder one run on GPU (for dataset generation): ', start.elapsed_time(end))

                    elif args.latcpu:
                        end = time.time()
                        encoder_latencies.append((end - start) * 1000)
                        if not args.latsilent:
                            print(
                                'Encoder one run on CPU (for dataset generation): ', (end - start) * 1000)

                # only use the 10% to 90% latencies to avoid outliers
                encoder_latencies.sort()
                encoder_latencies = encoder_latencies[int(
                    args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]
                # print(f'Encoder latency for dataset generation: Mean: {np.mean(encoder_latencies)} ms; \t Std: {np.std(encoder_latencies)} ms')

                bsz = 1
                new_order = torch.arange(
                    bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
                if args.latgpu:
                    new_order = new_order.cuda()

                encoder_out_test_with_beam = model.encoder.reorder_encoder_out(
                    encoder_out_test, new_order)

                # dry runs
                for _ in range(5):
                    model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                                  encoder_out=encoder_out_test_with_beam)

            # decoder is more complicated because we need to deal with incremental states and auto regressive things
            if args.task == 'translation':
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

                if args.task == 'translation':
                    incre_states = {}
                    for k_regressive in range(decoder_iterations):
                        model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam[:, :k_regressive + 1],
                                      encoder_out=encoder_out_test_with_beam, incremental_state=incre_states)
                elif args.task == 'language_modeling':
                    decoder_out = model.forward(**lm_inputs['net_input'])

                if args.latgpu:
                    end.record()
                    torch.cuda.synchronize()
                    decoder_latencies.append(start.elapsed_time(end))
                    if not args.latsilent:
                        print(
                            'Decoder one run on GPU (for dataset generation): ', start.elapsed_time(end))

                elif args.latcpu:
                    end = time.time()
                    decoder_latencies.append((end - start) * 1000)
                    if not args.latsilent:
                        print(
                            'Decoder one run on CPU (for dataset generation): ', (end - start) * 1000)

            # only use the 10% to 90% latencies to avoid outliers
            decoder_latencies.sort()
            decoder_latencies = decoder_latencies[int(
                args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]

            if args.task == 'translation':
                lats = [np.mean(encoder_latencies), np.mean(decoder_latencies), np.std(encoder_latencies),
                        np.std(decoder_latencies)]
            elif args.task == 'language_modeling':
                lats = [np.mean(decoder_latencies), np.std(decoder_latencies)]
            fid.write(','.join(map(str, lats)))
            if args.flops:
                fid.write(',' + str(flops))
            fid.write('\n')

            print(features, lats)

            # print(config_sam)
            # print(lats[0] + lats[1], lats[-1] + lats[-2])


def cli_main():
    parser = options.get_training_parser()

    parser.add_argument('--flops', action='store_true',
                        help='measure SubTransformer latency on GPU')
    parser.add_argument('--latgpu', action='store_true',
                        help='measure SubTransformer latency on GPU')
    parser.add_argument('--latcpu', action='store_true',
                        help='measure SubTransformer latency on CPU')
    parser.add_argument('--latiter', type=int, default=300,
                        help='how many iterations to run when measure the latency')
    parser.add_argument('--latsilent', action='store_true',
                        help='keep silent when measure latency')

    parser.add_argument('--arch-path', type=str, default='./latency_dataset/arch.txt',
                        help='the path to write sampled architectures')
    parser.add_argument('--lat-dataset-path', type=str, default='./latency_dataset/lat.tmp',
                        help='the path to write latency dataset')
    parser.add_argument('--lat-dataset-size', type=int, default=200,
                        help='number of data points for the dataset')
    parser.add_argument('--thread-num', type=str, default='8',
                        help='number of threads for the cpu dataset')
    parser.add_argument('--context-window', type=int, default=2048,
                        help='window size for the input context')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    os.environ['MKL_NUM_THREADS'] = str(args.thread_num)

    if args.latcpu:
        args.cpu = True
        args.fp16 = False

    if args.pdb:
        pdb.set_trace()

    with torch.no_grad():
        main(args)


if __name__ == '__main__':
    cli_main()
