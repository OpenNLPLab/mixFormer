# This script collects architectures and their validation losses with a supernet

import os.path
import random
from glob import glob

import torch
import yaml
from tqdm import tqdm

import pdb
import time

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.trainer import Trainer


class Sampler(object):
    def __init__(self, args, trainer, task, epoch_itr):
        self.args = args
        self.trainer = trainer
        self.task = task
        self.epoch_iter = epoch_itr

    def scoring(self, sample):

        def get_itr():
            self.args.max_tokens_valid = 4096 * 2
            itr = self.task.get_batch_iterator(
                dataset=self.task.dataset('valid'),
                max_tokens=self.args.max_tokens_valid,
                max_sentences=6,
                max_positions=utils.resolve_max_positions(
                    self.task.max_positions(),
                    self.trainer.get_model().max_positions(),
                ),
                num_workers=8,
            ).next_epoch_itr(shuffle=False)

            return itr

        self.trainer.set_sample_config(sample)
        progress = get_itr()

        # reset validation loss meters
        for k in ['valid_acc1', 'vali d_acc5']:
            meter = self.trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        valid_cnt = 0

        for s in tqdm(progress):
            valid_cnt += 1
            if valid_cnt > self.args.valid_cnt_max:
                break
            self.trainer.valid_step(s)
        return self.trainer.get_meter('valid_acc1').avg

    def prepare_candidates(self, candidate_size):

        samples = []
        for i in range(candidate_size * 1):
            sample = utils.sample_configs(
                utils.get_all_choices(self.args),
                reset_rand_seed=False,
                super_decoder_num_layer=self.args.decoder_layers
            )
            samples.append(sample)
        return samples

    def collect_losses(self):

        data_path = os.path.dirname(self.args.data_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        f_arch_loss = open(self.args.data_path, 'w')

        start_time = time.time()

        candidates = self.prepare_candidates(self.args.candidate_size)

        # the first line
        # if self.args.task == 'translation':
        #     feature_info = utils.get_feature_info()
        # elif self.args.task == 'language_modeling':
        #     feature_info = utils.get_feature_info_lm()
        feature_info = utils.get_feature_info_classification()
        f_arch_loss.write(','.join(feature_info) + ',')
        f_arch_loss.write('loss\n')

        # other lines
        for sample in tqdm(candidates):
            score = self.scoring(sample)
            features = utils.get_config_features(sample)
            f_arch_loss.write(','.join(map(str, features)) + ',')
            f_arch_loss.write(str(score))
            f_arch_loss.write('\n')

        f_arch_loss.close()

        end_time = time.time()
        print('duration:', end_time - start_time, 'seconds')


def main(args):
    utils.import_user_module(args)
    utils.handle_save_path(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)

    # numpy.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)

    # Build trainer
    trainer = Trainer(args, task, model, criterion)

    # Load the latest checkpoint if one is available and restore the corresponding train iterator
    # no need to train, so just set a small subset to save loading time
    args.train_subset = 'valid'
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # sample architectures and collect their losses on the validation set
    with torch.no_grad():
        sampler = Sampler(args, trainer, task, epoch_itr)
        if not args.eval_only:
            sampler.collect_losses()


def load_arch_from_file(f):
    arch_flatten = yaml.safe_load(f.read())
    arch = dict()
    encoder = dict()
    decoder = dict()
    for k in arch_flatten.keys():
        new_k = k.replace('-', '_')
        new_k = new_k.replace('_all_subtransformer', '')
        new_k = new_k.replace('_subtransformer', '')
        if 'encoder' in k:
            encoder[new_k] = arch_flatten[k]
        elif 'decoder' in k:
            decoder[new_k] = arch_flatten[k]
    if len(encoder.keys()) > 0:
        arch['encoder'] = encoder
    if len(decoder.keys()) > 0:
        arch['decoder'] = decoder
    return arch


def cli_main():
    parser = options.get_training_parser()

    parser.add_argument('--eval-only', action='store_true',
                        help='whether we only evaluate some archs')
    parser.add_argument('--data-path', type=str, required=False,
                        help='path to store the loss data')
    parser.add_argument('--candidate-size', type=int, default=5000)
    parser.add_argument('--valid-cnt-max', type=int, default=1e9,
                        help='max number of sentences to use in validation set')

    parser.add_argument('--write-config-path', type=str,
                        help='path to write out the searched best SubTransformer')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    # random.seed(args.seed)

    if not os.path.isfile(args.restore_file):
        print('failed to load the model file')
        exit(0)

    if args.pdb:
        pdb.set_trace()

    # one GPU is fast enough to do the search
    args.distributed_world_size = 1

    main(args)


if __name__ == '__main__':
    cli_main()
