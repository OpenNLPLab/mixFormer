# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

from fairseq import options, utils
from fairseq.evolution import Evolution
from fairseq.evolution_new import EvolutionEnew
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.trainer import Trainer
from acc_dataset import Sampler
import torch
def main(args):
    utils.import_user_module(args)
    utils.handle_save_path(args)
    if args.no_rank_model:
        task = tasks.setup_task(args)

        if torch.cuda.is_available() and not args.cpu:
            torch.cuda.set_device(args.device_id)
        # Load valid dataset (we load training data below, based on the latest checkpoint)
        for valid_sub_split in args.valid_subset.split(','):
            task.load_dataset(valid_sub_split, combine=False, epoch=0)

        # Build model and criterion
        model = task.build_model(args)
        criterion = task.build_criterion(args)

        # Build trainer
        trainer = Trainer(args, task, model, criterion)
        args.train_subset = 'valid'
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

        # sample architectures and collect their losses on the validation set
        with torch.no_grad():
            sampler = Sampler(args, trainer, task, epoch_itr)
        evolver = EvolutionEnew(args, sampler)
        evolver.run_evo_search()
    # run evolutionary search to find the model with lowest loss and satisfies the latency requirement
    else:
        evolver = Evolution(args)
        evolver.run_evo_search()


def cli_main():
    parser = options.get_training_parser()
    parser.add_argument('--evo-configs', required=False, is_config_file=True)
    parser.add_argument('--evo-iter', type=int, default=30)
    parser.add_argument('--rank-list-size', type=int, default=100)
    parser.add_argument('--candidate-size', type=int, default=100)
    parser.add_argument('--population-size', type=int, default=125)
    parser.add_argument('--parent-size', type=int, default=25)
    parser.add_argument('--mutation-size', type=int, default=50)
    parser.add_argument('--crossover-size', type=int, default=50)
    parser.add_argument('--mutation-prob', type=float, default=0.3)
    parser.add_argument('--reverse', action='store_true', default=False)
    parser.add_argument('--no-rank-model', action='store_true', default=False)
    parser.add_argument('--loss-feature-list', type=int,
                        nargs='+', help='selected feature indices')
    parser.add_argument('--latency-feature-list', type=int,
                        nargs='+', help='selected feature indices')
    parser.add_argument('--loss-ranker-path', type=str,
                        help='path to the loss ranker')
    parser.add_argument('--latency-ranker-path', type=str,
                        help='path to the latency ranker')

    parser.add_argument('--latency-constraint', type=float,
                        default=-1, help='latency constraint')
    parser.add_argument('--modelSize-constraint', type=float,
                        default=-1, help='modelSize constraint')
    parser.add_argument('--valid-cnt-max', type=int, default=1e9,
                        help='max number of sentences to use in validation set')

    parser.add_argument('--write-config-path', type=str,
                        help='path to write out the searched best SubTransformer')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)
    args.distributed_world_size = 1
    main(args)


if __name__ == '__main__':
    cli_main()
