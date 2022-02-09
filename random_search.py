import os
import random
import time

import configargparse
from tqdm import tqdm

from fairseq import utils
from ranker import PerformanceRanker

search_space = {
    'iwslt14-large': {
        'encoder': {
            'encoder_embed_dim': [640, 512],
            'encoder_layer_num': [6],
            'encoder_ffn_embed_dim': [2048, 1536, 1024, 768],
            'encoder_self_attention_heads': [8, 4, 2],
        },
        'decoder': {
            'decoder_embed_dim': [640, 512],
            'decoder_layer_num': [6, 5, 4, 3, 2, 1],
            'decoder_ffn_embed_dim': [2048, 1536, 1024, 768],
            'decoder_self_attention_heads': [8, 4, 2],
            'decoder_ende_attention_heads': [8, 4, 2],
            'decoder_arbitrary_ende_attn': [-1, 1, 2]
        }
    },
    'iwslt14-hat': {
        'encoder': {
            'encoder_embed_dim': [640, 512],
            'encoder_layer_num': [6],
            'encoder_ffn_embed_dim': [2048, 1024, 512],
            'encoder_self_attention_heads': [4, 2],
        },
        'decoder': {
            'decoder_embed_dim': [640, 512],
            'decoder_layer_num': [6, 5, 4, 3, 2, 1],
            'decoder_ffn_embed_dim': [2048, 1024, 512],
            'decoder_self_attention_heads': [4, 2],
            'decoder_ende_attention_heads': [4, 2],
            'decoder_arbitrary_ende_attn': [-1, 1, 2]
        }
    },
    'wmt14.en-de': {
        'encoder': {
            'encoder_embed_dim': [640, 512],
            'encoder_layer_num': [6],
            'encoder_ffn_embed_dim': [3072, 2048, 1024],
            'encoder_self_attention_heads': [8, 4],
        },
        'decoder': {
            'decoder_embed_dim': [640, 512],
            'decoder_layer_num': [6, 5, 4, 3, 2, 1],
            'decoder_ffn_embed_dim': [3072, 2048, 1024],
            'decoder_self_attention_heads': [8, 4],
            'decoder_ende_attention_heads': [8, 4],
            'decoder_arbitrary_ende_attn': [-1, 1, 2]
        }
    },
    'language_modeling': {
        'decoder': {
            'decoder_embed_dim': [1152],
            'decoder_layer_num': [14, 12, 10],
            'decoder_ffn_embed_dim': [5120, 4096, 3072],
            'decoder_self_attention_heads': [16, 12, 8],
        }
    },
}


def arch_to_config(arch: dict):
    config = ''
    if 'encoder' in arch:
        modules = [arch['encoder'], arch['decoder']]
    else:
        modules = [arch['decoder']]
    for module in modules:
        for k in module:
            new_k = k.replace('_', '-')
            if 'ffn' in k or 'att' in k:
                new_k += '-all-subtransformer'
            else:
                new_k += '-subtransformer'
            config += '\n{}: {}\n'.format(new_k, module[k])
    return config


class RandomSearcher(object):
    def __init__(self, args):
        self.args = args

        self.loss_ranker = PerformanceRanker(
            feature_list=args.loss_feature_list)
        self.loss_ranker.load(args.loss_ranker_path)
        self.latency_ranker = PerformanceRanker(
            feature_list=args.latency_feature_list)
        self.latency_ranker.load(args.latency_ranker_path)

    def scoring(self, archs, features):
        losses = self.loss_ranker.predict(features).tolist()
        latencies = self.latency_ranker.predict(features).tolist()
        all_candidates = list(zip(archs, losses, latencies))

        valid_candidates = [
            cand for cand in all_candidates if cand[2] < self.args.latency_constraint
        ]
        valid_candidates.sort(key=lambda x: x[1])
        valid_candidates.sort(key=lambda x: x[2], reverse=True)

        return valid_candidates

    def search(self):
        start_time = time.time()

        samples, features = [], []

        for i in tqdm(range(self.args.candidate_size)):
            sample = utils.sample_configs(
                search_space[self.args.task],
                reset_rand_seed=False)
            samples.append(sample)
            features.append(utils.get_config_features(sample))

        arch_scores = self.scoring(samples, features)

        if not os.path.exists(self.args.write_config_path):
            os.makedirs(self.args.write_config_path)
        for arch_score in arch_scores[:self.args.topk]:
            file = self.args.write_config_path + \
                '/loss_{}_{}.yml'.format(arch_score[1], arch_score[2])
            i = 1
            while os.path.exists(file):
                file = file.replace('.yml', '_{}.yml'.format(i))
                i += 1
            print('store the search result to:', file)
            with open(file, 'w') as fo:
                fo.write(arch_to_config(arch_score[0]))
                print(arch_score[0])

        print('search cost:', time.time() - start_time, 'seconds')


def main(args):
    searcher = RandomSearcher(args)
    searcher.search()


def cli_main():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--candidate-size', type=int, default=100)
    parser.add_argument('--latency-constraint', type=int, default=200)
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--task', type=str, default='language_modeling')

    parser.add_argument('--search-configs', required=False,
                        is_config_file=True)

    parser.add_argument('--loss-feature-list', type=int,
                        nargs='+', help='selected feature indices')
    parser.add_argument('--latency-feature-list', type=int,
                        nargs='+', help='selected feature indices')
    parser.add_argument('--loss-ranker-path', type=str,
                        help='path to the loss ranker')
    parser.add_argument('--latency-ranker-path', type=str,
                        help='path to the latency ranker')
    parser.add_argument('--write-config-path', type=str,
                        help='path to write out the searched best SubTransformer')

    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    cli_main()
