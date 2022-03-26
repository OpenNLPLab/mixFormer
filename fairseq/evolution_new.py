# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import os
import random

import numpy as np
import fairseq.utils as utils

from ranker import PerformanceRanker


class Converter(object):
    def __init__(self, args):
        self.args = args

        if self.args.task == 'translation' or self.args.task == 'classification':
            self.super_encoder_layer_num = args.encoder_layers
            self.encoder_embed_choice = args.encoder_embed_choice
            self.encoder_layer_num_choice = args.encoder_layer_num_choice
            self.encoder_ffn_embed_dim_choice = args.encoder_ffn_embed_dim_choice
            self.encoder_self_attention_heads_choice = args.encoder_self_attention_heads_choice
            self.encoder_attention_choices = args.attn_cal_choice
            self.decoder_ende_attention_heads_choice = args.decoder_ende_attention_heads_choice
            self.decoder_arbitrary_ende_attn_choice = args.decoder_arbitrary_ende_attn_choice
        if self.args.task == 'translation' or self.args.task == 'language_modeling':
            self.super_decoder_layer_num = args.decoder_layers
            self.decoder_embed_choice = args.decoder_embed_choice
            self.decoder_layer_num_choice = args.decoder_layer_num_choice
            self.decoder_ffn_embed_dim_choice = args.decoder_ffn_embed_dim_choice
            self.decoder_self_attention_heads_choice = args.decoder_self_attention_heads_choice

    def config2gene(self, config):
        gene = []

        if self.args.task == 'translation' or self.args.task == 'classification':
            sample_encoder_layer_num = config['encoder']['encoder_layer_num']
            gene.append(config['encoder']['encoder_embed_dim'])
            gene.append(sample_encoder_layer_num)

            for i in range(self.super_encoder_layer_num):
                if i < sample_encoder_layer_num:
                    gene.append(config['encoder']['encoder_ffn_embed_dim'][i])
                else:
                    gene.append(config['encoder']['encoder_ffn_embed_dim'][0])

            for i in range(self.super_encoder_layer_num):
                if i < sample_encoder_layer_num:
                    gene.append(config['encoder']
                                ['encoder_self_attention_heads'][i])
                else:
                    gene.append(config['encoder']
                                ['encoder_self_attention_heads'][0])
            for i in range(self.super_encoder_layer_num):
                if i < sample_encoder_layer_num:
                    gene.append(config['encoder']
                                ['encoder_attention_choices'][i])
                else:
                    gene.append(config['encoder']
                                ['encoder_attention_choices'][0])

        if self.args.task == 'translation' or self.args.task == 'language_modeling':
            sample_decoder_layer_num = config['decoder']['decoder_layer_num']
            gene.append(config['decoder']['decoder_embed_dim'])
            gene.append(sample_decoder_layer_num)

            for i in range(self.super_decoder_layer_num):
                if i < sample_decoder_layer_num:
                    gene.append(config['decoder']['decoder_ffn_embed_dim'][i])
                else:
                    gene.append(config['decoder']['decoder_ffn_embed_dim'][0])

            for i in range(self.super_decoder_layer_num):
                if i < sample_decoder_layer_num:
                    gene.append(config['decoder']
                                ['decoder_self_attention_heads'][i])
                else:
                    gene.append(config['decoder']
                                ['decoder_self_attention_heads'][0])

        if self.args.task == 'translation':
            for i in range(self.super_decoder_layer_num):
                if i < sample_decoder_layer_num:
                    gene.append(config['decoder']
                                ['decoder_ende_attention_heads'][i])
                else:
                    gene.append(config['decoder']
                                ['decoder_ende_attention_heads'][0])

            for i in range(self.super_decoder_layer_num):
                gene.append(config['decoder']['decoder_arbitrary_ende_attn'][i])

        return gene

    def gene2config(self, gene):

        current_index = 0

        if self.args.task == 'translation':
            config = {
                'encoder': {
                    'encoder_embed_dim': None,
                    'encoder_layer_num': None,
                    'encoder_ffn_embed_dim': None,
                    'encoder_self_attention_heads': None,
                },
                'decoder': {
                    'decoder_embed_dim': None,
                    'decoder_layer_num': None,
                    'decoder_ffn_embed_dim': None,
                    'decoder_self_attention_heads': None,
                    'decoder_ende_attention_heads': None,
                    'decoder_arbitrary_ende_attn': None
                }
            }
        elif self.args.task == 'language_modeling':
            config = {
                'decoder': {
                    'decoder_embed_dim': None,
                    'decoder_layer_num': None,
                    'decoder_ffn_embed_dim': None,
                    'decoder_self_attention_heads': None,
                }
            }
        elif self.args.task == 'classification':
            config = {
                'encoder': {
                    'encoder_embed_dim': None,
                    'encoder_layer_num': None,
                    'encoder_ffn_embed_dim': None,
                    'encoder_self_attention_heads': None,
                    'encoder_attention_choices': None
                }
            }

        if self.args.task == 'translation' or self.args.task == 'classification':
            config['encoder']['encoder_embed_dim'] = gene[current_index]
            current_index += 1

            config['encoder']['encoder_layer_num'] = gene[current_index]
            current_index += 1

            config['encoder']['encoder_ffn_embed_dim'] = gene[current_index:
                                                              current_index + self.super_encoder_layer_num]
            current_index += self.super_encoder_layer_num

            config['encoder']['encoder_self_attention_heads'] = gene[current_index:
                                                                     current_index + self.super_encoder_layer_num]
            current_index += self.super_encoder_layer_num
            config['encoder']['encoder_attention_choices'] = gene[current_index:
                                                                  current_index + self.super_encoder_layer_num]

        if self.args.task == 'translation' or self.args.task == 'language_modeling':
            config['decoder']['decoder_embed_dim'] = gene[current_index]
            current_index += 1

            config['decoder']['decoder_layer_num'] = gene[current_index]
            current_index += 1

            config['decoder']['decoder_ffn_embed_dim'] = gene[current_index:
                                                              current_index + self.super_decoder_layer_num]
            current_index += self.super_decoder_layer_num

            config['decoder']['decoder_self_attention_heads'] = gene[current_index:
                                                                     current_index + self.super_decoder_layer_num]
            current_index += self.super_decoder_layer_num

        if self.args.task == 'translation':
            config['decoder']['decoder_ende_attention_heads'] = gene[current_index:
                                                                     current_index + self.super_decoder_layer_num]
            current_index += self.super_decoder_layer_num

            config['decoder']['decoder_arbitrary_ende_attn'] = gene[current_index:
                                                                    current_index + self.super_decoder_layer_num]

        return config

    def get_gene_choice(self):
        gene_choice = []
        if self.args.task == 'translation' or self.args.task == 'classification':
            gene_choice.append(self.encoder_embed_choice)
            gene_choice.append(self.encoder_layer_num_choice)

            for i in range(self.super_encoder_layer_num):
                gene_choice.append(self.encoder_ffn_embed_dim_choice)

            for i in range(self.super_encoder_layer_num):
                gene_choice.append(self.encoder_self_attention_heads_choice)
            for i in range(self.super_encoder_layer_num):
                gene_choice.append(self.encoder_attention_choices)
        if self.args.task == 'translation' or self.args.task == 'language_modeling':
            gene_choice.append(self.decoder_embed_choice)
            gene_choice.append(self.decoder_layer_num_choice)

            for i in range(self.super_decoder_layer_num):
                gene_choice.append(self.decoder_ffn_embed_dim_choice)

            for i in range(self.super_decoder_layer_num):
                gene_choice.append(self.decoder_self_attention_heads_choice)

        if self.args.task == 'translation':
            for i in range(self.super_decoder_layer_num):
                gene_choice.append(self.decoder_ende_attention_heads_choice)

            for i in range(self.super_decoder_layer_num):
                gene_choice.append(self.decoder_arbitrary_ende_attn_choice)

        return gene_choice


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class EvolutionEnew(object):
    def __init__(self, args, sampler):
        self.population_size = args.population_size
        self.args = args
        self.parent_size = args.parent_size
        self.mutation_size = args.mutation_size
        self.mutation_prob = args.mutation_prob
        self.crossover_size = args.crossover_size
        assert self.population_size == self.parent_size + \
               self.mutation_size + self.crossover_size
        self.converter = Converter(args)
        self.sampler = sampler
        self.gene_choice = self.converter.get_gene_choice()
        self.gene_len = len(self.gene_choice)
        self.evo_iter = args.evo_iter
        self.loss_ranker = PerformanceRanker(feature_list=args.loss_feature_list)
        self.latency_ranker = PerformanceRanker(feature_list=args.latency_feature_list)

        self.loss_ranker.load(args.loss_ranker_path)
        self.latency_ranker.load(args.latency_ranker_path)
        self.latency_constraint = args.latency_constraint
        self.modelSize_constraint = args.modelSize_constraint
        self.write_config_path = args.write_config_path
        self.best_config = None

    def arch_to_config(self, arch: dict):
        config = ''
        if 'encoder' in arch and 'decoder' in arch:
            modules = [arch['encoder'], arch['decoder']]
        elif 'encoder' in arch and 'decoder' not in arch:
            modules = [arch['encoder']]
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

    def run_evo_search(self):
        popu = self.random_sample(self.population_size)

        all_scores_list = []

        for i in range(self.evo_iter):
            print(f"| Start Iteration {i}:")
            popu_scores = self.get_scores(popu)
            print(f"| Iteration {i}, Lowest loss: {min(popu_scores)}")
            if self.args.reverse:
                sorted_ind = np.array(popu_scores).argsort()[::-1][:self.parent_size]
            else:
                sorted_ind = np.array(popu_scores).argsort()[:self.parent_size]

            self.best_config = self.converter.gene2config(popu[sorted_ind[0]])
            print(f"| Config for lowest loss model: {self.best_config}")
            print(
                f"| Predicted latency for lowest loss model: {self.latency_ranker.predict(utils.get_config_features(self.converter.gene2config(popu[sorted_ind[0]])))}")

            parents_popu = [popu[m] for m in sorted_ind]

            parents_score = [popu_scores[m] for m in sorted_ind]
            all_scores_list.append(parents_score)

            mutate_popu = []

            k = 0
            while k < self.mutation_size:
                mutated_gene = self.mutate(random.choices(parents_popu)[0])
                if self.satisfy_constraints(mutated_gene):
                    mutate_popu.append(mutated_gene)
                    k += 1

            crossover_popu = []

            k = 0
            while k < self.crossover_size:
                crossovered_gene = self.crossover(
                    random.sample(parents_popu, 2))
                if self.satisfy_constraints(crossovered_gene):
                    crossover_popu.append(crossovered_gene)
                    k += 1

            popu = parents_popu + mutate_popu + crossover_popu

        best_config = self.best_config
        if not os.path.exists(self.args.write_config_path):
            os.makedirs(self.args.write_config_path)

        loss = self.loss_ranker.predict(
            [utils.get_config_features(best_config)]).tolist()[0]
        latency = self.latency_ranker.predict(
            [utils.get_config_features(best_config)]).tolist()[0]
        file = self.args.write_config_path + \
               '/loss_{}_{}.yml'.format(loss, latency)
        i = 1
        while os.path.exists(file):
            file = file.replace('.yml', '_{}.yml'.format(i))
            i += 1
        print('store the search result to:', file)
        with open(file, 'w') as fo:
            fo.write(self.arch_to_config(best_config))
        # with open(self.write_config_path, 'w') as fid:
        #     encoder_layer_num = best_config['encoder']['encoder_layer_num']
        #     decoder_layer_num = best_config['decoder']['decoder_layer_num']

        #     fid.write(
        #         f"encoder-embed-dim-subtransformer: {best_config['encoder']['encoder_embed_dim']}\n")
        #     fid.write(
        #         f"decoder-embed-dim-subtransformer: {best_config['decoder']['decoder_embed_dim']}\n\n")

        #     fid.write(
        #         f"encoder-ffn-embed-dim-all-subtransformer: {best_config['encoder']['encoder_ffn_embed_dim'][:encoder_layer_num]}\n")
        #     fid.write(
        #         f"decoder-ffn-embed-dim-all-subtransformer: {best_config['decoder']['decoder_ffn_embed_dim'][:decoder_layer_num]}\n\n")

        #     fid.write(
        #         f"encoder-layer-num-subtransformer: {best_config['encoder']['encoder_layer_num']}\n")
        #     fid.write(
        #         f"decoder-layer-num-subtransformer: {best_config['decoder']['decoder_layer_num']}\n\n")

        #     fid.write(
        #         f"encoder-self-attention-heads-all-subtransformer: {best_config['encoder']['encoder_self_attention_heads'][:encoder_layer_num]}\n")
        #     fid.write(
        #         f"decoder-self-attention-heads-all-subtransformer: {best_config['decoder']['decoder_self_attention_heads'][:decoder_layer_num]}\n")
        #     fid.write(
        #         f"decoder-ende-attention-heads-all-subtransformer: {best_config['decoder']['decoder_ende_attention_heads'][:decoder_layer_num]}\n\n")

        #     fid.write(
        #         f"decoder-arbitrary-ende-attn-all-subtransformer: {best_config['decoder']['decoder_arbitrary_ende_attn'][:decoder_layer_num]}\n\n")

        return self.best_config

    def crossover(self, genes):
        crossovered_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][i])
            else:
                crossovered_gene.append(genes[1][i])

        return crossovered_gene

    def mutate(self, gene):
        mutated_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choices(self.gene_choice[i])[0])
            else:
                mutated_gene.append(gene[i])

        return mutated_gene

    def get_scores(self, genes):
        configs = []
        for gene in genes:
            configs.append(utils.get_config_features(
                self.converter.gene2config(gene)))

        scores = self.loss_ranker.predict(configs).tolist()

        return scores

    def satisfy_constraints(self, gene):
        satisfy = True

        config = [utils.get_config_features(self.converter.gene2config(gene))]

        if self.latency_ranker.predict(config).tolist()[0] > self.latency_constraint:
            satisfy = False
        self.sampler.trainer.set_sample_config(self.converter.gene2config(gene))
        if self.sampler.trainer.model.get_sampled_params_numel(self.converter.gene2config(gene)) > self.modelSize_constraint:
            satisfy = False
        return satisfy

    def random_sample(self, sample_num):
        popu = []
        i = 0
        while i < sample_num:
            samp_gene = []
            for k in range(self.gene_len):
                samp_gene.append(random.choices(self.gene_choice[k])[0])

            if self.satisfy_constraints(samp_gene):
                popu.append(samp_gene)
                i += 1

        return popu


def test():
    config = {
        'encoder': {
            'encoder_embed_dim': 512,
            'encoder_layer_num': 4,
            'encoder_ffn_embed_dim': [1024, 1025, 1026, 1027],
            'encoder_self_attention_heads': [4, 5, 6, 7],
        },
        'decoder': {
            'decoder_embed_dim': 512,
            'decoder_layer_num': 5,
            'decoder_ffn_embed_dim': [2048, 2049, 2050, 2051, 2052],
            'decoder_self_attention_heads': [4, 6, 7, 8, 9],
            'decoder_ende_attention_heads': [3, 4, 5, 6, 7],
            'decoder_arbitrary_ende_attn': [1, 2, 3, 4, 5, 6, 7]
        }
    }

    args = Namespace(encoder_layers=6,
                     decoder_layers=7,
                     encoder_embed_choice=[768, 512],
                     decoder_embed_choice=[768, 512],
                     encoder_ffn_embed_dim_choice=[3072, 2048],
                     decoder_ffn_embed_dim_choice=[3072, 2048],
                     encoder_layer_num_choice=[6, 5],
                     decoder_layer_num_choice=[6, 5, 4, 3],
                     encoder_self_attention_heads_choice=[8, 4],
                     decoder_self_attention_heads_choice=[8, 4],
                     decoder_ende_attention_heads_choice=[8],
                     decoder_arbitrary_ende_attn_choice=[1, 2]
                     )

    converter = Converter(args)
    gene_get = converter.config2gene(config)

    print(gene_get)
    print(len(gene_get))

    config_get = converter.gene2config(gene_get)

    print(config_get)

    print(len(converter.get_gene_choice()))
    print(converter.get_gene_choice())


if __name__ == '__main__':
    test()
