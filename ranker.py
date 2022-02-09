import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import random
import configargparse
import numpy as np
from numpy import sqrt
from scipy.stats import kendalltau, pearsonr


random.seed(233)
np.random.seed(233)

from fairseq.utils import get_config_features, get_feature_info, get_feature_info_lm

class PerformanceRanker(object):
    def __init__(self, dataset_path='', feature_list=None):
        self.dataset_path = dataset_path
        self.feature_list = feature_list

        self.dataset = None

        self.train_x = None
        self.train_y = None

        self.valid_x = None
        self.valid_x = None

        self.model = None

        self.feature_names = []

        self.task = 'translation'

    def save(self, fname):
        self.model.save_model(fname)

    def load(self, fname):
        self.model = lgb.Booster(model_file=fname)

    def train(self):
        self.valid_y = list(self.valid_y)
        self.train_y = list(self.train_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'max_depth': 6,
            'num_leaves': 30,
            'learning_rate': 0.1,
            'verbose': -1,
            'num_threads': 8,
        }
        lgb_train = lgb.Dataset(
            self.train_x, self.train_y, feature_name=self.feature_names)
        lgb_eval = lgb.Dataset(self.valid_x, self.valid_y, reference=lgb_train)

        print('training set size:', len(self.train_y))
        print('validation set size:', len(self.valid_y))
        self.model = lgb.train(params,
                               lgb_train,
                               valid_sets=lgb_eval,
                               early_stopping_rounds=5,
                               verbose_eval=False)

        # from sklearn.inspection import permutation_importance
        # r = permutation_importance(self.model, self.valid_x, self.valid_y, n_repeats=30, random_state=0)
        # for i in r.importances_mean.argsort()[::-1]:
        #     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        #         print(f"{lgb_train.feature_name[i]:<8}"
        #               f"{r.importances_mean[i]:.3f}"
        #               f" +/- {r.importances_std[i]:.3f}")

        # plot = lgb.create_tree_digraph(self.model, show_info=['internal_value'], precision=2, )
        # plt.savefig('tree.pdf', format='pdf', bbox_inches="tight")
        # with open('plot.svg', 'w', encoding='utf8') as f:
        #     f.write(plot._repr_svg_())

    def validate(self, data):
        x, y = data
        test_data = x.tolist()
        gold = np.array(y).flatten()
        pred = self.model.predict(test_data)
        print('output class:', len(set(pred)))
        kendalltau_metric = kendalltau(pred, y)[0]
        pearsonr_metric = pearsonr(pred, y)[0]
        rmse = sqrt(mean_squared_error(pred, gold))

        print('kendalltau corr of outputs:', kendalltau_metric)
        print('pearson corr of outputs:', pearsonr_metric)
        print('RMSE:', rmse)

        return rmse

    def predict(self, archs):
        if isinstance(archs, dict):
            archs = get_config_features(archs)
        return self.model.predict(self.feature_slection(archs), num_threads=8)

    def feature_rank(self, sample_num=200, feature_num=5, prune=False):
        r1 = []
        r2 = []
        for l in [sample_num]:
            corr_list = []
            pearsonr_list = []
            kendalltau_list = []
            raw_features = self.train_x[:l]
            targets = self.train_y[:l]
            for i in range(len(raw_features[0])):
                if not ([f[i] for f in raw_features] == [f[i] for f in raw_features][0]).all():
                    pearsonr_metric = pearsonr(
                        [f[i] for f in raw_features], targets)[0]
                    kendalltau_metric = kendalltau(
                        [f[i] for f in raw_features], targets)[0]
                    pearsonr_list.append((i, pearsonr_metric))
                    kendalltau_list.append((i, kendalltau_metric))
                    corr_list.append((i, abs(kendalltau_metric)))

            # f_importance = f_regression(raw_features, np.array(targets))
            # mutual_info = mutual_info_regression(raw_features, np.array(targets))
            # mutual_info = [(i, mutual_info[i]) for i in range(len(mutual_info))]

            corr_list.sort(key=lambda x: x[1], reverse=True)
            # mutual_info.sort(key=lambda x: x[1], reverse=True)
            # pearsonr_list.sort(key=lambda x: x[1], reverse=True)
            # kendalltau_list.sort(key=lambda x: x[1], reverse=True)

            r1.append((l, [p[0] for p in pearsonr_list[:3]]))
            r2.append((l, [p[0] for p in kendalltau_list[:3]]))

        print('selected features:')
        if self.task == 'translation':
            feature_names = get_feature_info()
        elif self.task == 'language_modeling':
            feature_names = get_feature_info_lm()
        keep_list = []
        for f in corr_list[:feature_num]:
            if f[0] not in keep_list:
                keep_list.append(f[0])
                print(feature_names[f[0]], f[1])

        if prune:
            self.feature_list = keep_list
        else:
            self.feature_list = list(range(len(feature_names)))
        self.feature_names = [feature_names[i] for i in self.feature_list]
        print(self.feature_list)

        self.train_x = ranker.feature_slection(
            self.train_x,
            is_training=True
        )
        self.valid_x = ranker.feature_slection(
            self.valid_x,
            # is_training=True
        )
        self.test_x = ranker.feature_slection(
            self.test_x,
            # is_training=True
        )
        self.train_x = self.train_x[:sample_num]
        self.train_y = self.train_y[:sample_num]
        self.valid_x = self.valid_x[:]
        self.valid_y = self.valid_y[:]

    def feature_slection(self, data, is_training=False):
        if not (isinstance(data[0], list) or isinstance(data[0], np.ndarray)):
            data = [data]
        # select some features
        keep_list = self.feature_list
        if self.task == 'translation':
            feature_names = get_feature_info()
        elif self.task == 'language_modeling':
            feature_names = get_feature_info_lm()
        if isinstance(keep_list[0], int):
            keep_list = [feature_names[d] for d in keep_list]

        if is_training:
            features_count = {}
            for i, n in enumerate(feature_names):
                features_count[n] = len(set([f[i] for f in data]))

            raw_space, selected_space = 1, 1
            for i in features_count.values():
                raw_space *= i
            for i in keep_list:
                selected_space *= features_count[i]
            print('raw feature count:', features_count)
            print('raw feature space:', raw_space)
            print('selected features:', keep_list)
            print('selected feature space: {} ({:2.9f})'.format(
                selected_space, float(selected_space) / float(raw_space)))

        self.feature_dict = {}

        for i, n in enumerate(feature_names):
            self.feature_dict[n] = i

        features = []
        drop_ids = [self.feature_dict[f] for f in keep_list]
        for x in data:
            feature = []
            for i, d in enumerate(x):
                if i in drop_ids:
                    feature.append(d)
            features.append(feature)
        return np.array(features)

    def split(self):
        sample_num = len(self.dataset['x'])
        train_num = int(np.floor(0.8 * sample_num))
        valid_num = int(np.floor(0.1 * sample_num))

        self.train_x = self.dataset['x'][:train_num]
        self.train_y = self.dataset['y'][:train_num]

        self.valid_x = self.dataset['x'][train_num:(train_num + valid_num)]
        self.valid_y = self.dataset['y'][train_num:(train_num + valid_num)]

        self.test_x = self.dataset['x'][(train_num + valid_num):]
        self.test_y = self.dataset['y'][(train_num + valid_num):]

        pass

    def parse_line(self, line, is_latency=False):

        line = line[:-1].split(',')
        floats = list(map(float, line))

        if len(floats) >= 10:
            features = floats[:10]

            if not is_latency:
                outputs = floats[-1]
            else:
                latency_std = floats[-2] + floats[-3]
                latency_mean = floats[-4] + floats[-5]
                if latency_std / latency_mean < 0.015:
                    outputs = latency_mean
                else:
                    outputs = None
        else:
            features = floats[:4]
            if is_latency:
                latency_std = floats[-1]
                latency_mean = floats[-2]
                if latency_std / latency_mean < 0.015:
                    outputs = latency_mean
                else:
                    outputs = None
            else:
                outputs = floats[4]
                
        return features, outputs

    def read_dataset(self):
        features_all = []
        output_all = []

        # read the dataset
        with open(self.dataset_path, 'r') as fid:
            title = fid.readline()
            self.task = 'translation' if 'encoder' in title else 'language_modeling'

            if self.task == 'translation':
                self.feature_names = title.split(',')[:10]
            else:
                self.feature_names = title.split(',')[:4]
            is_latency_data = 'latency' in title

            for line in fid:
                if 'embed_dim' in line:
                    continue
                features, outputs = self.parse_line(line, is_latency_data)
                if outputs is not None:
                    features_all.append(features)
                    output_all.append(outputs)
        
        # remove duplicated examples
        examples = list(zip(features_all, output_all))
        hash_table = []
        unique_examples = []
        for e in examples:
            hash_key = str(e[0])
            if hash_key not in hash_table:
                hash_table.append(hash_key)
                unique_examples.append(e)
        examples = unique_examples

        random.shuffle(examples)
        features_all, output_all = zip(*examples)
        len_x = len(features_all[0])
        features_all = np.array(features_all).flatten().reshape(-1, len_x)
        self.dataset = {'x': features_all, 'y': output_all}


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('-data', type=str,
                        default='loss_dataset/iwslt.loss.data')

    parser.add_argument('-save', type=str, default='checkpoints/iwslt14.de-en/loss_ranker',
                        help='path to save the loss ranker')

    args = parser.parse_args()

    res_all = []
    files = []
    keep_list = list(range(10))

    ranker = PerformanceRanker(
        dataset_path=args.data
    )

    ranker.read_dataset()
    ranker.split()
    ranker.feature_rank(sample_num=2000, prune=('loss' not in args.data))
    ranker.train()
    ranker.save(args.save)

    valid_x = ranker.valid_x
    for j in range(len(ranker.feature_list)):
        for i in range(len(valid_x)):
            valid_x[i][j] = random.random()
        ranker.validate((ranker.valid_x, ranker.valid_y))
    ranker.validate((ranker.test_x, ranker.test_y))
    print('selected feature list:', ranker.feature_list)
