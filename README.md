# Installation

## Install pytorch

NVIDIA Turing architecture or lower GPUs (TITAN V, 2080, etc.):
```
conda create -n ranknas python=3.7

conda activate ranknas

conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch
```

NVIDIA Ampere architecture GPUs (3090, A100, etc.):
```
conda create -n ranknas python=3.8

conda activate ranknas

conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch
```

## Install other dependent libraries

```
pip install -e .

cd torchprofile && pip install -e . && cd ..
```

# Data pre-process
```
bash configs/iwslt14.de-en/preprocess.sh

bash configs/wmt14.en-de/preprocess.sh

bash configs/wikitext-103/preprocess.sh
```

# 1. Train the supernet

IWSLT14.De-En:
```
python train_dynamic.py --configs=configs/iwslt14.de-en/supertransformer/large.yml
```

WMT14.En-De:
```
python train_dynamic.py --configs=configs/wmt14.en-de/supertransformer/large.yml
```

WikiText-103:
```
python -u train_dynamic.py --configs=configs/wikitext-103/supertransformer/config.yml
```

Classification: cifar10:
```
sh train_supernet_cifar10.sh
```
After training the super-transformers, check `checkpoints`.

# 2.1 Collect architectures and their performance
## 2.1.1 Collect the loss data with the supernet

IWSLT14.De-En:
```
CUDA_VISIBLE_DEVICES=0 python -u loss_dataset.py --configs=configs/iwslt14.de-en/loss_dataset/config_large.yml
```

WMT14.En-De:
```
CUDA_VISIBLE_DEVICES=0 python -u loss_dataset.py --configs=configs/wmt14.en-de/loss_dataset/config_large.yml
```

WikiText-103:
```
CUDA_VISIBLE_DEVICES=0 python -u loss_dataset.py --configs=configs/wikitext-103/loss_dataset/config.yml
```

Classification: cifar10:
```
sh acc_dataset.sh
```
After collecting the data, check `loss_dataset`.

## 2.1.2 Collect the latency data

IWSLT14.De-En:
```
CUDA_VISIBLE_DEVICES=0 python -u latency_dataset.py --configs=configs/iwslt14.de-en/latency_dataset/large_gpu_1080ti.yml
```

WMT14.En-De:
```
CUDA_VISIBLE_DEVICES=0 python -u latency_dataset.py --configs=configs/wmt14.en-de/latency_dataset/gpu_1080ti.yml
```

WikiText-103:
```
CUDA_VISIBLE_DEVICES=0 python -u latency_dataset.py --configs=configs/wikitext-103/latency_dataset/gpu.yml
```

Classification: cifar10:
```
sh collect_latency.sh
```
After collecting the data, check `latency_dataset`.

# 2.2 Train the performance ranker
## 2.2.1 Train the loss ranker

IWSLT14.De-En:
```
python -u ranker.py -data loss_dataset/iwslt14_large_loss.data -save checkpoints/iwslt14.de-en/loss_ranker
```

WMT14.En-De:
```
python -u ranker.py -data loss_dataset/wmt14_large_loss.data -save checkpoints/wmt14.en-de/loss_ranker
```

WikiText-103:
```
python -u ranker.py -data loss_dataset/wiki103_loss.data -save checkpoints/wikitext-103/loss_ranker
```

Classification: cifar10:
```
sh train_acc_ranker.sh
```

After training the loss ranker, check `checkpoints`.

## 2.2.2 Train the latency ranker

IWSLT14.De-En:
```
python -u ranker.py -data latency_dataset/iwslt14_large_latency.data -save checkpoints/iwslt14.de-en/latency_ranker
```

WMT14.En-De:
```
python -u ranker.py -data latency_dataset/wmt14_large_latency.data -save checkpoints/wmt14.en-de/latency_ranker
```

WikiText-103:
```
python -u ranker.py -data latency_dataset/wiki103_gpu_latency.data -save checkpoints/wikitext-103/latency_ranker
```

Classification: cifar10:
```
python -u ranker.py -data latency_dataset/cifar_mixatt_gpu.data -save checkpoints/cifar10/latency_ranker
```

After training the latency ranker, check `checkpoints`.

**Note that training the ranker will print the selected feature indices, which are required for the following searching stage.**

# 3. Search with the performance ranker

WikiText-103:
```
python evo_search.py --configs=configs/wikitext-103/evo_search/config.yml \
 --latency-feature-list SELECTED_LATENCY_FEATURES \
 --loss-feature-list SELECTED_LOSS_FEATURES \
 --loss-ranker-path checkpoints/wikitext-103/loss_ranker \
 --latency-ranker-path checkpoints/wikitext-103/latency_ranker \
 --latency-constraint 500 \
 --candidate-size 100000 \
 --write-config-path configs

python random_search.py \
 --topk 1 \
 --latency-feature-list SELECTED_LATENCY_FEATURES \
 --loss-feature-list SELECTED_LOSS_FEATURES \
 --loss-ranker-path checkpoints/wikitext-103/loss_ranker \
 --latency-ranker-path checkpoints/wikitext-103/latency_ranker \
 --latency-constraint 500 \
 --candidate-size 100000 \
 --write-config-path configs
```

Classification: cifar10:
```
sh search_cifar_mix.sh
```

Replace `SELECTED_LATENCY_FEATURES` and `SELECTED_LOSS_FEATURES` with your results of Step 2.2.1 and Step 2.2.2.
An example: `0 1 2 3 4 5`.

After searching, check `configs`.

**Note that it will print the search results (stored in a yaml file).**

# 4.1 Train a discovered sub-transformer from scratch
```
python train.py \
 --sub-configs=configs/wikitext-103/subtransformer/config.yml \
 --configs=configs/PATH_TO_YOUR_SEARCH_RESULT.yml 
```

Classification: cifar10:
```
sh train_subnet_cifar10.sh
```

**Replace PATH_TO_YOUR_SEARCH_RESULT to the result of Step 3**

# 4.2 Evaluate the trained sub-transformer

Machine Translation:
```
PATH_TO_YOUR_SEARCH_RESULT=best.yml
python tools/average_checkpoints.py -path checkpoints/iwslt/test/PATH_TO_YOUR_SEARCH_RESULT -n 10

bash configs/iwslt14.de-en/test_large.sh checkpoints/iwslt/test/PATH_TO_YOUR_SEARCH_RESULT/averaged.pt configs/iwslt/test/PATH_TO_YOUR_SEARCH_RESULT
```

**Replace PATH_TO_YOUR_SEARCH_RESULT to the result of Step 3**

Language Modeling:
```
PATH_TO_MODEL=checkpoints/wikitext-103/subtransformer/checkpoint_best.pt
GPU_ID=0
bash configs/wikitext-103/test.sh PATH_TO_MODEL GPU_ID
```
