spring.submit run --gpu -n1 -p MMG \
"python evo_search.py --configs=configs/cifar10/supertransformer/supernet-S.yml --reverse --no-rank-model \
 --latency-feature-list 1 4 0 3 2 \
 --loss-feature-list 4 1 2 3 0 \
 --loss-ranker-path checkpoints/cifar10_mixformer/acc_ranker \
 --latency-ranker-path checkpoints/cifar10_mixformer/latency_ranker \
 --latency-constraint 15.06 \
 --candidate-size 100000 \
 --modelSize-constraint 25000000 \
 --write-config-path configs/cifar10_mixformer/ "

