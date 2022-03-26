spring.submit run --gpu -n1 -p MMG \
"python evo_search.py --configs=configs/cifar10/supertransformer/supernet-S.yml --reverse\
 --latency-feature-list 1 2 3 0 \
 --loss-feature-list 2 3 1 0 \
 --loss-ranker-path checkpoints/cifar10_cosformer/acc_ranker \
 --latency-ranker-path checkpoints/cifar10_cosformer/latency_ranker \
 --latency-constraint 13.06 \
 --candidate-size 100000 \
 --write-config-path configs "

