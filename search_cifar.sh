spring.submit run --gpu -n1 -p MMG \
"python evo_search.py --configs=configs/cifar10/supertransformer/large.yml --reverse\
 --latency-feature-list 0 2 3 \
 --loss-feature-list 0 1 2 3 \
 --loss-ranker-path checkpoints/cifar10/acc_ranker \
 --latency-ranker-path checkpoints/cifar10/latency_ranker \
 --latency-constraint 100 \
 --candidate-size 100000 \
 --write-config-path configs "

