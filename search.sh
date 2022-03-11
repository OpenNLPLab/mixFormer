spring.submit run --gpu -n1 \
"python evo_search.py --configs=configs/iwslt14.de-en/supertransformer/large.yml \
 --latency-feature-list 5 9 0 6 2 \
 --loss-feature-list 0 1 2 3 4 5 6 7 8 9 \
 --loss-ranker-path checkpoints/iwslt14.de-en/loss_ranker \
 --latency-ranker-path checkpoints/iwslt14.de-en/latency_ranker \
 --latency-constraint 100 \
 --candidate-size 100000 \
 --write-config-path configs "

