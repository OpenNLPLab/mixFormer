T=`date +%m%d%H%M`
g=$(($1<8?$1:8)) ##--distributed-port 12343
spring.submit run  --gpu -n$1 --ntasks-per-node $g \
"python train_dynamic.py --configs=configs/iwslt14.de-en/supertransformer/large.yml \
2>&1 |tee log.train.$T "
