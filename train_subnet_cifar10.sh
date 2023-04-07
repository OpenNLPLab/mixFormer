T=`date +%m%d%H%M`
TASK=train_subnet
g=$(($1<8?$1:8)) ##--distributed-port 12343
spring.submit run --gpu -n$1 --ntasks-per-node $g  -p MMG \
"python train.py \
 --sub-configs=configs/cifar10/subtransformer/supnet.yml \
 --configs=PATH_TO_YOUR_SEARCH_RESULT \
 2>&1 |tee checkpoints/$TASK/log.train.$T "
