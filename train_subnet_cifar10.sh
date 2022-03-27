T=`date +%m%d%H%M`
TASK=cifar10_subnet_mixformer_87.4
g=$(($1<8?$1:8)) ##--distributed-port 12343
spring.submit run --gpu -n$1 --ntasks-per-node $g  -p MMG \
"python train.py  \
 --sub-configs=configs/cifar10/subtransformer/supnet.yml \
 --configs=configs/cifar10_mixformer/loss_89.07_11.52.yml \
 2>&1 |tee checkpoints/$TASK/log.train.$T "
