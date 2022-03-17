T=`date +%m%d%H%M`
TASK=cifar10_subnet_cosformer
g=$(($1<8?$1:8)) ##--distributed-port 12343
spring.submit run --gpu -n$1 --ntasks-per-node $g -p MMG \
"python train.py \
 --sub-configs=configs/cifar10/subtransformer/supnet.yml \
 --configs=configs/cifar10/subtransformer/subnet_87.4.yml \
 2>&1 |tee log.train.{$TASK}.$T "
