T=`date +%m%d%H%M`
TASK=cifar_cosformer_acc_dataset
g=$(($1<8?$1:8)) ##--distributed-port 12343
spring.submit run --gpu -n$1 --ntasks-per-node $g  \
"python -u acc_dataset.py --configs=configs/cifar10/acc_dataset/config.yml \
2>&1 |tee log.train.{$TASK}.$T "
