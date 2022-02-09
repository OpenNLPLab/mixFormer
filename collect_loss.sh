spring.submit \
	arun --gpu -n$1 --quotatype auto --debug\
    "python loss_dataset.py --configs=configs/wikitext-103/loss_dataset/config.yml"