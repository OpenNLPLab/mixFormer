spring.submit \
	arun --gpu -n$1  --quotatype auto --debug\
    "python train_dynamic.py --configs=configs/wikitext-103/supertransformer/config.yml"
    #"python train_dynamic.py --configs=configs/wmt14.en-de/supertransformer/large.yml"
