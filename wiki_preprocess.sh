spring.submit \
	arun --gpu -n$1  --quotatype auto --debug\
    "bash configs/wikitext-103/preprocess.sh"
    #"python train_dynamic.py --configs=configs/wmt14.en-de/supertransformer/large.yml"
