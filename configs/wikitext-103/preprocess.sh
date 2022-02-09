#!/bin/bash

# Download the raw dataset
wget -O wikitext-103-v1.zip 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
unzip wikitext-103-v1.zip

# Binarize the dataset
TEXT=wikitext-103
python -u tools/preprocess.py \
       --only-source \
       --trainpref $TEXT/wiki.train.tokens \
       --validpref $TEXT/wiki.valid.tokens \
       --testpref $TEXT/wiki.test.tokens \
       --destdir data-bin/wikitext-103 \
       --workers 24