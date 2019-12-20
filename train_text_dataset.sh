#!/bin/bash

# DATASET_PATH=../../dataset/ArtistReviewCorpus_20180608
# LANG=ja
DATASET_PATH=../../dataset/MARD
LANG=en
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
OUTPUT_PATH=model/mard-fasttext-model-finetuned/fasttext.gensim.model
SIZE=300
WINDOW=8
MIN_COUNT=1
EPOCH=5

python train_text_dataset.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --dataset-path=$DATASET_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --epoch=$EPOCH
