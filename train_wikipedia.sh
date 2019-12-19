#!/bin/bash

WIKIPEDIA_DUMP_PATH=../../dataset/Wikipedia/jawiki-latest-pages-articles.xml.bz2
LANG=ja
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
OUTPUT_PATH=model/enwiki-fasttext-model/fasttext.gensim.model
SIZE=300
WINDOW=8
MIN_COUNT=5
EPOCH=5

# train fastText model
python train_wikipedia.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --epoch=$EPOCH
