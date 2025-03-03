#!/bin/bash

conda create -n llm_based_control_rewrite python=3.9.17
conda activate llm_based_control_rewrite
cd  Control_Text_Adaptation/

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

pip install -r requirements.txt

pip install -e .

cd data_auxiliary/en/
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gzip -d cc.en.300.vec.gz
cd ../../

python -m nltk.downloader stopwords
python -m nltk.downloader punkt

python llm_based_control_rewrite/utils/prepare_word_embeddings_frequency_ranks.py

#now you are ready to run the llm rewrite.