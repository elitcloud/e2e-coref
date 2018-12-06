#!/bin/bash

# Download pretrained embeddings.
curl -O http://lsz-gpu-01.cs.washington.edu/resources/glove_50_300_2.txt
#curl -O https://nlp.stanford.edu/data/glove.840B.300d.zip
#unzip glove.840B.300d.zip
#rm glove.840B.300d.zip

# Download pretrained elmo
curl -kL -o elmo.tar.gz https://tfhub.dev/google/elmo/2?tf-hub-format=compressed
mkdir -p elmo
tar -xvzf elmo.tar.gz -C elmo
rm elmo.tar.gz

# Download pretrained coref model
curl -O http://lsz-gpu-01.cs.washington.edu/resources/coref/char_vocab.english.txt

ckpt_file=c2f_final.tgz
curl -O http://lsz-gpu-01.cs.washington.edu/resources/coref/$ckpt_file
mkdir -p logs
tar -xzvf $ckpt_file -C logs
rm $ckpt_file
