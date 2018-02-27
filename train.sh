#!/usr/bin/env python
#-*- coding:utf-8 -*-
#!/bin/bash
# train.sh
# usage:
#   source train.sh [dataset] [base_dir] &
#   . train.sh &
#   tail -f $L
if [ -n "$1" ]; then
    dataset=$1
    shift
fi

if [ -n "$1" ]; then
    base_dir=$1
    shift
fi

export CUDA_VISIBLE_DEVICES=0
base_dir=/media/btows/SDB/train_corpus/
dataset=med
dt=$(date +%Y%m%d%H%M)
L=$base_dir/train_${dt}.log
rm -rf $base_dir/training/
python3 preprocess.py --base_dir $base_dir --input $dataset #>>$L 2>&1

python3 train.py --base_dir $base_dir >>$L 2>&1 #--restore_path=$restore_path
