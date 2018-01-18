#!/bin/bash

echo create train_lmdb12...

rm -rf /home/vincent/github/data/mtcnn/12/train_lmdb12

cd /home/vincent/github/caffe

./build/tools/convert_mtcnn \
/home/vincent/github/data/mtcnn/ \
/home/vincent/github/data/mtcnn/12/label-train.txt \
/home/vincent/github/data/mtcnn/12/train_lmdb12 \
--backend=lmdb \
--shuffle=true \
--encoded=true \
--encode_type=jpg

