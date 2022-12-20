#! /bin/bash

read -p "CUDA_VISIBLE_DEVICES [0,1,2,3]: " idx
idx=${idx:-0,1,2,3}

CUDA_VISIBLE_DEVICES=$idx python train.py --outdir=./checkpoints --data=../../Dataset/FFHQ --cfg=stylegan2 --cond=1 --gpus=4 --aug=noaug