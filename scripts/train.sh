#!/bin/bash

set -ex
export WANDB_CONFIG_DIR="/data/users/kangqiyue/kqy/DEEPGNN_RT"
model_name="DEEPGNN"


for num_of_layer in 5 8 16
do
CUDA_VISIBLE_DEVICES=2 python train.py \
        --model_name=$model_name \
        --seed=1 \
        --num_layers=$num_of_layer \

done


for num_of_layer in 5 8 16
do
CUDA_VISIBLE_DEVICES=2 python train.py \
        --model_name=$model_name \
        --seed=2 \
        --num_layers=$num_of_layer \

done


for num_of_layer in 5 8 16
do
CUDA_VISIBLE_DEVICES=2 python train.py \
        --model_name=$model_name \
        --seed=3 \
        --num_layers=$num_of_layer \

done