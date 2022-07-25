#!/bin/bash

set -ex

for model_name in "GCN_sum" "GCN_mean" "GCN_attention_GRU" "DEEPGCN_sum_readout" "DEEPGCN_mean_readout" "DEEPGCN_attention_GRU"
do

    for num_of_layer in 3 5 8 16 24
    do
    CUDA_VISIBLE_DEVICES=0 python train.py \
            --model_name=$model_name \
            --seed=1 \
            --num_layers=$num_of_layer \
            --inference

    done


    for num_of_layer in 3 5 8 16 24
    do
    CUDA_VISIBLE_DEVICES=0 python train.py \
            --model_name=$model_name \
            --seed=2 \
            --num_layers=$num_of_layer \
            --inference


    done


    for num_of_layer in 3 5 8 16 24
    do
    CUDA_VISIBLE_DEVICES=0 python train.py \
            --model_name=$model_name \
            --seed=3 \
            --num_layers=$num_of_layer \
            --inference


    done

done