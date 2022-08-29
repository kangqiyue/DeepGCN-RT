#!/bin/bash

set -ex
export CUBLAS_WORKSPACE_CONFIG=:16:8


for name in "GCN_attention_GRU" "GCN_edge_attention_GRU" "GCN_edge_attention_GRU_without_residual" "GCN_edge_mean" "GCN_edge_sum" "GCN_edge_attention_GRU_no_denselayer"
do
  for s in 1 2 3
  do
    for num_of_layer in 3 5 8 16
    do
    CUDA_VISIBLE_DEVICES=0 /zhangshuai/software/anaconda3/envs/dgl/bin/python train.py \
            --model_name=$name \
            --seed=$s \
            --num_layers=$num_of_layer

    done
  done
done