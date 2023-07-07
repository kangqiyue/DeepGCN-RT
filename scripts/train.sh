#!/bin/bash

set -ex
export CUBLAS_WORKSPACE_CONFIG=:16:8


for name in "normal_GCN" "GCN_edge" "GCN_edge_residual" "DeepGCN-RT"
do
  for s in 1 2 3
  do
    for num_of_layer in 3 5 8 16
    do
    CUDA_VISIBLE_DEVICES=0 /zhangshuai/software/anaconda3/envs/dgl/bin/python train.py \
            --model_name=$name \
            --seed=$s \
            --num_layers=$num_of_layer \
            --inference

    done
  done
done