#!/bin/bash

set -ex
export TORCH_HOME="/zhangshuai/.cache/torch"
export http_proxy=http://192.168.105.204:3128
export https_proxy=http://192.168.105.204:3128

script=`realpath -s $0`
exec_dir=`dirname $script`
cd $exec_dir


proj="DEEPGNN-RT"
topic="0726"
export WANDB_PROJECT=$proj-$topic
export CUBLAS_WORKSPACE_CONFIG=:16:8


for name in "GCN_edge_attention_GRU" "GCN_edge_attention_GRU_without_residual"
do
  for s in 1 2 3
  do

    for num_of_layer in 3 5 8 16
    do
    CUDA_VISIBLE_DEVICES=0 /zhangshuai/software/anaconda3/envs/dgl/bin/python train.py \
            --model_name=$name \
            --seed=$s \
            --num_layers=$num_of_layer \
            --norm="none" \
            --inference
    done
  done
done


for name in "GCN_edge_attention_GRU_no_denselayer" "GCN_edge_mean" "GCN_edge_sum" "GCN_attention_GRU"
do
  for s in 1
  do
    for num_of_layer in 3 5 8 16
    do
    CUDA_VISIBLE_DEVICES=0 /zhangshuai/software/anaconda3/envs/dgl/bin/python train.py \
            --model_name=$name \
            --seed=$s \
            --num_layers=$num_of_layer \
            --norm="none" \
            --inference
    done
  done
done




