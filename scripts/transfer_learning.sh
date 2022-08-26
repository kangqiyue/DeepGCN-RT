#!/bin/bash

set -ex
export TORCH_HOME="/zhangshuai/.cache/torch"
export http_proxy=http://192.168.105.204:3128
export https_proxy=http://192.168.105.204:3128
export CUBLAS_WORKSPACE_CONFIG=:16:8

script=`realpath -s $0`
exec_dir=`dirname $script`
cd $exec_dir

best_model_file="/zhangshuai/work/DEEPGNN_RT/output/deep_gnn_22_8_3/GCN_edge_attention_GRU/GCN_edge_attention_GRU_none_16_lr_0.001_seed_3/best_model_weight.pth"

for d in 'Cao_HILIC_116' \
         'Eawag_XBridgeC18_364' \
         'FEM_lipids_72' \
         'FEM_long_412' \
         'FEM_short_73' \
         'IPB_Halle_82' \
         'LIFE_new_184' \
         'LIFE_old_194' \
         'MTBLS87_147' \
         'UniToyama_Atlantis_143'
do

        for s in 0 1 2 3 4 5 6 7 8 9
        do CUDA_VISIBLE_DEVICES=0 /zhangshuai/software/anaconda3/envs/dgl/bin/python transfer_learning.py \
                --seed=$s \
                --dataset=$d \
                --best_model_file=$best_model_file
        done

done


best_model_file="no"

for d in 'Cao_HILIC_116' \
         'Eawag_XBridgeC18_364' \
         'FEM_lipids_72' \
         'FEM_long_412' \
         'FEM_short_73' \
         'IPB_Halle_82' \
         'LIFE_new_184' \
         'LIFE_old_194' \
         'MTBLS87_147' \
         'UniToyama_Atlantis_143'
do

        for s in 0 1 2 3 4 5 6 7 8 9
        do CUDA_VISIBLE_DEVICES=0 /zhangshuai/software/anaconda3/envs/dgl/bin/python transfer_learning.py \
                --seed=$s \
                --dataset=$d \
                --best_model_file=$best_model_file
        done

done
