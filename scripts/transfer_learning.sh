#!/bin/bash

set -ex


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
        do CUDA_VISIBLE_DEVICES=7 python transfer_learning.py \
                --seed=$s \
                --dataset=$d
        done

done

