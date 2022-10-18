# DeepGNN-RT

This repository correspond to the article: **Deep Graph Convolutional Network for Small-Molecule Retention Time Prediction** by Qiyue Kang et al. (the paper is in submission).



## Note 

This repository contians GNN models for rentention time prediction, including DeepGCN-RT, and GCN models (GAT and GIN were also implemented). The ```models.py dataset.py and train.py``` contain the model, dataset, and train codes, respectively. 
In addition, the ```transfer_learning.py``` contains the transfer learning code(10 fold cross validation). The results of transfer learning for all models are contained in folder named ```result```.

## Environment
The environment dependencies for Linux system are contained in the file named ```environment.yaml```. Use ```conda update``` to build the environment.


## Run the code
To run the training code, the following command could be used:

```
python train.py \
--model_name "GCN_edge_attention_GRU" \
--dataset "SMRT"
--num_layers 16 \
--hid_dim 200 \
--epochs 200 \
--lr 0.001 \
--batch_size 64\
--early_stop 30 \
--seed 1 
```
In addition, the train and transfer learning processes could also be started by the bash scripts, see the folder named ```scripts```.

To run the training process on SMRT data set:
```
sh scripts/train.sh
```


To run the transfer learning on nine transfer learning data sets, use:
```
scripts/transfer_learning.sh
```

To run the inference code, the following command could be used:
```
python inference.py \
--SMILES "demo_SMILES"\
--model_path "model path"
```

