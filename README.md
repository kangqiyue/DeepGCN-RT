# DeepGNN-RT

This repository correspond to the article: **Deep Graph Neutral Network for Small Molecule Retention Time Prediction and Their Transfer Learning**


## Note 

This repository contians GNN models for rentention time prediction, including DeepGNN-RT, GAT, GCN, and GIN mdoels. The _models.py dataset.py_ and _train.py_ conrtain the model, dataset, and train codes, respectively. 
In addition, the _transfer_learning.py_ contains the transfer learning code(10 fold cross validation). The results of transfer learning for all models are contained in folder named _**result**_.

## Environment
The environment dependencies for Linux system are contained in the file named _**environment.yaml**_.


## Run the code
To run the training code, the following command could be used:

    python train.py \
    --model_name "DEEPGNN" \
    --dataset "SMRT"
    --num_layers 16 \
    --hid_dim 200 \
    --epochs 200 \
    --lr 0.001 \
    --batch_size 64\
    --early_stop 30 \
    --seed 1 


To run the inference code, the following command could be used:

    python inference.py \
    --SMILES "demo_SMILES"\
    --model_path "model path"

