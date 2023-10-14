# DeepGCN-RT

## News
We have released the training code and the trained model weights. If you find them helpful, please cite our article!

**Kang, Q.**; **Fang, P.**; Zhang, S.; Qiu, H.; **Lan, Z**. Deep Graph Convolutional Network for Small-Molecule Retention Time Prediction. Journal of Chromatography A. 2023, 464439, ISSN 0021-9673.

Article url: https://doi.org/10.1016/j.chroma.2023.464439.
(https://www.sciencedirect.com/science/article/pii/S0021967323006647)


## Model performance

The model performances were evaluated through metrics including mean absolute error (MAE), median absolute error (MedAE), mean absolute percentage error (MAPE), mean square error (MSE), and R2. 


### Model performance


|            | Depth | MAE    |       | MedAE  |       | MAPE   |        | R2     |        | MSE   |     |
|------------|:-----:|--------|-------|--------|-------|--------|--------|--------|--------|-------|-----|
|            |       |  Mean  |  Std  |  Mean  |  Std  |  Mean  |   Std  |  Mean  |   Std  |  Mean | Std |
| DeepGCN-RT | 3     | 27.97  | 0.20  | 14.01  | 0.07  | 0.035  | 0.000  | 0.892  | 0.002  | 3303  | 55  |
| DeepGCN-RT | 5     | 27.00  | 0.19  | 12.91  | 0.18  | 0.034  | 0.000  | 0.892  | 0.001  | 3288  | 33  |
| DeepGCN-RT | 8     | 26.61  | 0.09  | 12.44  | 0.05  | 0.034  | 0.000  | 0.892  | 0.001  | 3286  | 31  |
| DeepGCN-RT | 16    | 26.55  | 0.17  | 12.38  | 0.12  | 0.033  | 0.000  | 0.892  | 0.001  | 3299  | 45  |


## Note 

This repository contians GNN models for rentention time prediction, including DeepGCN-RT, and plain GCN model. The ```models.py dataset.py and train.py``` contain source code. The model weights are included in the ```model_path``` folder. The ```transfer_learning.py``` contains the transfer learning code(10 fold cross validation). The full results of transfer learning for all models are contained in folder named ```result```.

## Build conda environment
The environment dependencies for Linux system are contained in the file named ```environment.yaml```. Use ```conda update``` to build the environment.


## Run the code
To run the training code, the following command could be used:

```
python train.py \
--model_name "DeepGCN-RT" \
--dataset "SMRT"
--num_layers 16 \
--hid_dim 200 \
--epochs 200 \
--lr 0.001 \
--batch_size 64\
--early_stop 30 \
--seed 1 
```
Alternatively, the train and transfer learning processes could also be started by the **shell** scripts using for loop. To run the training process on SMRT data set, using the following command:
```
sh scripts/train.sh
```


To run the transfer learning on nine transfer learning data sets, use:
```
sh scripts/transfer_learning.sh
```


## Model inference using the DeepGCN-RT model.
To run the inference code, the following command could be used:
```
python inference.py \
--SMILES "demo_SMILES"\
--model_path "model path"
```

