import argparse
import copy
import time
import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dgl.dataloading import GraphDataLoader

from dataset import  load_smrt_data_one_hot, get_node_dim, get_edge_dim
from dataset import get_node_dim, get_edge_dim
from models import GATModel, GCNModel, GINModel, AttentivfFPModel, DeeperGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def train(model, device, dataloader, optimizer, loss_fn, loss_fn_MAE,):
    num_batches = len(dataloader)
    train_loss = 0
    train_loss_MAE = 0
    model.train()
    for step, (bg, labels) in enumerate(dataloader):

        bg = bg.to(device)
        labels = labels.reshape(-1, 1)
        labels = labels.to(device)
        pred = model(bg)
        loss = loss_fn(pred, labels)
        # runing loss in each batch
        train_loss += loss.item()
        # MAE Loss
        loss_MAE = loss_fn_MAE(pred, labels)
        train_loss_MAE += loss_MAE.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / num_batches


def test(model,device, dataloader, loss_fn, loss_fn_MAE):
    num_batches = len(dataloader)
    test_loss = 0
    test_loss_MAE = 0

    model.eval()
    with torch.no_grad():
        for step, (bg, labels) in enumerate(dataloader):
            bg = bg.to(device)
            labels= labels.reshape(-1,1)
            labels = labels.to(device)
            pred = model(bg)
            test_loss += loss_fn(pred, labels).item()
            test_loss_MAE += loss_fn_MAE(pred, labels).item()
    test_loss /= num_batches
    test_loss_MAE /= num_batches
    return test_loss, test_loss_MAE


def main():
    import wandb

    #set seed
    seed_torch(seed=args.seed)
    args.name = f"{args.model_name}_{args.num_layers}_lr_{args.lr}_seed_{args.seed}"
    print(args)

    #train args
    batch_size = args.batch_size
    early_stop = args.early_stop
    lr = args.lr
    dropout = args.dropout
    num_layers = args.num_layers
    model_name = args.model_name
    hid_dim = args.hid_dim
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    loss_MAE = nn.L1Loss(reduction="mean")
    # check cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #save path
    file_savepath =f"./output/GNN_4_16_{args.model_name}_layer_{args.num_layers}_lr_{args.lr}_seed_{args.seed}"
    if not os.path.isdir(file_savepath):
        os.makedirs(file_savepath)
    print(file_savepath)


    '''dataset and data_loader'''
    train_dataset, valid_dataset, test_dataset = load_smrt_data_one_hot(random_state=args.seed)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_dataloader = GraphDataLoader(test_dataset, batch_size=len(test_dataset))

    '''init model'''
    if model_name == "GAT":
        model = GATModel(node_in_dim=get_node_dim(), hidden_feats = [200]*num_layers)

    elif model_name == "GCN":
        model = GCNModel(node_in_dim=get_node_dim(), hidden_feats = [200]*num_layers)

    elif model_name == "GIN":
        full_atom_feature_dims = get_node_dim()
        full_bond_feature_dims = get_edge_dim()
        model = GINModel(num_node_emb=full_atom_feature_dims, num_edge_emb=full_bond_feature_dims, num_layers=num_layers, emb_dim=200, dropout=dropout)

    elif model_name == "DEEPGNN":
        model = DeeperGCN(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hid_dim=200,num_layers=num_layers, dropout=dropout, mlp_layers=args.mlp_layers)

    model.to(device)

    print('----args----')
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))
    print('----model----')
    print(model)
    print(f"---------params-------------")
    from train_func import count_parameters, count_no_trainable_parameters, count_trainable_parameters
    print(f"all params: {count_parameters(model)}\n"
          f"trainable params: {count_trainable_parameters(model)}\n"
          f"freeze params: {count_no_trainable_parameters(model)}\n")

    #log_file
    best_loss = float("inf")
    best_model = copy.deepcopy(model)
    times, log_file = [], []

    print('---------- Training ----------')
    #optim
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, verbose=True)
    # training & validation & testing
    for i in range(args.epochs):
        t1 = time.time()
        train_loss = train(model, device, train_dataloader, optimizer, loss_fn, loss_MAE)
        t2 = time.time()
        times.append(t2 - t1)

        valid_loss, valid_MAE = test(model, device, valid_dataloader, loss_fn, loss_MAE)
        test_loss, test_MAE = test(model, device, test_dataloader, loss_fn, loss_MAE)

        print(f'Epoch {i} |lr: {optimizer.param_groups[0]["lr"]:.6f} | Train: {train_loss:.4f} | Valid: {valid_loss:.4f} | Test: {test_loss:.4f} | '
              f'Valid_MAE: {valid_MAE:.4f} | Test_MAE: {test_MAE:.4f} |'
              f'time/epoch: {sum(times) / len(times):.1f}')

        #local file log
        log_file_loop = [i, optimizer.param_groups[0]["lr"], train_loss, valid_loss, test_loss, valid_MAE, test_MAE]
        log_file.append(log_file_loop)
        scheduler.step()

        if valid_loss < best_loss:
            es = 0
            best_loss = valid_loss
            best_model = copy.deepcopy(model)
        else:
            es += 1
            print("Counter {} of {}".format(es, early_stop))
            # early stopping
            if es > early_stop:
                print("Early stop, best_loss: ", best_loss)
                break

    #result
    import pandas as pd
    result = pd.DataFrame(log_file)
    result.columns = ["epoch", "lr", "train_loss", "valid_loss", "test_loss", "valid_MAE", "test_MAE"]
    result.to_csv(file_savepath + "/log_file.csv")
    with open(file_savepath+"/namespace.txt", "w") as f:
        f.write(str(vars(args)))
    index = result.iloc[:,3].idxmin(axis =0)
    print("the index of min loss is shown as follows:",result.iloc[index, :])
    torch.save(best_model.state_dict(), file_savepath + "/best_model_weight.pth")


if __name__ == '__main__':
    """
    Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description='GNN_RT_MODEL')
    #wandb name, dataset name, model name
    parser.add_argument('--name', type=str, default="test", help='wandb_running_name')
    parser.add_argument('--dataset', type=str, default='SMRT', help='Name of dataset.')
    parser.add_argument('--model_name', type=str, default='DEEPGNN', help='Name of model, choose from: GAT, GCN, GIN, AFP, DEEPGNN')

    # GNN model args
    parser.add_argument('--num_layers', type=int, default=5, help='Number of GNN layers.')
    parser.add_argument('--mlp_layers', type=int, default=1, help='Number of MLP layers in DEEPGNN.')
    parser.add_argument('--hid_dim', type=int, default=200, help='Hidden channel size.')

    # training args
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--early_stop', type=int, default=30, help='Early stop epoch.')

    parser.add_argument('--seed', type=int, default=1, help='set seed')

    args = parser.parse_args()
    print(args)

    main()

