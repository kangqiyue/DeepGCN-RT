import argparse
import copy
import time
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dgl.dataloading import GraphDataLoader

from dataset import  load_smrt_data_one_hot, get_node_dim, get_edge_dim
from dataset import get_node_dim, get_edge_dim
from models import GATModel, GCNModel, GINModel, AttentivfFPModel, DeeperGCN
from train_func import count_parameters, count_no_trainable_parameters, count_trainable_parameters

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


# def train function
def train(model, device, dataloader, optimizer, loss_fn, loss_fn_MAE,):
    num_batches = len(dataloader)
    train_loss = 0
    train_loss_MAE = 0
    model.train()
    for step, (bg, labels) in enumerate(dataloader):

        bg = bg.to(device)
        labels = labels.reshape(-1, 1)
        labels = labels.to(device)
        # batched_graph = batched_graph.to(device)
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


def test(model,device, dataloader, loss_fn, loss_fn_MAE, return_pred=False):
    num_batches = len(dataloader)
    test_loss = 0
    test_loss_MAE = 0

    model.eval()
    with torch.no_grad():
        for step, (bg, labels) in enumerate(dataloader):
            # node_feats = [n.to(device) for n in node_feats]
            # edge_feats = [e.to(device) for e in edge_feats]
            bg = bg.to(device)
            labels= labels.reshape(-1,1)
            labels = labels.to(device)
            # batched_graph = batched_graph.to(device)
            pred = model(bg)
            test_loss += loss_fn(pred, labels).item()
            test_loss_MAE += loss_fn_MAE(pred, labels).item()

    test_loss /= num_batches
    test_loss_MAE /= num_batches
    return (test_loss, test_loss_MAE, labels, pred) if return_pred else (test_loss, test_loss_MAE)


def main():
    seed_torch(args.seed)

    #train args
    epochs = args.epochs
    early_stop = args.early_stop
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout
    num_layers = args.num_layers
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    loss_MAE = nn.L1Loss(reduction="mean")
    dataset_name = args.dataset
    # check cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #save path
    file_savepath =f"./output/GNN_TL/{args.model_name}_{dataset_name}_seed{args.seed}"
    result_save_path = f"./output/GNN_TL_result"
    if not os.path.isdir(file_savepath):
        os.makedirs(file_savepath)
    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)
    print(file_savepath)

    '''dataset'''
    from dataset import TLDataset
    dataset = TLDataset(name=dataset_name, raw_dir= "/data/users/kangqiyue/kqy/DEEPGNN_RT/dataset/10_subdataset/processed") #D:\Molecule\DEEPGNN_RT\test_data


    '''k fold validation'''
    result = []
    # Define the K-fold Cross Validator
    kfold =  KFold(n_splits=10, shuffle=True, random_state=0)
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'\nFOLD {fold}')
        print('--------------------------------')
        '''init model'''
        model = DeeperGCN(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hid_dim=200, num_layers=num_layers,
                          dropout=dropout, mlp_layers=args.mlp_layers)

        '''load best model params'''
        best_model_path = "/data/users/kangqiyue/kqy/DEEPGNN_RT/output/GNN_DEEPGNN_mlp1_layer_16_lr_0.001_seed_1/best_model_weight.pth"
        checkpoint = torch.load(best_model_path, map_location=device)  # 加载断点
        model.load_state_dict(checkpoint)  # 加载模型可学习参数
        print(f"model loaded from: {best_model_path}")
        model.to(device)

        print('----args----')
        print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))
        print('----model----')
        # print(model)
        print(f"---------params-------------")
        print(f"all params: {count_parameters(model)}\n"
              f"trainable params: {count_trainable_parameters(model)}\n"
              f"freeze params: {count_no_trainable_parameters(model)}\n")

        '''data_loader'''
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, sampler=train_subsampler)
        test_dataloader = GraphDataLoader(dataset, batch_size=len(test_subsampler), sampler = test_subsampler, shuffle=False)

        # log_file
        best_loss = float("inf")
        best_model_stat = copy.deepcopy(model.state_dict())
        times, log_file = [], []

        print('---------- Training ----------')
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, verbose=True)

        # training & validation & testing
        for i in range(epochs):
            t1 = time.time()
            train_loss = train(model, device, train_dataloader, optimizer, loss_fn, loss_MAE)
            test_loss, test_MAE = test(model, device, test_dataloader, loss_fn, loss_MAE)
            t2 = time.time()
            times.append(t2 - t1)

            print(
                f'Epoch {i} |lr: {optimizer.param_groups[0]["lr"]:.6f} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | '
                f'Test_MAE: {test_MAE:.4f} | time/epoch: {sum(times) / len(times):.1f}')

            # local file log
            log_file_loop = [i, optimizer.param_groups[0]["lr"], train_loss, test_loss, test_MAE]
            log_file.append(log_file_loop)
            scheduler.step()

            if test_loss < best_loss:
                es = 0
                best_loss = test_loss
                best_model_stat = copy.deepcopy(model.state_dict())
            else:
                es += 1
                print("Counter {} of {}".format(es, early_stop))
                # early stopping
                if es > early_stop:
                    print("Early stop, best_loss: ", best_loss)
                    break

        save_log_file(best_model_stat, log_file, file_savepath, fold)

        #pred

        pred_summary = return_prediction(model, best_model_stat, device, file_savepath, loss_fn, loss_MAE, fold, test_dataloader)

        info = {"seed": args.seed,
                "fold_num": fold,
                # "num of compound": num_com,
                # "rt_range": rt_range,
                # "len_train": len(x_train),
                # "len_test": len(x_test)
                }

        mae_result = {**info, **pred_summary}
        mae_result = pd.DataFrame(mae_result, index=[0])
        result.append(mae_result)
    result1 = pd.concat(result)
    result1.to_csv(os.path.join(result_save_path, f"{dataset_name}_result_10_cv_seed_{args.seed}.csv"))

#
# def train_with_es(model, device, train_dataloader, test_dataloader, lr, loss_fn, loss_MAE, epochs=200, early_stop=30,):
#     # set seed
#     # seed_torch(seed=args.seed)
#     # # wandb login
#     # with open("wandb_login/wandb.env") as f:
#     #     for line in f:
#     #         key, val = line.strip().split("=", 1)
#     #         os.environ[key] = val
#     # wandb.init(project="deepGNN_22_4_13", entity="qwertyer")
#     # args.name = f"{args.model_name}_{args.num_layers}_lr_{args.lr}_seed_{args.seed}" + "_transfer_learning"
#     # wandb.run.name = args.name
#     # wandb.config.update(args)  # adds all of the arguments as config variables
#     print(args)
#
#     #mdoel
#     model.to(device)
#
#     #log_file
#     best_loss = float("inf")
#     # best_model = copy.deepcopy(model)
#     times, log_file = [], []
#
#     print('---------- Training ----------')
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, verbose=True)
#
#     # training & validation & testing
#     for i in range(epochs):
#         t1 = time.time()
#         train_loss = train(model, device, train_dataloader, optimizer, loss_fn, loss_MAE)
#         test_loss, test_MAE = test(model, device, test_dataloader, loss_fn, loss_MAE)
#         t2 = time.time()
#         times.append(t2 - t1)
#
#         print(f'Epoch {i} |lr: {optimizer.param_groups[0]["lr"]:.6f} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | '
#               f'Test_MAE: {test_MAE:.4f} | time/epoch: {sum(times) / len(times):.1f}')
#         # #wandb log
#         # wandb.log({
#         #     "lr":optimizer.param_groups[0]["lr"],
#         #     "train_loss": train_loss,
#         #     "test_loss": test_loss,
#         #     "test_MAE": test_MAE
#         # })
#
#         #local file log
#         log_file_loop = [i, optimizer.param_groups[0]["lr"], train_loss, test_loss, test_MAE]
#         log_file.append(log_file_loop)
#         scheduler.step()
#
#         if test_loss < best_loss:
#             es = 0
#             best_loss = test_loss
#             best_model_stat = copy.deepcopy(model.state_dict())
#         else:
#             es += 1
#             print("Counter {} of {}".format(es, early_stop))
#             # early stopping
#             if es > early_stop:
#                 print("Early stop, best_loss: ", best_loss)
#                 break
#
#     return (log_file, best_model_stat)


def save_log_file(best_model_stat, log_file, file_savepath, fold):
    result = pd.DataFrame(log_file)
    result.columns = ["epoch", "lr", "train_loss",  "test_loss", "test_MAE"]
    result.to_csv(os.path.join(file_savepath, f"fold_{fold}_log_file.csv"))

    # print min
    index = result.iloc[:,3].idxmin(axis =0)
    print("the index of min loss is shown as follows:",result.iloc[index, :])
    torch.save(best_model_stat, os.path.join(file_savepath, f"fold_{fold}_best_model_weight.pth"))

    # return result["test_loss"].idxmin(axis =0)


def return_prediction(model, best_model_stat, device, file_savepath, loss_fn, loss_MAE, fold, dataloader):
    # best_path = os.path.join(file_savepath, f"fold_{fold}_best_model_weight.pth")
    model.load_state_dict(best_model_stat)

    model.to(device)
    _, _, pred, y = test(model, device, dataloader, loss_fn, loss_MAE, return_pred=True)
    y = y.reshape(-1, 1).cpu()
    pred = pred.reshape(-1, 1).cpu()
    print(f"y_test's shape: {y.shape}; pred's shape: {pred.shape}")

    # save pred dataset
    r2 = torch.cat([y, pred], dim=1)
    r2 = pd.DataFrame(r2.cpu().numpy())
    r2.columns = ["y_label", "pred"]
    r2.to_csv(os.path.join(file_savepath ,f"fold_{fold}_pred_data.csv"))

    from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    rt_summary = {
        "mean_absolute_error": mean_absolute_error(y, pred),
        "median_absolute_error": median_absolute_error(y, pred),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y, pred),
        "r2_score": r2_score(y, pred),
        "mean_squared_error": mean_squared_error(y, pred)
    }
    return rt_summary



if __name__ == '__main__':
    """
    Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description='GNN_RT_MODEL')
    #wandb name, dataset name, model name
    parser.add_argument('--name', type=str, default="test", help='wandb_running_name')
    parser.add_argument('--dataset', type=str, default='transfer learning', help='Name of dataset.')
    parser.add_argument('--model_name', type=str, default='DEEPGNN', help='Name of model, choose from: GAT, GCN, GIN, AFP, DEEPGNN')

    # GNN model args
    parser.add_argument('--num_layers', type=int, default=16, help='Number of GNN layers.')
    parser.add_argument('--mlp_layers', type=int, default=1, help='Number of MLP layers in DEEPGNN.')
    parser.add_argument('--hid_dim', type=int, default=200, help='Hidden channel size.')

    # training args
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--early_stop', type=int, default=30, help='Early stop epoch.')

    parser.add_argument('--seed', type=int, default=0, help='set seed')

    args = parser.parse_args()
    print(args)

    main()

