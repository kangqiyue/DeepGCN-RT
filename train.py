import argparse
import copy
import time

import dgl
import numpy as np
import random
import os
import pandas as pd
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dgl.dataloading import GraphDataLoader
from dgl.nn import SumPooling, AvgPooling

from dataset import  load_smrt_data_one_hot, get_node_dim, get_edge_dim
from dataset import get_node_dim, get_edge_dim
from models import GATModel, GCNModel, GINModel, AttentivfFPModel, DeeperGCN, GCNModelAFPreadout, GCNModelWithEdgeAFPreadout
from utils import count_parameters, count_no_trainable_parameters, count_trainable_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    dgl.seed(seed)
    dgl.random.seed(seed)



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


def test(model,device, dataloader, loss_fn, loss_fn_MAE, return_pred=False):
    num_batches = len(dataloader)
    test_loss = 0
    test_loss_MAE = 0
    preds = []
    labels_all = []

    model.eval()
    with torch.no_grad():
        for step, (bg, labels) in enumerate(dataloader):
            bg = bg.to(device)
            labels= labels.reshape(-1,1)
            labels = labels.to(device)
            pred = model(bg)

            preds.append(pred)
            labels_all.append(labels)
            test_loss += loss_fn(pred, labels).item()
            test_loss_MAE += loss_fn_MAE(pred, labels).item()

    test_loss /= num_batches
    test_loss_MAE /= num_batches
    return (test_loss, test_loss_MAE, labels_all, preds) if return_pred else (test_loss, test_loss_MAE)


def main():
    import wandb
    #set seed
    seed_torch(seed=args.seed)

    #wandb login
    with open("wandb_login/wandb.env") as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            os.environ[key] = val
    wandb.init(project="deepGNN_22_8_3_inference", entity="qwertyer")
    args.name = f"{args.model_name}_{args.norm}_{args.num_layers}_lr_{args.lr}_seed_{args.seed}"
    wandb.run.name = args.name
    wandb.config.update(args)  # adds all of the arguments as config variables
    print(args)

    #train args
    batch_size = args.batch_size
    early_stop = args.early_stop
    lr = args.lr
    model_name = args.model_name
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    loss_MAE = nn.L1Loss(reduction="mean")
    # check cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #save path
    file_savepath =f"./output/deep_gnn_22_8_3/{args.model_name}/{args.name}"
    if not os.path.isdir(file_savepath):
        os.makedirs(file_savepath)
    print(file_savepath)


    '''dataset and data_loader'''
    train_dataset, valid_dataset, test_dataset = load_smrt_data_one_hot(random_state=args.seed, raw_dir = "dataset")
    train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_dataloader = GraphDataLoader(test_dataset, batch_size=len(test_dataset))

    '''init model'''
    if model_name == "GCN_attention_GRU":
        model = GCNModelAFPreadout(node_in_dim=get_node_dim(), hidden_feats=[args.hid_dim]*args.num_layers, output_norm=args.norm, gru_out_layer=args.gru_out_layer, dropout=args.dropout)
    elif model_name == "GCN_edge_attention_GRU":
        model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hidden_feats=[args.hid_dim]*args.num_layers, output_norm=args.norm,
                                           gru_out_layer= args.gru_out_layer, update_func=args.update_func, dropout=args.dropout)
    elif model_name == "GCN_edge_attention_GRU_without_residual":
        model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hidden_feats=[args.hid_dim]*args.num_layers, output_norm=args.norm,
                                           residual=False, gru_out_layer= args.gru_out_layer, update_func=args.update_func, dropout=args.dropout)
    elif model_name == "GCN_edge_mean":
        model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hidden_feats=[args.hid_dim]*args.num_layers, output_norm=args.norm,
                                           gru_out_layer= args.gru_out_layer, update_func=args.update_func, dropout=args.dropout)
        model.readout = AvgPooling()
    elif model_name == "GCN_edge_sum":
        model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hidden_feats=[args.hid_dim]*args.num_layers, output_norm=args.norm,
                                           gru_out_layer= args.gru_out_layer, update_func=args.update_func,dropout=args.dropout)
        model.readout = SumPooling()
    # elif model_name == "GCN_edge_attention_GRU_1_layer":
    #     args.gru_out_layer = 1
    #     wandb.config.update(args, allow_val_change=True)
    #     model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hidden_feats=[200]*num_layers, output_norm=args.norm,
    #                                        gru_out_layer= args.gru_out_layer, update_func=args.update_func)
    elif model_name == "GCN_edge_attention_GRU_no_denselayer":
        model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hidden_feats=[args.hid_dim]*args.num_layers, output_norm=args.norm,
                                           gru_out_layer= args.gru_out_layer, update_func=args.update_func,dropout=args.dropout)
        model.out = nn.Linear(200, 1)
    else:
        raise NotImplementedError(f'Aggregator {args.model_name} is not supported.')


    '''inference'''
    if args.inference:

        all_infer_result = pd.DataFrame()
        '''save folder'''
        metric_file_savepath = 'output/SMRT_result_metrics_22_8_22'
        if not os.path.isdir(metric_file_savepath):
            os.makedirs(metric_file_savepath)
        print(metric_file_savepath)

        '''load best model params'''
        # best_model_path = "/data/users/kangqiyue/kqy/DEEPGNN_RT/output/GNN_DEEPGNN_mlp1_layer_16_lr_0.001_seed_1/best_model_weight.pth"
        best_model_path = os.path.join(file_savepath, "best_model_weight.pth")
        checkpoint = torch.load(best_model_path, map_location=device)  # 加载断点
        model.load_state_dict(checkpoint)  # 加载模型可学习参数
        print(f"model loaded from: {best_model_path}")
        model.to(device)
        for i, dataloader in enumerate([train_dataloader, valid_dataloader, test_dataloader]):
            _, _, y, pred = test(model, device, dataloader, loss_fn, loss_MAE, return_pred=True)
            #convert list to tensor
            y = torch.cat(y)
            pred = torch.cat(pred)

            y = y.reshape(-1, 1).cpu()
            pred = pred.reshape(-1, 1).cpu()
            print(f"y_test's shape: {y.shape}; pred's shape: {pred.shape}")

            # save pred dataset
            result = torch.cat([y, pred], dim=1)
            result = pd.DataFrame(result.cpu().numpy())
            result.columns = ["y_label", "pred"]
            # result.to_csv(os.path.join(file_savepath, f"SMRT_train_prediction_result.csv"))
            result.to_csv(os.path.join(metric_file_savepath, args.name+f"_{i}_prediction_data.csv" ))

            rt_summary = {
                "mean_absolute_error": mean_absolute_error(y, pred),
                "median_absolute_error": median_absolute_error(y, pred),
                "mean_absolute_percentage_error": mean_absolute_percentage_error(y, pred),
                "r2_score": r2_score(y, pred),
                "mean_squared_error": mean_squared_error(y, pred),
                "model_path": best_model_path,
                "len of dataset": len(dataloader.dataset),
                "model_name":model_name,
                "gnn layers": args.num_layers,
                "num of index":i
            }
            print(f"---------------"
                  f"{best_model_path}; dataset {len(dataloader.dataset)}\n ")
            print(rt_summary)
            if i == 0:
                log_stats = {**{f'train/{k}': v for k, v in rt_summary.items()} }
            elif i == 1:
                log_stats = {**{f'valid/{k}': v for k, v in rt_summary.items()} }
            elif i == 2:
                log_stats = {**{f'test/{k}': v for k, v in rt_summary.items()} }
            else:
                raise NotImplementedError
            wandb.log(log_stats)

            print(f"---------------")
            rt_summary = pd.DataFrame.from_dict(rt_summary, orient="index")
            all_infer_result = pd.concat([all_infer_result, rt_summary], axis=1)

        all_infer_result.to_csv(os.path.join(metric_file_savepath, args.name+"_prediction_metrics.csv" ))
        return


    model.to(device)
    print('----args----')
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))
    print('----model----')
    print(model)
    print(f"---------params-------------")
    print(f"all params: {count_parameters(model)}\n"
          f"trainable params: {count_trainable_parameters(model)}\n"
          f"freeze params: {count_no_trainable_parameters(model)}\n")

    #log_file
    best_loss = float("inf")
    best_test_MAE = float("inf")
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
            best_test_MAE = test_MAE
            best_model = copy.deepcopy(model)
        else:
            es += 1
            print("Counter {} of {}".format(es, early_stop))
            # early stopping
            if es > early_stop:
                print("Early stop, best_loss: ", best_loss)
                break

        # wandb log
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "valid_MAE": valid_MAE,
            "test_MAE": test_MAE,
            "best_valid_loss": best_loss,
            "best_test_MAE": best_test_MAE
        })

    #result
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
    Model Parameters
    """
    parser = argparse.ArgumentParser(description='GNN_RT_MODEL')
    #wandb name, dataset name, model name
    parser.add_argument('--name', type=str, default="test", help='wandb_running_name')
    parser.add_argument('--dataset', type=str, default='SMRT', help='Name of dataset.')
    parser.add_argument('--model_name', type=str, default='GCN_edge_attention_GRU', help='Name of model, choose from: GAT, GCN, GIN, AFP, DEEPGNN')

    # GNN model args
    parser.add_argument('--num_layers', type=int, default=16, help='Number of GNN layers.')
    parser.add_argument('--gru_out_layer', type=int, default=2, help='readout layer')
    parser.add_argument('--norm', type=str, default='none', help='choose from: batch_norm, layer_norm, none')
    parser.add_argument('--update_func', type=str, default='no_relu', help='choose from: batch_norm, layer_norm, none')

    # training args
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--early_stop', type=int, default=30, help='Early stop epoch.')
    parser.add_argument('--seed', type=int, default=1, help='set seed')

    #inference or not
    parser.add_argument("--inference", action="store_true", help="Whether inference")

    args = parser.parse_args()
    print(args)

    main()

