import torch
import argparse
from models import DeeperGCN
from dataset import smiles2graph, feature_to_dgl_graph,get_node_dim, get_edge_dim


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeeperGCN(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hid_dim=200,num_layers=16, dropout=0, mlp_layers=1)

    '''load best model params'''
    # best_model_path = "/data/users/kangqiyue/kqy/DEEPGNN_RT/output/GNN_DEEPGNN_mlp1_layer_16_lr_0.001_seed_1/best_model_weight.pth"
    # best_model_path = "D:\DEEPGNN_RT\output\DeepGNN-RT\GNN_DEEPGNN_layer_16_lr_0.001_seed_1\\best_model_weight.pth"
    best_model_path = args.model_path
    checkpoint = torch.load(best_model_path, map_location=device)  # 加载断点
    model.load_state_dict(checkpoint)  # 加载模型可学习参数
    print(f"model loaded from: {best_model_path}")
    model.to(device)

    # test = "O=C(c1ccc2c(c1)N(CC(O)=NCc1ccco1)C(=O)[C@@H]1CCCCN21)N1CCCCC1"
    test = args.SMILES
    g = feature_to_dgl_graph(smiles2graph(test))
    g = g.to(device)

    model.eval()
    with torch.no_grad():
        output = model(g)
        print(output)
        print(f"The predicted RT for {test}\nis \n {output.cpu().numpy()}")



if __name__ == '__main__':
    """
    Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description='GNN_RT_MODEL_inference')
    parser.add_argument('--SMILES', type=str, help='SMILES of the small molecule')
    parser.add_argument('--model_path', type=str, help='model path for DeepGNN-RT')
    args = parser.parse_args()
    print(args)

    main()


