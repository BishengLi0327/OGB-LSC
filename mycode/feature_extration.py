import torch
from torch_geometric.data import DataLoader

from gnn import GNN
import os
from tqdm import tqdm
import argparse
import numpy as np
import random
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import warnings
warnings.filterwarnings("ignore", category=Warning)


def main():
    # Model settings
    parser = argparse.ArgumentParser(description='GNN model settings.')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--gnn', type=str, default='gcn', help='GNN gin, gin-virtual, gcn, gcn-virtual')
    parser.add_argument('--graph_pooling', type=str, default='sum', help='graph pooling strategy mean or sum')
    parser.add_argument('--drop_ratio', type=float, default=0, help='dropout ratio')
    parser.add_argument('--num_layers', type=int, default=5, help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=600, help='dimensionality of hidden units in GNNs')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--model_dir', type=str, default='', help='model log directory')
    args = parser.parse_args()

    print(args)

    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed(17)
    random.seed(17)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

    dataset = PygPCQM4MDataset(root='../dataset')

    split_idx = dataset.get_idx_split()

    train_dataset = dataset[split_idx['train']]  # 训练集中有3045360张图
    valid_dataset = dataset[split_idx['valid']]  # 验证集中有380670张图
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=False, **shared_params).to(device)
        model_path = os.path.join(args.model_dir, 'gin_checkpoint.pt')
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
        model_path = os.path.join(args.model_dir, 'ginv_checkpoint.pt')
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=False, **shared_params).to(device)
        model_path = os.path.join(args.model_dir, 'gcn_checkpoint.pt')
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=True, **shared_params).to(device)
        model_path = os.path.join(args.model_dir, 'gcnv_checkpoint.pt')
    else:
        raise ValueError("Invalid GNN type")

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            batch = batch.to(device)
            features = model(batch)
            if i == 0:
                train_features = features
                train_labels = batch.y
            else:
                train_features = torch.cat([train_features, features], 0)
                train_labels = torch.cat([train_labels, batch.y], 0)
    print(train_features.shape)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_loader, desc='Iteration')):
            batch = batch.to(device)
            features = model(batch)
            if i == 0:
                valid_features = features
                valid_labels = batch.y
            else:
                valid_features = torch.cat([valid_features, features], 0)
                valid_labels = torch.cat([valid_labels, batch.y], 0)
    print(valid_features.shape)

    train_features = train_features.data.cpu().numpy()
    train_labels = train_labels.data.cpu().numpy()
    # 将特征分开存储
    for i in range(len(train_features)+1):
        if i > 0 and i % 304536 == 0:
            tmp = list(zip(train_features[int((i / 304536 - 1) * 304536): i], train_labels[int((i / 304536 - 1) * 304536): i]))

            np.savetxt(args.gnn + '_train_dataset' + str(int(i / 304536)) + '.txt', tmp, fmt='%s')

    # 将所有的特征存在一个文件里
    # np.savetxt(args.gnn + '_train_features.txt', train_features)
    # np.savetxt(args.gnn + '_train_labels.txt', train_labels)

    valid_features = valid_features.data.cpu().numpy()
    valid_labels = valid_labels.data.cpu().numpy()
    for i in range(len(valid_features)+1):
        if i > 0 and i % 38067 == 0:
            tmp = list(zip(valid_features[int((i / 38067 - 1) * 38067): i], valid_labels[int((i / 38067 - 1) * 38067): i]))
            np.savetxt(args.gnn + '_valid_dataset' + str(int(i / 38067)) + '.txt', tmp, fmt='%s')

    # np.savetxt(args.gnn + '_valid_features.txt', valid_features)
    # np.savetxt(args.gnn + '_valid_labels.txt', valid_labels)


if __name__ == '__main__':
    main()
