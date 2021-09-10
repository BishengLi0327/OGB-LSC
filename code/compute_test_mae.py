# 这个程序本身是用来算模型在测试集上的MAE的，但是数据集划分的测试集 本身并没有label，因此无法预测

import argparse
import os

import numpy as np
import random
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from gnn import GNN

from ogb.lsc import PygPCQM4MDataset
from ogb.lsc import PCQM4MEvaluator


def test(model, device, loader, evaluator):
    model.eval()
    y_pred = []
    y_true = []

    for step, batch in enumerate(tqdm(loader, desc='Iteration')):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {'y_true': y_true, 'y_pred': y_pred}

    return evaluator.eval(input_dict)['mae']


def main():
    parser = argparse.ArgumentParser(description="GNN baselines on pcqm4m with Pytorch Geometrics")
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    # parser.add_argument('--epochs', type=int, default=100,
    #                     help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    # parser.add_argument('--log_dir', type=str, default="",
    #                     help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='directory to save checkpoint')
    parser.add_argument('--save_mae_dir', type=str, default='', help='directory to save checkpoint for mae')
    # parser.add_argument('--save_test_dir', type=str, default='', help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else "cpu"

    dataset = PygPCQM4MDataset('../dataset')

    split_index = dataset.get_idx_split()

    test_loader = DataLoader(dataset[split_index["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    evaluator = PCQM4MEvaluator()

    if args.checkpoint_dir is not '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=True, **shared_params).to(device)
    else:
        raise ValueError('Invalid GNN type')

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f'Checkpoint file not found at {checkpoint_path}')

    # reading in checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Predicting on test data...')
    test_mae = test(model, device, test_loader, evaluator)
    print(test_mae)
    print('Saving checkpoint...')
    checkpoint_test = {'test_mae': test_mae}
    torch.save(checkpoint_test, os.path.join(args.save_mae_dir, 'checkpoint_test.pt'))


if __name__ == '__main__':
    main()
