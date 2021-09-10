import torch
from torch_geometric.data import DataLoader

from tqdm import tqdm

from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset


dataset = PygPCQM4MDataset('../dataset')
split_idx = dataset.get_idx_split()

train_dataset = dataset[split_idx["train"]]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

train_node_num = []
for train_graph in tqdm(train_dataset):
    train_node_num.append(train_graph.num_nodes)
print(max(train_node_num))

valid_node_num = []
for valid_graph in tqdm(val_dataset):
    valid_node_num.append(valid_graph.num_nodes)
print(max(valid_node_num))

test_node_num = []
for test_graph in tqdm(test_dataset):
    test_node_num.append(test_graph.num_nodes)
print(max(test_node_num))
