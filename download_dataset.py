from ogb.utils import smiles2graph
from ogb.lsc import PygPCQM4MDataset
from ogb.lsc import DglPCQM4MDataset

# 全局取消证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# download and transform the dataset
pyg_dataset = PygPCQM4MDataset(root='./dataset', smiles2graph=smiles2graph)
# dgl_dataset = DglPCQM4MDataset(root='./dataset', smiles2graph=smiles2graph)

# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
# atom_encoder = AtomEncoder(emb_dim=100)
# bond_encoder = BondEncoder(emb_dim=100)

# atom_emb = atom_encoder(node_feat)
# edge_emb = bond_encoder(edge_feat)
