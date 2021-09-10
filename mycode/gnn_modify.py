import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from pooling import concat_pool
from conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

# from sklearn.linear_model import Lasso
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import StandardScaler
#
#
# def LassoRegression(degree, alpha):
#     return Pipeline([
#         ("poly", PolynomialFeatures(degree=degree)),
#         ("std_scaler", StandardScaler()),
#         ("lasso_reg", Lasso(alpha=alpha))
#     ])


# 将node embedding转化为graph embedding
class GNN(torch.nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300,
                 gnn_type='gin', virtual_node=True, residual=False,
                 drop_ratio=0, JK='last', graph_pooling='sum'):
        """
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        """
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, JK=JK,
                                                 drop_ratio=drop_ratio, residual=residual, gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layers, emb_dim, JK=JK,
                                     drop_ratio=drop_ratio, residual=residual, gnn_type=gnn_type)

        # 需要修改的部分，在pooling上做文章，将所有的node_embedding concat起来，形成一个长向量，不足的用0补全

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif self.graph_pooling == 'max':
            self.pool = global_max_pool
        elif self.graph_pooling == 'attention':
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim),
                                                                    torch.nn.BatchNorm1d(2*emb_dim),
                                                                    torch.nn.ReLU(),
                                                                    torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == 'set2set':
            self.pool = Set2Set(emb_dim, processing_steps=2)
        elif self.graph_pooling == 'concat':
            self.pool = concat_pool
        else:
            raise ValueError("Invalid graph pooling type")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        elif graph_pooling == 'concat':
            # 这块也要修改，打算做一个lasso回归
            self.graph_pred_linear = torch.nn.Linear(51 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        # lasso_reg = LassoRegression(1000, 0.01)
        # lasso_reg.fit(h_graph.cpu().detach().numpy(), batched_data.y.cpu().detach().numpy())
        # output = lasso_reg.predict(h_graph)

        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            return torch.clamp(output, min=0, max=50)


if __name__ == '__main__':
    GNN(num_tasks=1)
