import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_scatter import scatter

from conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean
from torch_scatter.utils import broadcast

class graph_cp_pooling(torch.nn.Module):
    def __init__(self, in_fea, rank):
        super(graph_cp_pooling, self).__init__()
        self.W = torch.nn.Linear(in_fea, rank)

    def forward(self, x):
        #fea = torch.tanh(self.W(x)
        #fea = torch.prod(fea,0).unsqueeze(0)
        return torch.cat([torch.prod(torch.tanh(self.W(x_i)), 0).unsqueeze(0) for x_i in x])
        #readout = self.V(fea)
        #return fea
    
def sum_pool(x):
    #print(torch.cat([torch.sum(x_i, 0) for x_i in x]).shape)
    return torch.cat([torch.prod(torch.tanh(x_i), 0).unsqueeze(0) for x_i in x])
    #return scatter_add_(x, batch, dim=0, dim_size=size)


class GNN(torch.nn.Module):

    def __init__(self, num_tasks, device, num_layer = 5, emb_dim = 300, rank_dim = 64,
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.rank_dim = rank_dim
        self.device = device
        print(self.device)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, device=device, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            #self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
            self.graph_pred_linear = torch.nn.Linear(self.rank_dim, self.num_tasks)
        self.GraphCP = graph_cp_pooling(self.emb_dim, self.rank_dim)

    #def forward(self, batched_data):
    def forward(self, x, e_idx, e_attr=None):
        x, e_idx, e_attr = [x_i.to(self.device) for x_i in x], [e_i.to(self.device) for e_i in e_idx], [ea_i.to(self.device) for ea_i in e_attr]
        #h_node = self.gnn_node(batched_data)
        h_node = self.gnn_node(x, e_idx, e_attr)
        #h_node = h_node.to_data_list()
        #h_node = torch.cat(h_node)
        #h_graph = self.pool(h_node, batch_len)
        h_graph = self.GraphCP(h_node)
        #h_graph = sum_pool(h_node)
        #h_graph = torch.stack(h_graph)
        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_tasks = 10)
