import os.path as osp

from torch_geometric.data import DataLoader
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.datasets import GNNBenchmarkDataset
import torch_geometric.transforms as T
#from torch.utils.data import DataLoader
from torch_geometric.nn import BatchNorm, global_add_pool#, GCNConv
#from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from tqdm import tqdm
#from torch_geometric.nn import MessagePassing
import argparse


from typing import List, Optional, Set, Callable, get_type_hints
from torch_geometric.typing import Adj, Size
from torch_scatter.utils import broadcast

import os
import re
import inspect
import os.path as osp
from uuid import uuid1
from itertools import chain
from inspect import Parameter
from collections import OrderedDict

import torch
from torch import Tensor
from jinja2 import Template
from torch.utils.hooks import RemovableHandle
from torch_sparse import SparseTensor
#from torch_scatter import gather_csr, scatter, segment_csr

from torch_geometric.nn.conv.utils.helpers import expand_left
from torch_geometric.nn.conv.utils.jit import class_from_module_repr
from torch_geometric.nn.conv.utils.typing import (sanitize, split_types_repr, parse_types, resolve_types)
from torch_geometric.nn.conv.utils.inspector import Inspector, func_header_repr, func_body_repr
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add


argparser = argparse.ArgumentParser("multi-gpu training")
argparser.add_argument('--epochs', type=int, default=300)
argparser.add_argument('--hidden', type=int, default=5)
argparser.add_argument('--emb', type=int, default=75)
argparser.add_argument('--layers', type=int, default=4)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--dropout', type=float, default=0.0)
argparser.add_argument('--rank', type=int, default=15)
argparser.add_argument('--batch', type=int, default=1000)
args = argparser.parse_args()

train_dataset = GNNBenchmarkDataset(osp.join('data','MNIST'),name='MNIST',split='train', transform = T.Cartesian(cat=False))
val_dataset = GNNBenchmarkDataset(osp.join('data','MNIST'),name='MNIST',split='valid', transform = T.Cartesian(cat=False))
test_dataset = GNNBenchmarkDataset(osp.join('data','MNIST'),name='MNIST',split='test', transform = T.Cartesian(cat=False))
n_efeat = train_dataset[0].edge_attr.shape[1]
n_nfeat = train_dataset[0].x.shape[1]
n_class = train_dataset.num_classes

train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch)
test_loader = DataLoader(test_dataset, batch_size=args.batch)
    
class MessagePassing(torch.nn.Module):
    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    def __init__(self, aggr: Optional[str] = "add",
                 flow: str = "source_to_target", node_dim: int = -2):

        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)

        self.__user_args__ = self.inspector.keys(
            ['message', 'aggregate', 'update']).difference(self.special_args)


    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype == torch.long
            assert edge_index.dim() == 2
            assert edge_index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.index_select(self.node_dim, index)


    def __collect__(self, args, edge_index, size, x):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        for arg in args:
            if arg[-2:] not in ['_i', '_j']:

                pass
            else:
                dim = 0 if arg[-2:] == '_j' else 1
                data = x

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    #data = data[dim]
                    data_sum = data[dim]
                    data_prod = data[dim+1]

                if isinstance(data_sum, Tensor) and isinstance(data_prod, Tensor):

                    self.__set_size__(size, dim, data_sum)
                    data_sum = self.__lift__(data_sum, edge_index, j if arg[-2:] == '_j' else i)
                    data_prod = self.__lift__(data_prod, edge_index, j if arg[-2:] == '_j' else i)

        return data_sum, data_prod

    def propagate(self, edge_index: Adj, x, size: Size = None, edge_attr = None, norm=None):
        size = self.__check_input__(edge_index, size)

        if isinstance(edge_index, Tensor) or not self.fuse:
            x_sum,x_prod = self.__collect__(self.__user_args__, edge_index, size, x)
            x_sum = self.message(x_sum, edge_attr, norm)
            x_prod = self.message(x_prod, edge_attr, norm)
            x_sum, x_prod = self.aggregate((x_sum, x_prod), edge_index[1],ptr=None)

        return x_sum, x_prod

    def message_simple(self, x_j: Tensor) -> Tensor:
        return x_j


    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        
        return self.scatter_sum(inputs[0], index, dim=self.node_dim),self.scatter_product(inputs[1], index, dim=self.node_dim)

    def update(self, inputs: Tensor) -> Tensor:
        return inputs
    
    
    def scatter_sum(self, src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
        index = broadcast(index, src, dim)
        if out is None:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() == 0:
                size[dim] = 0
            else:
                size[dim] = int(index.max()) + 1
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index, src)
        else:
            return out.scatter_add_(dim, index, src)
    
    
    def scatter_product(self, src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None) -> torch.Tensor:
    
        index = broadcast(index, src, dim)
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.ones(size, dtype=src.dtype, device=src.device)
        out.scatter_(dim, index, src, reduce='multiply')
        return torch.nn.Parameter(out, requires_grad=True)
    
    
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, hidden_dim, rank_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.w1 = torch.nn.Linear(emb_dim, hidden_dim)
        self.w2 = torch.nn.Linear(emb_dim, rank_dim)
        self.v = torch.nn.Linear(rank_dim, hidden_dim)
        self.root_emb = torch.nn.Embedding(1, hidden_dim)
        #self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.v.reset_parameters()
        
    def gcn_norm(self,edge_index, edge_weight=None, num_nodes=None, dtype=None):

        if isinstance(edge_index, SparseTensor):
            adj_t = edge_index
            if not adj_t.has_value():
                adj_t = adj_t.fill_value(1., dtype=dtype)
            deg = sparsesum(adj_t, dim=1)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
            adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
            return adj_t

        else:
            num_nodes = maybe_num_nodes(edge_index, num_nodes)

            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                         device=edge_index.device)


            row, col = edge_index[0], edge_index[1]
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        x_sum, x_prod = self.w1(x),self.w2(x)
        #x_prod = self.w2(x)
        edge_embedding = edge_attr#self.bond_encoder(edge_attr.squeeze())

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        sum_agg, prod_agg = self.propagate(edge_index, x=(x_sum,x_prod), edge_attr = edge_attr, norm=norm)

        return self.v(prod_agg)+sum_agg + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

        return aggr_out

class Net(torch.nn.Module):
    def __init__(self,
                 emb_dim,
                 hidden_dim,
                 rank_dim,
                 n_layers,
                 dropout):
        super(Net, self).__init__()

        self.node_emb = torch.nn.Linear(n_nfeat, emb_dim)
        self.edge_emb = torch.nn.Linear(n_efeat, emb_dim)
        self.n_layers = n_layers
        self.dropout = dropout

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        #self.convs.append(GCNConv(emb_dim=n_nfeat,hidden_dim=emb_dim,rank_dim=emb_dim))
        #self.batch_norms.append(BatchNorm(emb_dim))
        for _ in range(n_layers):
            #conv = PNAConv(in_channels=75, out_channels=75,
            #               aggregators=aggregators, scalers=scalers, deg=deg,
            #               edge_dim=50, towers=5, pre_layers=1, post_layers=1,
            #               divide_input=False)
            #conv = GCNConv(emb_dim=emb_dim, hidden_dim=hidden_dim, rank_dim=rank_dim)
            conv = GCNConv(emb_dim=emb_dim, hidden_dim=emb_dim, rank_dim=emb_dim)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(emb_dim))

        self.mlp = Sequential(Linear(emb_dim, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, n_class))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for layer in range(self.n_layers):
            if layer == self.n_layers - 1:
                x = F.relu(self.batch_norms[layer](self.convs[layer](x, edge_index,edge_attr)))
            else:
                #x = F.relu(batch_norm[layer](conv[layer](x, edge_index, edge_attr)))
                x = F.dropout(F.relu(self.batch_norms[layer](self.convs[layer](x, edge_index,edge_attr))), self.dropout, training = self.training)

        #for conv, batch_norm in zip(self.convs, self.batch_norms):
        #    x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)


def train(model, epoch, device):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        out = F.log_softmax(out, dim=1)
        loss =  F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        out = F.log_softmax(out, dim=1).max(1)[1]
        correct += out.eq(data.y).sum().item()
    return correct / len(test_dataset)

@torch.no_grad()
def eval(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        out = F.log_softmax(out, dim=1).max(1)[1]
        correct += out.eq(data.y).sum().item()
    return correct / len(val_dataset)



if __name__ == '__main__':
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(emb_dim = args.emb,
                 hidden_dim = args.hidden,
                 rank_dim=args.rank,
                 n_layers=args.layers,
                 dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                  min_lr=0.00001)

    best_eval = 0
    best_test = 0
    for epoch in range(1, args.epochs):
        loss = train(model, epoch, device)
        val_mae = eval(model, val_loader, device)
        test_mae = test(model, test_loader, device)
        if val_mae > best_eval:
            best_eval = val_mae
            best_test = test_mae
        scheduler.step(val_mae)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}', f'Best val: {best_eval:.4f}', f'Best test: {best_test:.4f}')
