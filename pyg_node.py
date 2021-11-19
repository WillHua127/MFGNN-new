from __future__ import division
from __future__ import print_function
import matplotlib
import itertools

import os.path as osp
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import torch
from torch import Tensor
from jinja2 import Template
from torch.utils.hooks import RemovableHandle
from torch_sparse import SparseTensor
#from torch_scatter import gather_csr, scatter, segment_csr
from typing import List, Optional, Set, Callable, get_type_hints
from torch_geometric.typing import Adj, Size
from torch_scatter.utils import broadcast
from torch_geometric.datasets import Planetoid,WebKB,WikipediaNetwork

import torch_geometric

from torch_geometric.nn.conv.utils.helpers import expand_left
from torch_geometric.nn.conv.utils.jit import class_from_module_repr
from torch_geometric.nn.conv.utils.typing import (sanitize, split_types_repr, parse_types, resolve_types)
from torch_geometric.nn.conv.utils.inspector import Inspector, func_header_repr, func_body_repr
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.nn.inits import zeros


from utils import load_data, accuracy, full_load_data, data_split, random_disassortative_splits, rand_train_test_idx, load_graph_data, semi_supervised_splits
#from models import CPPooling, TwoCPPooling, GAT



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--idx', type=int, default=0,
                    help='Split number.')
parser.add_argument('--dataset_name', type=str,
                    help='Dataset name.', default = 'cornell')
parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
parser.add_argument("--heads", type=int, default=8,
                    help="number of hidden attention heads")
parser.add_argument("--out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--layers", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument('--sub_dataname', type=str,
                    help='subdata name.', default = 'DE')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--task', type=str,
                    help='semi-supervised learning or supervised learning.', default = 'sl')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
    
    
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

                #if isinstance(data, Tensor):
                if isinstance(data_sum, Tensor) and isinstance(data_prod, Tensor):
                    self.__set_size__(size, dim, data_sum)
                    data_sum = self.__lift__(data_sum, edge_index, j if arg[-2:] == '_j' else i)
                    data_prod = self.__lift__(data_prod, edge_index, j if arg[-2:] == '_j' else i)
        return data_sum, data_prod

    def propagate(self, edge_index: Adj, x, size: Size = None, edge_attr = None, norm=None):

        size = self.__check_input__(edge_index, size)

        if isinstance(edge_index, Tensor) or not self.fuse:
            x_sum,x_prod = self.__collect__(self.__user_args__, edge_index, size, x)
            x_sum = self.message(x_sum)
            x_prod = self.message(x_prod)
            x_sum, x_prod = self.aggregate((x_sum, x_prod), edge_index[1],ptr=None)

        return x_sum, x_prod

    def message(self, x_j: Tensor) -> Tensor:
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
        self.w2 = torch.nn.Linear(emb_dim+1, rank_dim)
        self.v = torch.nn.Linear(rank_dim, hidden_dim)
        self.att1= torch.nn.Linear(hidden_dim, 1, bias=False)
        self.att2 = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.att_vec = torch.nn.Linear(2, 2, bias=False)
        self.root_emb = torch.nn.Embedding(1, hidden_dim)
        #self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        #self.bias = torch.nn.Parameter(torch.Tensor(hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.att1.reset_parameters()
        self.att2.reset_parameters()
        self.att_vec.reset_parameters()
        self.v.reset_parameters()
        #zeros(self.bias)

    def attention(self, prod, bias):
        T = 2
        att = torch.softmax(self.att_vec(torch.sigmoid(torch.cat([self.att1(prod) ,self.att2(bias)],1)))/T,1)
        return att[:,0][:,None],att[:,1][:,None]

    def forward(self, x, edge_index):
        x_sum, x_prod = self.w1(x),self.w2(torch.cat((x, torch.ones([x.shape[0],1]).to('cuda:0')),1))#self.w2(x)
        #x_sum, x_prod = self.w1(x),torch.tanh(self.w2(torch.cat((x, torch.ones([x.shape[0],1])),1)))#self.w2(x)
        #x_prod = self.w2(x)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        #deg = degree(row, x.size(0), dtype = x.dtype) + 1
        #deg_inv_sqrt = deg.pow(-0.5)
        #deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        #norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        #edge_index, edge_attr = self.gcn_norm(edge_index,edge_weight=edge_attr)
        sum_agg, prod_agg = self.propagate(edge_index, x=(x_sum,x_prod))
        prod_agg = self.v(prod_agg)
        #rst = prod_agg
        att_prod, att_sum = self.attention(prod_agg, sum_agg)
        rst = att_prod*prod_agg + att_sum*sum_agg
        

        return rst
    

    def update(self, aggr_out):
        return aggr_out
    
class GCN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 dropout):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.dropout = dropout
        feat_drop = dropout
        # input projection (no residual)
        self.gat_layers.append(GCNConv(
            in_dim, num_hidden,num_hidden))

        for l in range(1, num_layers):
            self.gat_layers.append(GCNConv(
                num_hidden, num_hidden, num_hidden))
        # output projection
        #self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes))
        self.gat_layers.append(GCNConv(
            num_hidden, num_classes, num_hidden))

    def forward(self, x, edge_index):
        h = x
        for l in range(self.num_layers):
            h = F.relu(self.gat_layers[l](h, edge_index))
        # output projection
        logits = self.gat_layers[-1](h, edge_index)
        return logits

# Load data
#edge_dict, features, labels, edge_index = full_load_data(args.dataset_name, args.sub_dataname)

if args.dataset_name in {'cora', 'citeseer','pubmed'}:
    data = Planetoid(osp.join('torch_geometric_data',args.dataset_name),name=args.dataset_name) 
elif args.dataset_name in {'cornell','texas','wisconsin'}:
    data = WebKB(osp.join('torch_geometric_data',args.dataset_name),name=args.dataset_name)
elif args.dataset_name in {'squirrel','chameleon'}:
    data = WikipediaNetwork(osp.join('torch_geometric_data',args.dataset_name),name=args.dataset_name)

edge = torch_geometric.utils.add_self_loops(data[0].edge_index)[0]
#edge = data[0].edge_index
labels = data[0].y
features = data[0].x
#norm = g.ndata.pop('norm')
    
num_class = labels.max()+1

if args.cuda:
    edge = edge.cuda()
    features = features.cuda()
    #adj = adj.cuda()
    labels = labels.cuda()
    #norm = norm.cuda()
    #idx_train = idx_train.cuda()
    #idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()

    
def test(model, idx_train, idx_val, idx_test):
    model.eval()
    output = model(features,edge)
    pred = torch.argmax(F.softmax(output,dim=1) , dim=1)
    pred = F.one_hot(pred).float()
    output = F.log_softmax(output, dim=1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test
    
    
def train_supervised():
    patience = 50
    best_result = 0
    best_std = 0
    best_dropout = None
    best_weight_decay = None
    best_lr = None
    best_time = 0
    best_epoch = 0

    lr = [0.05, 0.01,0.002]#,0.01,
    weight_decay = [1e-4,5e-4,5e-5, 5e-3] #5e-5,1e-4,5e-4,1e-3,5e-3
    dropout = [0.1, 0.2, 0.3]#, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9]
    for args.lr, args.weight_decay in itertools.product(lr, weight_decay):
        result = np.zeros(10)
        t_total = time.time()
        num_epoch = 0
        for idx in range(10):
            #idx_train, idx_val, idx_test = rand_train_test_idx(labels)
            idx_train, idx_val, idx_test = random_disassortative_splits(labels, num_class)
            #idx_train, idx_val, idx_test = data_split(idx, args.dataset_name)
            #rank = OneVsRestClassifier(LinearRegression()).fit(features[idx_train], labels[idx_train]).predict(features)
            #print(rank)
            #adj = reconstruct(old_adj, rank, num_class)

            model = GCN(
                    num_layers=args.layers,
                    in_dim=features.shape[1],
                    num_hidden=args.hidden,
                    num_classes=labels.max().item() + 1,
                    dropout=args.dropout)

            if args.cuda:
                #adj = adj.cuda()
                idx_train = idx_train.cuda()
                idx_val = idx_val.cuda()
                idx_test = idx_test.cuda()
                model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            vlss_mn = np.inf
            vacc_mx = 0.0
            vacc_early_model = None
            vlss_early_model = None
            curr_step = 0
            best_test = 0
            best_training_loss = None
            for epoch in range(args.epochs):
                num_epoch = num_epoch+1
                t = time.time()
                model.train()
                optimizer.zero_grad()
                output = model(features,edge)
                #print(F.softmax(output,dim=1))
                output = F.log_softmax(output, dim=1)
                #print(output)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                acc_train = accuracy(output[idx_train], labels[idx_train])
                loss_train.backward()
                optimizer.step()

                if not args.fastmode:
                    # Evaluate validation set performance separately,
                    # deactivates dropout during validation run.
                    model.eval()
                    output = model(features,edge)
                    output = F.log_softmax(output, dim=1)

                val_loss = F.nll_loss(output[idx_val], labels[idx_val])
                val_acc = accuracy(output[idx_val], labels[idx_val])

                if val_acc >= vacc_mx or val_loss <= vlss_mn:
                    if val_acc >= vacc_mx and val_loss <= vlss_mn:
                        vacc_early_model = val_acc
                        vlss_early_model = val_loss
                        best_test = test(model, idx_train, idx_val, idx_test)
                        best_training_loss = loss_train
                    vacc_mx = np.max((val_acc, vacc_mx))
                    vlss_mn = np.min((val_loss, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break

            print("Optimization Finished! Best Test Result: %.4f, Training Loss: %.4f"%(best_test, best_training_loss))

            #model.load_state_dict(state_dict_early_model)
            # Testing
            result[idx] = best_test

            del model, optimizer
            if args.cuda: torch.cuda.empty_cache()
        five_epochtime = time.time() - t_total
        print("Total time elapsed: {:.4f}s, Total Epoch: {:.4f}".format(five_epochtime, num_epoch))
        print("learning rate %.4f, weight decay %.6f, dropout %.4f, Test Result: %.4f"%(args.lr, args.weight_decay, args.dropout, np.mean(result)))
        if np.mean(result)>best_result:
                best_result = np.mean(result)
                best_std = np.std(result)
                #best_dropout = args.dropout
                best_weight_decay = args.weight_decay
                best_lr = args.lr
                best_time = five_epochtime
                best_epoch = num_epoch

    print("Best learning rate %.4f, Best weight decay %.6f, dropout %.4f, Test Mean: %.4f, Test Std: %.4f, Time/Run: %.4f, Time/Epoch: %.4f"%(best_lr, best_weight_decay, 0, best_result, best_std, best_time/5, best_time/best_epoch))
    
    

train_supervised()



