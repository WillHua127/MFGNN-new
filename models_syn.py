import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch

from torch import Tensor
from torch_sparse import SparseTensor
#from torch_scatter import gather_csr, scatter, segment_csr
from typing import List, Optional, Set, Callable
from torch_geometric.typing import Adj, Size
from torch_scatter.utils import broadcast

import torch_geometric

from torch_geometric.nn.conv.utils.inspector import Inspector
from torch_geometric.utils.num_nodes import maybe_num_nodes


    
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
        #print(index, src.shape, dim)
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
    
    
class TGNNConv(MessagePassing):
    def __init__(self, emb_dim, hidden_dim, rank_dim):
        super(TGNNConv, self).__init__(aggr='add')

        self.w1 = torch.nn.Linear(emb_dim, hidden_dim)
        self.w2 = torch.nn.Linear(emb_dim+1, hidden_dim)
        self.v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.att1= torch.nn.Linear(hidden_dim, 1, bias=False)
        self.att2 = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.att_vec = torch.nn.Linear(2, 2, bias=False)
        self.root_emb = torch.nn.Embedding(1, hidden_dim)
        #self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        #self.bias = torch.nn.Parameter(torch.Tensor(hidden_dim))
        self.act = torch.nn.Hardtanh()
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
        #x_sum, x_prod = self.w1(x),self.w2(torch.cat((x, torch.ones([x.shape[0],1]).to('cuda:0')),1))#self.w2(x)
        x_sum, x_prod = self.w1(x),torch.tanh(self.w2(torch.cat((x, torch.ones([x.shape[0],1])),1)))#self.w2(x)

        row, col = edge_index
        sum_agg, prod_agg = self.propagate(edge_index, x=(x_sum,x_prod))

        return sum_agg#rst
    

    def update(self, aggr_out):
        return aggr_out
    
class TGNN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 dropout):
        super(TGNN, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.dropout = dropout
        feat_drop = dropout
        # input projection (no TGNNConv)
        self.gat_layers.append(TGNNConv(
            in_dim, num_hidden,num_hidden))

        for l in range(1, num_layers):
            self.gat_layers.append(TGNNConv(
                num_hidden, num_hidden, num_hidden))
        # output projection
        #self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes))
        self.gat_layers.append(TGNNConv(
            num_hidden, num_classes, num_hidden))

    def forward(self, x, edge_index):
        h = x
        for l in range(self.num_layers):
            h = F.relu(self.gat_layers[l](h, edge_index))
        # output projection
        logits = self.gat_layers[-1](h, edge_index)
        return logits
