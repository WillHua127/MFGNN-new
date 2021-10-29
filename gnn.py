from typing import List, Optional,Set
from torch_geometric.typing import Adj, Size
from torch_scatter.utils import broadcast


import torch
from torch import Tensor

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import degree
import torch.nn.functional as F


    
class MessagePassing(torch.nn.Module):

    def __init__(self, aggr: Optional[str] = "add",
                 flow: str = "source_to_target", node_dim: int = -2):

        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim


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


    def __collect__(self, edge_index, size, x):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        dim = 0
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
            data_sum = self.__lift__(data_sum, edge_index, i)
            data_prod = self.__lift__(data_prod, edge_index, i)

        return data_sum, data_prod

    def propagate(self, edge_index: Adj, x, size: Size = None, dim_size = None, norm=None):

        size = self.__check_input__(edge_index, size)

        if isinstance(edge_index, Tensor) or not self.fuse:
            x_sum,x_prod = self.__collect__(edge_index, size, x)
            x_sum = self.message(x_sum, norm)
            x_prod = self.message(x_prod, norm)
            x_sum, x_prod = self.aggregate((x_sum, x_prod), edge_index[1],ptr=None, dim_size=dim_size)
        return x_sum, x_prod

    def message_simple(self, x_j: Tensor) -> Tensor:
        return x_j


    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        return self.scatter_sum(inputs[0], index, dim=self.node_dim, dim_size=dim_size),self.scatter_product(inputs[1], index, dim=self.node_dim,dim_size=dim_size)

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
    def __init__(self, in_feat, out_feat):
        super(GCNConv, self).__init__(aggr='add')

        self.w1 = torch.nn.Linear(in_feat, out_feat)
        self.w2 = torch.nn.Linear(in_feat, out_feat)
        self.v = torch.nn.Linear(out_feat, out_feat)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.v.reset_parameters()
        
    def gcn_norm(self,edge_index, num_nodes=None, dtype=None, add_self_loops=True):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value=1, num_nodes=num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight


        row, col = edge_index[0], edge_index[1]
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row]  * deg_inv_sqrt[col]

    def forward(self, x, edge_index):
        x_sum_tar,x_prod_tar = self.w1(x[1]),self.w2(x[1])
        #x_prod = self.w2(x)
        row, col = edge_index

        deg = degree(col, x[0].size(0), dtype = x[0].dtype)+1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]


        sum_agg, prod_agg = self.propagate(edge_index, x=(x_sum_tar,x_prod_tar), norm=norm, dim_size=x_sum_tar.shape[0])

        return self.v(prod_agg)+(sum_agg)#+ F.relu6(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * F.relu(x_j)
    
    def update(self, aggr_out):
        return aggr_out
