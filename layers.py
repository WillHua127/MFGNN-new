import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

import dgl.function as fn
        
        
class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None):
        super(GraphConv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_feats, out_feats))

        self.reset_parameters()

        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)


    def forward(self, graph, fea):
        with graph.local_scope():
           feat_src = torch.mm(fea, self.weight)
           graph.srcdata['h'] = feat_src
           graph.update_all(fn.copy_src('h', 'm'), fn.sum(msg='m', out='h'))
           rst = graph.dstdata['h']
           rst = self.activation(rst)

           return rst

class FClayer(Module):
    def __init__(self, in_fea, out_class):
        super(FClayer, self).__init__()
        self.FC = Parameter(torch.FloatTensor(in_fea, out_class))
        self.reset_parameters()
        

    def reset_parameters(self):
        #gain = nn.init.calculate_gain('relu')
        stdv = 1. / math.sqrt(self.FC.size(1))
        self.FC.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        output = torch.mm(input, self.FC)
        return output
    
    
class CPlayer(Module):
    def __init__(self, in_fea, hidden, rank):
        super(CPlayer, self).__init__()
        self.W = Parameter(torch.FloatTensor(in_fea, rank))
        self.V = Parameter(torch.FloatTensor(hidden, rank))
        self.reset_parameters()
        
    def reset_parameters(self):
        #gain = nn.init.calculate_gain('relu')
        #stdv = 1. / math.sqrt(self.V.size(0))
        #self.W.data.uniform_(-stdv, stdv)
        #self.V.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)
                
    def cpmm(self, x, edge):
        hat_x = x.clone()
        for i in range(x.shape[0]):
            hat_x[i] = (torch.prod(x[edge[i]],dim=0))
        return hat_x
    
    def _elementwise_product(self, nodes):
        return {'neigh':torch.prod(nodes.mailbox['m'],dim=1)}
    
    def forward(self, g, x):
         with g.local_scope():
            feat_src = feat_dst = x = torch.mm(x, self.W)

            g.srcdata['h'] = feat_src
            g.update_all(fn.copy_src('h', 'm'), self._elementwise_product)
            #trans_x = g.ndata['norm']*g.dstdata['neigh']
            trans_x = g.dstdata['neigh']
            
            out = torch.mm(trans_x, self.V.T)
            #out = out

            return out
                
        #x = torch.mm(input, self.W)
        #x = self.cpmm(x, edge_dict)
        #output = F.relu(torch.mm(x,self.V.T))
        #output = torch.mm(output, self.fc)
        
        #return output
