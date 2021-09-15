import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
        
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 negative_slope=0.2):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
    
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
    
            
    
    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src = feat_dst = self.fc(feat).view(-1, self._num_heads, self._out_feats)#.view(-1, 1, 1)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = edge_softmax(graph, e)
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            
            rst = graph.dstdata['ft']
            return rst
        
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
        self.act = F.tanh
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
    
    def forward(self, g, x, norm=None):
         with g.local_scope():
            if isinstance(x, tuple):
                feat_src = x[0]
                feat_dst = x[1]
                feat_src = self.act(torch.mm(feat_src, self.W))
                feat_dst = self.act(torch.mm(feat_dst, self.W))
                g.srcdata['h'] = feat_src
                g.dstdata['h'] = feat_dst
                g.update_all(fn.copy_src('h', 'm'), self._elementwise_product)
                # divide in_degrees
                out = g.dstdata['neigh'] * g.dstdata['h']
                out = torch.mm(out, self.V.T)
                return out
            else:
                feat_src = feat_dst = torch.mm(x, self.W)
                g.srcdata['h'] = feat_src
                g.update_all(fn.copy_src('h', 'm'), self._elementwise_product)
                out = g.dstdata['neigh']

                #out = norm*torch.mm(out, self.V.T)
                out = torch.mm(out, self.V.T)

                return out
        
class GINConv(nn.Module):
    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False,
                 hidden = 32,
                 rank = 32,
                 out = 8):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        #self.V = Parameter(torch.FloatTensor(out, rank))
        #self.W = Parameter(torch.FloatTensor(hidden, rank))
        #self.batch_norm = nn.BatchNorm1d(hidden)
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
        
        #self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.W)

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')

            feat_src = feat_dst = feat
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                #rst = F.relu(self.batch_norm(self.apply_func(rst)))
                rst = self.apply_func(rst)
                #rst = torch.mm(rst, self.W)
                #cp_rst = torch.prod(rst, 0).unsqueeze(0)
                #readout = torch.mm(cp_rst, self.V.T)
            #print(readout.shape)
            return rst

