import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
        
import dgl.function as fn
import dgl.utils as utl
        
    
class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = utl.expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.feat_drop = nn.Dropout(feat_drop)
        self.w_self = nn.Linear(self._in_src_feats, out_feats)
        self.w_neigh = nn.Linear(self._in_src_feats, out_feats)
        self.w_identity = nn.Linear(self._in_src_feats, out_feats)
        
        
        self.att_vec_low = nn.Linear(out_feats, 1)
        self.att_vec_high = nn.Linear(out_feats, 1)
        #self.att_vec_neigh_low = nn.Linear(out_feats, 1, bias=bias)
        #self.att_vec_neigh_high = nn.Linear(out_feats, 1, bias=bias)
        self.att_vec_identity= nn.Linear(out_feats, 1)
                                          
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.w_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_neigh.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_identity.weight, gain=gain)
        
        nn.init.xavier_uniform_(self.att_vec_low.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_vec_high.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_vec_identity.weight, gain=gain)

    def _mean_reducer(self, nodes):
        return {'neigh':torch.mean(nodes.mailbox['m'],dim=1)}
    
    def _mean_high_reducer(self, nodes):
        return {'neigh_high':-torch.mean(nodes.mailbox['m_high'],dim=1)}
    
    def attention(self, output_low, output_high, output_mlp):
        low_norm = (torch.norm(output_low,dim=1).detach())[:,None] + 1e-16#.detach()
        high_norm = (torch.norm(output_high,dim=1).detach())[:,None]+ 1e-16 #.detach()
        mlp_norm = (torch.norm(output_mlp,dim=1).detach())[:,None] + 1e-16#.detach() torch.norm((support_mlp).detach(),dim=1) +
        # T_norm = low_norm + high_norm + mlp_norm + 1e-16
        
        
        att_mlp = F.elu(self.att_vec_identity(torch.cat([output_mlp.detach()/mlp_norm],1)),alpha = 5)
        att_low = F.elu(self.att_vec_low(torch.cat([output_low.detach()/low_norm],1)),alpha = 5)
        att_high = F.elu(self.att_vec_high(torch.cat([output_high.detach()/high_norm],1)),alpha = 5)
 
        return torch.sigmoid(att_low),torch.sigmoid(att_high),torch.sigmoid(att_mlp)

 
    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            #if graph.is_block:
            feat_dst = feat_src[:graph.number_of_dst_nodes()]

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0: graph.dstdata['neigh'] = torch.zeros(feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            graph.srcdata['h'] = feat_src# * graph.ndata['norm']
            graph.update_all(fn.copy_src('h', 'm'), self._mean_reducer)
            #graph.update_all(fn.copy_src('h', 'm_high'), self._mean_high_reducer)
            h_neigh = graph.dstdata['neigh']
            #h_neigh_high = graph.dstdata['neigh_high']
            
            from_self = self.w_self(h_self)
            from_neigh = self.w_neigh(h_neigh)
            low = F.relu(self.w_self(h_self)+self.w_neigh(h_neigh))
            high = F.relu(self.w_self(h_self)-self.w_neigh(h_neigh))
            identity = F.relu(self.w_identity(feat_dst))
            att_low, att_high, att_mlp = self.attention(low, high, identity)

            rst = 3*(att_low*low + att_high*high + att_mlp*identity)

            return rst

        
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, output_layer = 0):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.output_layer = output_layer
        self.weight_low = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_high = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features))
        self.att_vec_mlp= nn.Linear(out_features, 1, bias=bias)
        self.att_vec_low = nn.Linear(out_features, 1, bias=bias)
        self.att_vec_high = nn.Linear(out_features, 1, bias=bias)
        
        
        if bias:
            self.bias, self.bias_low, self.bias_high, self.bias_mlp = Parameter(torch.FloatTensor(out_features)), Parameter(torch.FloatTensor(out_features)), Parameter(torch.FloatTensor(out_features)), Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        stdv = 1. / math.sqrt(self.weight_low.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.att_vec_mlp.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_vec_low.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_vec_high.weight, gain=gain)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.bias_low.data.uniform_(-stdv, stdv)
            self.bias_high.data.uniform_(-stdv, stdv)
            self.bias_mlp.data.uniform_(-stdv, stdv)
                
    def attention(self, output_low, output_high, output_mlp):
        low_norm = (torch.norm(output_low,dim=1).detach())[:,None] + 1e-16#.detach()
        high_norm = (torch.norm(output_high,dim=1).detach())[:,None]+ 1e-16 #.detach()
        mlp_norm = (torch.norm(output_mlp,dim=1).detach())[:,None] + 1e-16#.detach() torch.norm((support_mlp).detach(),dim=1) +
        # T_norm = low_norm + high_norm + mlp_norm + 1e-16
        
        
        att_mlp = F.elu(self.att_vec_mlp(torch.cat([output_mlp.detach()/mlp_norm],1)),alpha = 5)
        att_low = F.elu(self.att_vec_low(torch.cat([output_low.detach()/low_norm],1)),alpha = 5)
        att_high = F.elu(self.att_vec_high(torch.cat([output_high.detach()/high_norm],1)),alpha = 5)
 
        return torch.sigmoid(att_low),torch.sigmoid(att_high),torch.sigmoid(att_mlp)

    def forward(self, input, adj_low, adj_high):
        #adj = adj.to_dense()
        #nnodes = adj_low.size(1)
        #adj_low = adj #self.I * torch.eye(nnodes) + self.P * adj
        #adj_high = torch.eye(nnodes) - adj_low
        support_low = torch.mm(input, self.weight_low)
        #support_high = support_low - torch.spmm(adj_low, support_low)
        support_high = torch.mm(input, self.weight_high)
        support_low = F.relu(support_low) #+self.bias_low
        support_high = F.relu(support_high)
        output_low = torch.spmm(adj_low, support_low)
        
        
        #support_high = torch.mm(input, self.weight_high)
        #support_high = F.relu(support_high) #+self.bias_high
        output_high = torch.spmm(adj_high, support_high)
        
        
        output_mlp = F.relu(torch.mm(input, self.weight_mlp)) # +self.bias_mlp
        
        att_low, att_high, att_mlp = self.attention(output_low, output_high, output_mlp)
        
        output_low = 3*(att_low*output_low + att_high*output_high + att_mlp*output_mlp)
        
        #output = 3*(att[:,0][:,None]*output_low + att[:,1][:,None]*output_high+ att[:,2][:,None]*output_mlp )# output_mlp   # output_low + output_high + # # + self.c*output+self.c_high*output_high/(self.c+self.c_high)
        # if self.output_layer:
        #     output = torch.mm(output, self.weight)
        if self.bias is not None:
            return output_low #+ self.bias
        else:
            return output_low

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
