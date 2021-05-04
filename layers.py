import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
        
    

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
        #self.weight_high = Parameter(torch.FloatTensor(in_features, out_features))
        #self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features))
        #self.weight = Parameter(torch.FloatTensor(out_features, out_features))
        #self.att_trans = Parameter(torch.FloatTensor(in_features, out_features))
        #self.att_trans_high = Parameter(torch.FloatTensor(out_features, 1))
        #self.att_trans_mlp = Parameter(torch.FloatTensor(out_features, 1))
        #self.att_vec = Parameter(torch.FloatTensor(out_features, 1))
        #self.I = (torch.tensor(0.5))
        #self.P = (torch.tensor(0.5))
        
        
        if bias:
            self.bias, self.bias_low, self.bias_high, self.bias_mlp = Parameter(torch.FloatTensor(out_features)), Parameter(torch.FloatTensor(out_features)), Parameter(torch.FloatTensor(out_features)), Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_low.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        #self.weight_high.data.uniform_(-stdv, stdv)
        #self.weight_mlp.data.uniform_(-stdv, stdv)
        #self.weight.data.uniform_(-stdv, stdv)
        
        #self.att_trans.data.uniform_(-stdv, stdv)
        #self.att_trans_high.data.uniform_(-stdv, stdv)
        #self.att_trans_mlp.data.uniform_(-stdv, stdv)
        #self.att_vec.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.bias_low.data.uniform_(-stdv, stdv)
            self.bias_high.data.uniform_(-stdv, stdv)
            self.bias_mlp.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #adj = adj.to_dense()
        #nnodes = adj.size(1)
        adj_low = adj #self.I * torch.eye(nnodes) + self.P * adj
        #adj_high = torch.eye(nnodes) - adj_low
        support_low = torch.mm(input, self.weight_low)
        #support_low = F.relu(support_low) #+self.bias_low
        output_low = torch.spmm(adj_low, support_low)
        
        
        #support_high = torch.mm(input, self.weight_high)
        #support_high = F.relu(support_high) #+self.bias_high
        #output_high = torch.spmm(adj_high, support_high)
        
        
        #output_mlp = F.relu(torch.mm(input, self.weight_mlp)) # +self.bias_mlp
        
        #Compute Attention
        #combined_output = F.sigmoid(torch.mm(input, self.att_trans)) #output_mlp + output_low + output_high  #
        # combined_trans = torch.mm(combined_output,self.att_trans)
        # low_trans = torch.mm(output_low,self.att_trans)
        # high_trans = torch.mm(output_high,self.att_trans)
        # mlp_trans = torch.mm(output_mlp,self.att_trans)
        
        #att_low_output_trans = F.leaky_relu(torch.mm(torch.cat([combined_output,output_low],1), self.att_trans_low))
        #att_low_output_trans = F.leaky_relu(torch.mm(torch.cat([combined_output,output_low],1), self.att_trans_low))
        #att_high_output_trans = F.leaky_relu(torch.mm(output_high, self.att_trans_high))
        #att_mlp_trans = F.leaky_relu(torch.mm(output_mlp, self.att_trans_mlp))
        
        
        #att_low = F.elu(torch.mm(torch.cat([output_low],1), self.att_vec))
        #att_high = F.elu(torch.mm(torch.cat([output_high],1), self.att_vec))
        #att_mlp = F.elu(torch.mm(torch.cat([output_mlp],1), self.att_vec))
        
        #att = torch.softmax(torch.cat([att_low,att_high,att_mlp],1),1) #
        
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
