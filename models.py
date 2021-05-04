import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gcns = nn.ModuleList()
        self.gcns.append(GraphConvolution(nfeat, nhid, output_layer = 0))
        self.gcns.append(GraphConvolution(nhid, nclass, output_layer=0))
        self.dropout = dropout
    

    def forward(self, x, adj):
        fea = F.relu(self.gcns[0](x, adj)) #
        fea = F.dropout(fea, self.dropout, training=self.training)
        fea = self.gcns[-1](fea, adj)
        return fea

    
