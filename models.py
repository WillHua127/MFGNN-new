import torch.nn as nn
import torch.nn.functional as F
from layers import CPlayer, FClayer
from dgl.nn.pytorch import GraphConv
    



class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=F.relu))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=F.relu))
        # output layer
        self.layers.append(FClayer(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
    
class CPPooling(nn.Module):
    def __init__(self, in_fea, hidden, out_class, rank, dropout):
        super(CPPooling, self).__init__()
        self.cp = CPlayer(in_fea, hidden, rank)
        self.fc = FClayer(hidden, out_class)
        self.dropout = dropout

    def forward(self, g, x):
        fea = F.relu(self.cp(g, x))
        fea = F.dropout(fea, self.dropout, training=self.training)
        out = self.fc(fea)
        return out
    
    
class TwoCPPooling(nn.Module):
    def __init__(self, in_fea, hidden1, hidden2, out_class, rank1, rank2, dropout):
        super(TwoCPPooling, self).__init__()
        self.cp1 = CPlayer(in_fea, hidden1, rank1)
        self.cp2 = CPlayer(hidden1, hidden2, rank2)
        self.fc = FClayer(hidden2, out_class)
        self.dropout = dropout

    def forward(self, g, x):
        fea = F.relu(self.cp1(g, x))
        fea = F.dropout(fea, self.dropout, training=self.training)
        fea = F.relu(self.cp2(g, fea))
        fea = F.dropout(fea, self.dropout, training=self.training)
        out = self.fc(fea)
        return out

    
