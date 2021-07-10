import torch.nn as nn
import torch.nn.functional as F
from layers import CPlayer, FClayer

    

    
    
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
    def __init__(self, in_fea, hidden, out_class, rank1, rank2, dropout):
        super(TwoCPPooling, self).__init__()
        self.cp1 = CPlayer(in_fea, hidden, rank1)
        self.cp2 = CPlayer(hidden, out_class, rank2)
        self.dropout = dropout

    def forward(self, g, x):
        fea = F.relu(self.cp1(g, x))
        fea = F.dropout(fea, self.dropout, training=self.training)
        out = self.cp2(g, fea)
        return out

    
