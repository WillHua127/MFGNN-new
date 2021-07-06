import torch.nn as nn
import torch.nn.functional as F
from layers import CPlayer, FClayer

    

    
    
class CPPooling(nn.Module):
    def __init__(self, in_fea, hidden, out_class,dropout):
        super(CPPooling, self).__init__()
        self.cp = CPlayer(in_fea, hidden, 8*hidden)
        self.fc = FClayer(hidden, out_class)
        self.dropout = dropout

    def forward(self, g, x, norm):
        fea = F.relu(self.cp(g, x, norm))
        fea = F.dropout(fea, self.dropout, training=self.training)
        out = self.fc(fea)
        return out

    
