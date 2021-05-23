import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, SAGEConv
import torch
#from torch_geometric.nn import SGConv



class SAGEBC(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden))
        self.layers.append(SAGEConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        #print(blocks)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            #h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    
    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y
    

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden))
        self.layers.append(SAGEConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    
class SGCNonh(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGCNonh, self).__init__()
        self.W_l = nn.Linear(in_channels,out_channels)

    def forward(self, x):
    
        return self.W_l(x)
    
class SGC(nn.Module):
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W_l = nn.Linear(nfeat,nclass)
        self.W_h = nn.Linear(nfeat,nclass)
        self.W_i = nn.Linear(nfeat,nclass)
        self.att_vec_mlp= nn.Linear(nclass, 1)
        self.att_vec_low = nn.Linear(nclass, 1)
        self.att_vec_high = nn.Linear(nclass, 1)
    
    def attention(self, output_low, output_high, output_mlp):
        low_norm = (torch.norm(output_low,dim=1).detach())[:,None] + 1e-16#.detach()
        high_norm = (torch.norm(output_high,dim=1).detach())[:,None]+ 1e-16 #.detach()
        mlp_norm = (torch.norm(output_mlp,dim=1).detach())[:,None] + 1e-16#.detach() torch.norm((support_mlp).detach(),dim=1) +
        T = 1/(low_norm + high_norm + mlp_norm)
        
        
        
        att_mlp = F.elu(self.att_vec_mlp(torch.cat([output_mlp.detach()/mlp_norm],1)),alpha = 5)
        att_low = F.elu(self.att_vec_low(torch.cat([output_low.detach()/low_norm],1)),alpha = 5)
        att_high = F.elu(self.att_vec_high(torch.cat([output_high.detach()/high_norm],1)),alpha = 5)
        
        att = torch.softmax((torch.cat([att_low  ,att_high , att_mlp ],1)/T),1)
 
        #return torch.sigmoid(att_low),torch.sigmoid(att_high),torch.sigmoid(att_mlp)
        return att[:,0][:,None],att[:,1][:,None],att[:,2][:,None]

    def forward(self, f, f_l, f_h):
        f = F.relu(self.W_i(f))
        f_l = F.relu(self.W_l(f_l))
        f_h = F.relu(self.W_h(f_h))
        
        att_low, att_high, att_mlp = self.attention(f_l, f_h, f)
        
        return 3*(att_low*f_l+att_high*f_h+att_mlp*f)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gcns = nn.ModuleList()
        self.gcns.append(GraphConvolution(nfeat, nhid, output_layer = 0))
        self.gcns.append(GraphConvolution(nhid, nclass, output_layer=0))
        self.dropout = dropout
    

    def forward(self, x, adj, adj_high):
        fea = F.relu(self.gcns[0](x, adj, adj_high)) #
        #fea = (self.gcns[0](x, adj, adj_high))
        fea = F.dropout(fea, self.dropout, training=self.training)
        fea = self.gcns[-1](fea, adj, adj_high)
        return fea

    
