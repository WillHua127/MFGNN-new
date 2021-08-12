import torch
import torch.nn as nn
import torch.nn.functional as F

#from mlp import MLP

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class graph_cp_pooling(nn.Module):
    def __init__(self, in_fea, hidden, rank, dropout=0.6):
        super(graph_cp_pooling, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_fea, rank))
        #self.V = nn.Parameter(torch.FloatTensor(hidden, rank))
        #self.dropout = dropout
        self.reset_parameters()
       
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        #nn.init.xavier_uniform_(self.V)

    def forward(self, x):
        fea = F.tanh(torch.mm(x, self.W))
        #fea = F.dropout(fea, self.dropout, training = self.training)
        #fea = F.relu(fea)
        fea = torch.prod(fea,0).unsqueeze(0)
        #fea = torch.mm(fea, self.V.T)
        return fea

class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, rank_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        #self.batch_norms = torch.nn.ModuleList()

        #for layer in range(self.num_layers-1):
        #    if layer == 0:
        #        self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
        #    else:
        #        self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            #self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        self.cppools = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(rank_dim, output_dim))
                self.cppools.append(graph_cp_pooling(input_dim+1, hidden_dim, rank_dim))
            else:
                self.linears_prediction.append(nn.Linear(rank_dim, output_dim))
                self.cppools.append(graph_cp_pooling(input_dim +1, hidden_dim, rank_dim))
        #self.pred = nn.Linear(2*hidden_dim, output_dim)
   
   
    def __preprocess_neighbors_list_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        adj_list = []
        #t_n = 0
        for i, graph in enumerate(batch_graph):
            nnodes = len(graph.g)
            #t_n += nnodes
            edges = graph.edge_mat
            if not self.learn_eps:
                adj_list.append((torch.eye(nnodes)+torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), torch.Size([nnodes, nnodes]))).to(self.device))
            elif self.learn_eps:
                adj_list.append(torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), torch.Size([nnodes, nnodes])))

        return adj_list
   
   
    def next_layer_list(self, h, layer, padded_neighbor_list = None, Adj_list = None):
        ###pooling neighboring nodes and center nodes altogether
       
        pooled_feas = []    
        pooled_reps = []
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            for idx, adj in enumerate(Adj_list):
                #pooled_feas.append(F.relu(torch.spmm(adj, h[idx])))
                #pooled_feas.append(F.relu(self.mlps[layer](torch.spmm(adj, h[idx]))))
                pooled_feas.append(((torch.spmm(adj, h[idx]))))
                #pooled_feas.append(F.dropout(torch.spmm(adj, h[idx]), self.dropout, training = self.training))
                if self.neighbor_pooling_type == "average":
                    #If average pooling
                    degree = torch.spmm(adj, torch.ones((adj.shape[0], 1)).to(self.device))
                    pooled_feas[idx] = pooled_feas[idx]/degree
        return pooled_feas


    def forward(self, batch_graph):
        #X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        X_list = [graph.node_features.to(self.device) for graph in batch_graph]
        adj_list = self.__preprocess_neighbors_list_sumavepool(batch_graph)
        hidden_rep_list = [(torch.cat([self.cppools[0](torch.cat((X_list[idx], torch.ones([X_list[idx].shape[0],1]).to(self.device)), 1)) for idx in range(len(batch_graph))],0))]
        h_list = X_list

        for layer in range(self.num_layers-1):
            h_list = self.next_layer_list(h_list, layer, Adj_list = adj_list)
               
            hidden_rep_list.append(torch.cat([self.cppools[layer+1](torch.cat((h_list[idx], torch.ones([h_list[idx].shape[0],1]).to(self.device)), 1)) for idx in range(len(batch_graph))],0))

        score_over_layer = 0
   

        for layer, h in enumerate(hidden_rep_list):
            score_over_layer += F.dropout(self.linears_prediction[layer](h), self.final_dropout, training = self.training)

        return score_over_layer
