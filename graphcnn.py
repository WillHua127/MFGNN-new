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
    def __init__(self, in_fea, hidden, rank):
        super(graph_cp_pooling, self).__init__()
        self.W = nn.Linear(in_fea, rank)
        self.V = nn.Linear(rank, hidden)

    def forward(self, x):
        fea = self.W(x)
        fea = torch.prod(fea,0).unsqueeze(0)
        readout = self.V(fea)
        return readout

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
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        self.cppools = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))
                self.cppools.append(graph_cp_pooling(input_dim+1, hidden_dim, rank_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))
                self.cppools.append(graph_cp_pooling(hidden_dim+1, hidden_dim, rank_dim))
        self.pred = nn.Linear(hidden_dim, output_dim)


    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)
    
    
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


    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        adj_list = []
        #t_n = 0
        for i, graph in enumerate(batch_graph):
            nnodes = len(graph.g)
            #t_n += nnodes
            edges = graph.edge_mat
            start_idx.append(start_idx[i] + nnodes)
            edge_mat_list.append(edges + start_idx[i])
            if not self.learn_eps:
                adj_list.append(torch.eye(nnodes)+torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), torch.Size([nnodes, nnodes])))
            elif self.learn_eps:
                adj_list.append(torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), torch.Size([nnodes, nnodes])))
            #print(edge_mat_list)
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))
        #print(t_n, Adj_block.shape[0])

        return Adj_block.to(self.device), adj_list


    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
            
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h
    
    
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
                pooled_feas.append(F.relu(self.mlps[layer](torch.spmm(adj, h[idx]))))
                if self.neighbor_pooling_type == "average":
                    #If average pooling
                    degree = torch.spmm(adj, torch.ones((adj.shape[0], 1)).to(self.device))
                    pooled_feas[idx] = pooled_feas[idx]/degree
                #pooled_reps.append(F.relu(self.batch_norms[layer](self.mlps[layer](pooled_feas[idx]))))
                #pooled_reps.append(F.relu(self.mlps[layer](pooled_feas[idx])))
            #pooled = torch.spmm(Adj_block, h)
        
            
                
        #for fea in feas:
        #representation of neighboring and center nodes 
        #pooled_rep = self.mlps[layer](pooled)

        #h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        #h = F.relu(h)
        #return pooled_reps
        return pooled_feas


    def forward(self, batch_graph):
        #X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        X_list = [graph.node_features.to(self.device) for graph in batch_graph]
        #graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            #Adj_block, adj_list = self.__preprocess_neighbors_sumavepool(batch_graph)
            adj_list = self.__preprocess_neighbors_list_sumavepool(batch_graph)

        #list of hidden representation at each layer (including input)
        #hidden_rep_list = []
        #hidden_rep = [X_concat]
        #hidden_rep_list = [torch.cat(X_list, 0)]
        #features = torch.hstack([X_list[idx], torch.ones([X_list[idx].shape[0],1])])
        #hidden_rep_list = [torch.cat([self.cppools[0](X_list[idx]) for idx in range(len(batch_graph))],0)]
        hidden_rep_list = [torch.cat([self.cppools[0](torch.cat((X_list[idx], torch.ones([X_list[idx].shape[0],1]).to(self.device)), 1)) for idx in range(len(batch_graph))],0)]
        #hidden_rep_list = [torch.cat([self.cppools[0](torch.hstack([X_list[idx], torch.ones([X_list[idx].shape[0],1])])) for idx in range(len(batch_graph))],0)]
        #hidden_rep_list = [F.dropout(torch.cat([self.cppools[0](torch.hstack([X_list[idx], torch.ones([X_list[idx].shape[0],1])])) for idx in range(len(batch_graph))],0), self.final_dropout, training = self.training)]
        #h = X_concat
        h_list = X_list

        #for layer in range(self.num_layers-1):
        #    if self.neighbor_pooling_type == "max" and self.learn_eps:
        #        h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
        #    elif not self.neighbor_pooling_type == "max" and self.learn_eps:
        #        h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
        #    elif self.neighbor_pooling_type == "max" and not self.learn_eps:
        #        h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
        #    elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                #h = self.next_layer(h, layer, Adj_block = Adj_block)
        h_list = self.next_layer_list(h_list, 0, Adj_list = adj_list)
                
        #new_h_list = [torch.hstack([X_list[idx],h_list[idx]]) for idx in range(len(batch_graph))]
        #hidden_rep_list.append(F.dropout(torch.cat([self.cppools[1](torch.hstack([new_h_list[idx], torch.ones([h_list[idx].shape[0],1])])) for idx in range(len(batch_graph))],0), self.final_dropout, training = self.training))
            
            #hidden_rep_list.append(torch.cat([self.cppools[layer+1](h_list[idx]) for idx in range(len(batch_graph))],0))
        #hidden_rep_list.append(F.dropout(torch.cat([self.cppools[1](torch.hstack([h_list[idx], torch.ones([h_list[idx].shape[0],1])])) for idx in range(len(batch_graph))],0), self.final_dropout, training = self.training))
        #hidden_rep_list.append(torch.cat([self.cppools[1](torch.hstack([h_list[idx], torch.ones([h_list[idx].shape[0],1])])) for idx in range(len(batch_graph))],0))
        hidden_rep_list.append(torch.cat([self.cppools[1](torch.cat((h_list[idx], torch.ones([h_list[idx].shape[0],1]).to(self.device)), 1)) for idx in range(len(batch_graph))],0))

            #hidden_rep.append(h)

        score_over_layer = 0
    
        #perform pooling over all nodes in each graph in every layer
        #for layer, h in enumerate(hidden_rep):
        #final_rep = torch.cat(hidden_rep_list, 1)
        #final_rep = hidden_rep_list[0]
        #score_over_layer = F.dropout(self.pred(final_rep), self.final_dropout, training = self.training)
        #score_over_layer = self.pred(final_rep)

        for layer, h in enumerate(hidden_rep_list):
            #if layer == 0:
            #    pooled_h = h
            #else:
            #    pooled_h = torch.spmm(graph_pool, h)
            #pooled_h = self.cppools[layer](h)
            #print(pooled_h.shape)
            score_over_layer += F.dropout(self.linears_prediction[layer](h), self.final_dropout, training = self.training)
            #score_over_layer += self.linears_prediction[layer](h)
            #print(score_over_layer.shape)

        return score_over_layer
