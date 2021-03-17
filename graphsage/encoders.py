import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()
        
        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
#         print('Num Samples : ', self.num_sample)
#         print('Nodes : ', nodes)
#         print('Aggregator : ', self.aggregator.features)
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        for i,node in enumerate(nodes) : 
            if len(to_neighs[i]) == 0:
#                 print("Node with no neighs : ", node)
                to_neighs[i] = set([int(node)])
        neigh_feats = self.aggregator.forward(nodes, to_neighs, 
                self.num_sample)
#         print('Number of samples : ',self.num_sample)
#         print('Printing Shape of Neighbour Features Aggregated: ',neigh_feats.shape)
        if torch.isnan(neigh_feats).any():
            print(neigh_feats,'Neight Feats is NaN')
            exit(1)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
#         print(neigh_feats)
#         print(combined)
        combined = F.relu(self.weight.mm(combined.t()))
        if torch.isnan(combined).any():
            print(combined,'Combined Feats is NaN')
            exit(1)
        return combined
