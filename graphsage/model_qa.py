import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class Question_Ans(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df
        self.questionid = self.df['QuestionId'].values
        self.userid = self.df['UserId'].values
        self.ans = self.df['IsCorrect'].values
        
        self.ans = 2*self.ans - 1
        
        self.length=len(self.ans)
        
        
        if(mode=='train'):
            start=int(0*self.length)
            end=int(0.8*self.length)
        elif(mode=='val'):
            start=int(0.8*self.length)
            end=int(1*self.length)
        else:
            start = 0
            end = int(self.length)
            
            
        self.questionid = self.questionid[start:end]
        self.userid = self.userid[start:end]
        self.ans = self.ans[start:end]
        
        self.length=len(self.ans)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        qid = self.questionid[idx]
        uid = self.userid[idx]
        ans = self.ans[idx]
        return qid,uid,ans


class SupervisedGraphSage(nn.Module):

    def __init__(self, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.MSELoss()
    
    def get_score(self,q_embeds,u_embeds):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        scores = cos(test1,test2)
        scores = (scores+1)/2
        return scores
    
    def forward(self, questions, users):
        q_embeds = self.enc(questions)
        u_embeds = self.enc(users)
        scores = self.get_score(q_embeds,u_embeds)
        return scores.t()

    def loss(self, question, users, ans):
        scores = self.forward(question, users)
#         return self.xent(scores, labels.squeeze())
        return self.xent(scores, ans)

class CorrectnessPrediction():
    def __init__(self,df,num_feats,lr,batch_size):
        self.df = df
        self.users = df['UserId'].unique()
        self.questions = df['QuestionId'].unique()
        self.num_nodes = len(self.users) + len(self.questions)
        self.num_feats = num_feats
        self.user_map, self.question_map = self.sequencify(self.users,self.questions)
        self.lr = lr
        self.batch_size = batch_size
    
    def sequencify(self,users,questions):
        user_map, question_map = {}, {}
        for i in range(len(users)):
            user_map[users[i]] = i
        for i in range(len(questions)):
            question_map[questions[i]] = i+len(users)
        return user_map, question_map
    
    def load_data(self):
        self.feat_data = np.zeros((self.num_nodes, self.num_feats))
#         labels = np.empty((num_nodes,1), dtype=np.int64)
#         node_map = {}
#         label_map = {}
#         with open("cora/cora.content") as fp:
#             for i,line in enumerate(fp):
#                 info = line.strip().split()
#                 feat_data[i,:] = map(float, info[1:-1])
#                 node_map[info[0]] = i
#                 if not info[-1] in label_map:
#                     label_map[info[-1]] = len(label_map)
#                 labels[i] = label_map[info[-1]]
        
        # can be made faster using lambdas
        self.adj_lists = defaultdict(set)
        for index, row in self.df.iterrows():
            uid,qid = self.user_map[row['UserId']], self.question_map[row['QuestionId']]
            self.adj_lists[uid].add(qid)
            self.adj_lists[qid].add(uid)
            if(index%1000==0):
                print(str(index)+' Completed')
        return 

    def run_model(self):
        np.random.seed(1)
        random.seed(1)
#         feat_data, labels, adj_lists = load_cora()
        features = nn.Embedding(self.num_nodes, self.num_feats)
        features.weight = nn.Parameter(torch.FloatTensor(self.feat_data), requires_grad=False)
       # features.cuda()

        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, self.num_feats, 128, self.adj_lists, agg1, gcn=True, cuda=False)
        agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, self.adj_lists, agg2,
                base_model=enc1, gcn=True, cuda=False)
        enc1.num_samples = 5
        enc2.num_samples = 5

        graphsage = SupervisedGraphSage(enc2)
    #    graphsage.cuda()
        
        train_dataset = Question_Ans(self.df,mode='train')
        val_dataset = Question_Ans(self.df,mode='val')
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=self.batch_size, shuffle=False)
#         rand_indices = np.random.permutation(num_nodes)
#         test = rand_indices[:1000]
#         val = rand_indices[1000:1500]
#         train = list(rand_indices[1500:])

        optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=self.lr)
        times = []
        
        phase = 'train'
        for questions,users,ans in tqdm(train_dataloader[phase]):
#             batch_nodes = train[:256]
#             random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(questions,users, ans)
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
#             print(batch, loss.data[0])

        val_output = graphsage.forward(val) 
        print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
        print("Average batch time:", np.mean(times))