from __future__ import print_function
from __future__ import division
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import copy
from torch import autograd
from torch.autograd import Variable
import pickle as pkl
print("PyTorch Version: ",torch.__version__)
import tqdm
from tqdm import tqdm
import pickle

from model_qa import *


# %%
data_file = "../train_task_1_2_sample.csv"
meta_data_file = "../Data/data/metadata/answer_metadata_task_1_2.csv"
log_file = "../logs/task_2_log.txt"
batch_size = 32
lr = 0.01
num_epochs = 100
num_feats = 256 #size of embedding of input layer
if_cuda = torch.cuda.is_available()
# if_cuda = False

df = pd.read_csv(data_file)



print(len(df))


torch.cuda.is_available()


# %%
model = CorrectnessPrediction(df,num_feats,lr,batch_size,if_cuda=if_cuda, num_epochs=num_epochs)


# %%
adj_lists = model.load_data()


# %%
a = (adj_lists.keys())
len(list(a))


# %%
cos = nn.CosineSimilarity(dim=0, eps=1e-6)


# %%
v1 = torch.randn((128,32))
v2 = torch.randn((128,32))


# %%
cos(v1,v2).shape


# %%
val_losses, model_weights = model.run_model()


# %%
unique_node_list = [1,2,3,4,5]


# %%
def foo(a,b):
    c = a+b
    return 1


# %%
x = torch.LongTensor(unique_node_list)


# %%
y = x.cuda()


# %%
y.is_cuda


# %%



