import numpy as np
import torch

def get_metrics(conf_mat):
    acc = (conf_mat[0][0] + conf_mat[1][1]) / (conf_mat[0][0]+conf_mat[1][0]+conf_mat[0][1]+conf_mat[1][1])
    return conf_mat