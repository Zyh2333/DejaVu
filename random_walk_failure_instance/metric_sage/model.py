import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from random_walk_failure_instance.metric_sage.encoders import Encoder
from random_walk_failure_instance.metric_sage.aggregators import MeanAggregator
import pandas as pd
from random_walk_failure_instance.metric_sage.index_map import IndexMap
from random_walk_failure_instance.metric_sage.index_map import index_map

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes, metric, is_node_train_index = True):
        embeds = self.enc(nodes, metric, is_node_train_index)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, metric, labels, is_node_train_index = True):
        scores = self.forward(nodes, metric, is_node_train_index)
        return self.xent(scores, labels.squeeze())

def load_RCA(node_num, feat_num, df, time_data, time_list):
    num_nodes = node_num
    num_feats = feat_num
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    count = 0
    index_map_list = []
    for i,row in df.iterrows():
        for j, time in enumerate(time_list):
            if time.in_time(int(time_data[i:i+1]['timestamp']), i):
                labels[count] = j
                feat_data[count,:] = df[i:i+1]
                index_map_list.append(IndexMap(count, i))
                count += 1

    adj_lists = defaultdict(set)
    label_map = defaultdict(set)
    for i, label in enumerate(labels):
        label_map[label[0]].add(i)
    for s in label_map:
        for i in range(len(label_map[s])):
            for j in range(len(label_map[s])):
                adj_lists[list(label_map[s])[i]].add(list(label_map[s])[j])
    
    return feat_data, labels, adj_lists, index_map_list

def load_RCA_with_label(node_num, feat_num, df, time_data, time_list):
    num_nodes = node_num
    num_feats = feat_num
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    count = 0
    index_map_list = []
    for i,row in df.iterrows():
        for j, time in enumerate(time_list):
            if time.in_time(int(time_data[i:i+1]['timestamp']), i):
                labels[count] = time.label
                feat_data[count, :] = df[i:i+1]
                index_map_list.append(IndexMap(count, i))
                count += 1

    adj_lists = defaultdict(set)
    label_map = defaultdict(set)
    for i, label in enumerate(labels):
        label_map[label[0]].add(i)
    for s in label_map:
        for i in range(len(label_map[s])):
            for j in range(len(label_map[s])):
                adj_lists[list(label_map[s])[i]].add(list(label_map[s])[j])

    return feat_data, labels, adj_lists, index_map_list

def run_RCA(node_num, feat_num, df, time_data, time_list, metric, folder, class_num, label_file, train_=False, cuda=False):
    np.random.seed(1)
    random.seed(1)
    num_nodes = node_num
    feat_data, labels, adj_lists, index_map_list = load_RCA_with_label(node_num, feat_num, df, time_data, time_list)

    features = nn.Embedding(node_num, feat_num)

    # todo
    agg1 = MeanAggregator('agg1', features, metric, index_map_list, feat_num, 64, cuda=cuda)
    enc1 = Encoder('enc1', features, 64, 32, adj_lists, agg1, metric, index_map_list, gcn=True, cuda=cuda)
    agg2 = MeanAggregator('agg2', lambda nodes, metric, is_train_index: enc1(nodes, metric, is_train_index).t(), metric, index_map_list, feat_num, 32, cuda=cuda)
    enc2 = Encoder('enc2', lambda nodes, metric, is_train_index: enc1(nodes, metric, is_train_index).t(), enc1.embed_dim, class_num, adj_lists, agg2, metric, index_map_list,
                   base_model=enc1, gcn=True, cuda=cuda)

    # train parameters
    num_sample = 10
    batch_size = 10000
    epochs = 400
    learning_rate = 0.05
    enc1.num_sample = num_sample
    enc2.num_sample = num_sample

    graphsage = SupervisedGraphSage(class_num, enc2)
    if cuda:
        graphsage = graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    division = int(len(rand_indices) / 10)
    epochs = int(division / 4)
    test = rand_indices[:division]
    val = rand_indices[division:2 * division]
    train = list(rand_indices[2 * division:])
    # model diy name
    # suffix_diy = "data_modify"
    # suffix = "model_parameters_DejaVu-D-1684318342.544748_91_20_10000_0.7.pkl"
    # suffix = "model_parameters_DejaVu-A2-1684314724.261361_18_20_2000_0.05.pkl"
    # suffix = "model_parameters_DejaVu-A1-1683993521.079898_22_10_2000_0.05.pkl"
    suffix = "model_parameters_DejaVu-A2-1684041806.569868_18_10_2000_0.05.pkl"
    # suffix = "model_parameters_DejaVu-D-1684124503.9004421_91_10_10000_0.7.pkl"
    trained_model = graphsage
    trained_model.load_state_dict(torch.load("/Users/zhuyuhan/Documents/391-WHU/experiment/researchProject/MicroIRC/model/" + suffix))
    val_output = trained_model.forward(val, metric.loc[index_map(val, index_map_list)], True)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    return trained_model


if __name__ == "__main__":
    pass
