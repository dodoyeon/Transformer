import pdb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
# from Dataset import tr_cent_data, te_cent_data, tr_centroid#, te_centroid
# from metrics.LayerWiseMetrics import cdist2
# from kmeans_pytorch import kmeans
from sklearn import metrics
from sklearn.cluster import KMeans
import gzip
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################### K-means clustering Memory ########################
# reference : https://github.com/subhadarship/kmeans_pytorch
def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=True,
        iter_limit=0,
        #device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    #print(f'running k-means on {device}..')
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError
    # convert to float
    # X = X.float()
    # transfer to device
    #X = X.to(device)
    # initialize
    if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters)
    else:
        print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state#.to(device)
    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_distance_function(X, initial_state)
        # pdb.set_trace()
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to('cuda')
            selected = torch.index_select(X, 0, selected)
            # print(selected.size())
            if selected.size()[0] == 0:
                continue
            else:
                initial_state[index] = selected.mean(dim=0)
        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))
        # print(initial_state)
        # print(initial_state_pre)
        # increment iteration
        iteration = iteration + 1
        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break
    return choice_cluster.cpu(), initial_state.cpu()

def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError
    # convert to float
    # X = X.float()
    # transfer to device
    X = X.to(device)
    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)
    return choice_cluster.cpu()

def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)
    # N*1*M
    # A = data1.unsqueeze(dim=1)
    # 1*N*M
    # B = data2.unsqueeze(dim=0)
    # dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    # dis = dis.sum(dim=-1).squeeze()
    dis = torch.cdist(data1, data2)
    return dis

def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)
    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)
    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)
    cosine = A_normalized * B_normalized
    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

class KMEAN_Memory(nn.Module):
    def __init__(self, num_centroid, hidden_size, max_seq_length):
        super().__init__()
        self.num_centroid = num_centroid
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.centroid = nn.Embedding.from_pretrained(torch.normal(0.07, 0.39, size=(self.num_centroid,
                                          self.max_seq_length *
                                          self.hidden_size)))
    def forward(self, input):
        if self.centroid.weight.requires_grad == True:
            #pdb.set_trace()
            new_input = input.view(-1, self.max_seq_length*self.hidden_size)
            centroid = self.centroid.weight
            cluster_ids_input, new_centers = kmeans(new_input, num_clusters=self.num_centroid, cluster_centers=centroid, distance='euclidean')
            self.centroid = self.centroid.from_pretrained(new_centers)
            self.centroid.weight.requires_grad = True
            return self.centroid.weight, cluster_ids_input # cent(8,98304) ids(32)
        else:
            new_input = input.view(-1, self.max_seq_length * self.hidden_size)
            centroid = self.centroid.weight
            prediction = kmeans_predict(new_input, centroid)
            return prediction


def cent_table(centroid_updated, centroid_trg):
    dist = torch.cdist(centroid_updated, centroid_trg)  # (8,8)
    table = []
    dist_sort = dist.view(1, -1)  # (1,64)
    sort = torch.argsort(dist_sort, dim=1).squeeze()
    i = 0
    idx = sort[i].item()
    row_idx = (idx // dist.size(0))  # 타겟에 대한 업데이트 센트로이드의 인덱스
    col_idx = (idx % dist.size(0))  # 업데이트에 대한 타겟의 인덱스
    table.append([row_idx, col_idx])
    i += 1
    while len(table) < dist.size(0):
        if i < (dist.size(0)) ** 2:
            idx = sort[i].item()
            row_idx = (idx // dist.size(0))  # 타겟에 대한 업데이트 센트로이드의 인덱스
            col_idx = (idx % dist.size(0))  # 업데이트에 대한 타겟의 인덱스
            ten_table = torch.tensor(table)[:, 0]
            ten_table1 = torch.tensor(table)[:, 1]

            if (row_idx in ten_table.tolist()) or (col_idx in ten_table1.tolist()):
                i += 1
                if len(table) == 8:
                    break
                else:
                    continue
            else:
                table.append([row_idx, col_idx])
        else:
            ten_table2 = torch.tensor(table)
            check = torch.arange(dist.size(0))
            for m in range(dist.size(0)):
                if check[m] not in ten_table2[:, 0]:
                    s0 = check[m]
                if check[m] not in ten_table2[:, 1]:
                    s1 = check[m]
            table.append([s0, s1])
        i += 1
    table = torch.tensor(table)
    return table


def mapping(idx, table):
    dict_table = {}
    mapped_idx = []
    for i in range(table.size(0)):
        dict_table[table[i, 0].item()] = table[i, 1].item()

    for j in range(idx.size(0)):
        map_idx = dict_table[idx[j].item()]
        mapped_idx.append(map_idx)
    mapped_idx = torch.tensor(mapped_idx)
    return mapped_idx

# # DATASET: Random maden data
# if __name__ == "__main__":
#     num_centroid = 8
#     max_seq_length = 128
#     hidden_size = 768
#     # device = 'cpu'
#     # std = np.sqrt(0.2)
#     tr_example_size = 3200
#     te_example_size = 640
#     batch_size = 32
#     epochs = 6
#
#     # with gzip.open('tr_data.pickle','rb') as f:
#     #     tr_cent_data = pickle.load(f)
#
#     model = KMEAN_Memory(num_centroid, hidden_size, max_seq_length).to(device)
#     tr_data = tr_cent_data[:, :-1]  # (320,98304)
#     tr_target = tr_cent_data[:, -1]  # (320)
#
#     model.train()
#     model.centroid.weight.requires_grad = True
#     correct = 0
#
#     print("clustering start")
#     for epoch in range(epochs):  # unsupervised learning 이니까 아마도 에폭이 의미가 없는것 같다
#         # For silhouette score
#         tr_out = torch.ones(1).unsqueeze(0).to(device)
#
#         for b in range(0, len(tr_data), batch_size):
#             batch = tr_data[b:b + batch_size, :].to(device)  # tensor(32,98304)
#             centroid, result = model(batch) # result(32)
#
#             tr_out = torch.cat((tr_out, result.unsqueeze(1).float().to(device)), 0) # result(32,1)
#             # correct += (real_idx == batch_trg).sum().item()
#
#         # Get Silhouette Score
#         tr_out_arr = tr_out[1:, :].squeeze().cpu().detach().numpy()
#         tr_data_arr = tr_data.numpy()
#         s_score = metrics.silhouette_score(tr_data_arr, tr_out_arr, metric='euclidean')
#         print("train Silhouette score: %.3f" % (s_score))
#
#     # mapping predict and real
#     table = cent_table(centroid.cpu(), tr_centroid)  # table tensor([2,3,4,4,1,1,1,4])
#     real_idx = mapping(tr_out[1:, :], table).squeeze()
#     correct += (real_idx == tr_target).sum().item()
#     print("Accuracy of the Cluster in training: %d epochs| %.3f %%" % (epoch, 100 * correct / tr_example_size))
#     # correct = 0
#
#     # with gzip.open('te_data.pickle','rb') as f:
#     #     te_cent_data = pickle.load(f)
#
#     model.eval()
#     te_data = te_cent_data[:, :-1]  # (320,98304)
#     te_target = te_cent_data[:, -1]  # (320)
#     correct = 0
#     te_out = torch.ones(1).unsqueeze(0).to(device)
#     with torch.no_grad():
#         model.centroid.weight.requires_grad = False
#         for b in range(0, len(te_data), batch_size):
#             batch = te_data[b:b + batch_size, :].to(device)
#             batch_trg = te_target[b:b + batch_size].to(device)
#             pred = model(batch)  # False
#
#             te_out = torch.cat((te_out, pred.unsqueeze(1).float().to(device)), 0)
#
#         # mapping predict and real
#         table = cent_table(centroid.cpu(), tr_centroid)  # table tensor([2,3,4,4,1,1,1,4])
#         real_idx = mapping(te_out[1:, :], table).squeeze()
#         correct += (real_idx == te_target).sum().item()
#         print("Accuracy of the Cluster in testing: %.3f %%" % (100 * correct / te_example_size))
#
#         # Get Silhouette Score
#         te_out_arr = te_out[1:, :].squeeze().cpu().detach().numpy()
#         te_data_arr = te_data.numpy()
#         s_score = metrics.silhouette_score(te_data_arr, te_out_arr, metric='euclidean')
#         print("test Silhouette score: %.3f" % (s_score))

# DATASET: BERT feature data
import matplotlib.pyplot as plt
from Dataset import Bert_feature_dataset
from torch.utils.data import Dataset, DataLoader
import os
dir = os.path.abspath("BERT_feature/cola_tensors")
dataset = Bert_feature_dataset(dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

num_centroid = 8
max_seq_len = 128
hidden_size = 768

tr_ex_size = 0
te_ex_size = 0
batch_size = 32
# epochs = 6
k_max = 20
sil = []
k_num = []
# model.train()

# target이 없으므로 correct를 셀수 없다
# for k in range(2, k_max):
#     model = KMEAN_Memory(k, hidden_size, max_seq_len).to(device)
#     model.centroid.weight.requires_grad = True
#     # For silhouette score
#     tr_out = torch.ones(1).unsqueeze(0).to(device)
#     tr_data = torch.ones(1,max_seq_len * hidden_size).to(device)
#
#     k_num.append(k.int())
#     for i, batch in enumerate(dataloader): # batch(1,32,128,768)
#         centroid, result = model(batch.squeeze(0))
#         tr_out = torch.cat((tr_out, result.unsqueeze(1).float().to(device)), 0)
#         tr_data = torch.cat((tr_data, batch.squeeze(0).view(-1, max_seq_len * hidden_size)),0)
#     # Get Silhouette Score
#     tr_out_arr = tr_out[1:, :].squeeze().cpu().detach().numpy()
#     tr_data_arr = tr_data[1:, :].cpu().numpy()
#     s_score = metrics.silhouette_score(tr_data_arr, tr_out_arr, metric='euclidean')
#     sil.append(s_score)
#     print("k: %d Silhouette score: %.3f" % (k, s_score))
#
# plt.plot(k_num, sil)
# plt.xlabel("k (the number of centroids)")
# plt.ylabel("Silhouette score")
# plt.savefig("./image/fig2.png")

#### test sklearn clustering ####
sil_sk = []
s = 0
k_max_sk = 20

total_batch = torch.ones(1, max_seq_len * hidden_size) # .to(device)
total_out = torch.ones(1)# .unsqueeze(0).to(device)
for k_sk in range(2, k_max_sk):
    for i, batch in enumerate(dataloader):
        changed_batch = batch.view(-1, max_seq_len * hidden_size).squeeze(0).cpu()
        total_batch = torch.cat((total_batch, changed_batch),0)
        kmeans = KMeans(k_sk).fit(changed_batch)
        # kmean = MiniBatchKMeans(n_clusters = k_sk
        #                     random_state=0,
        #                     batch_size = 32).fit()
        labels = kmeans.labels_
        total_out = torch.cat((total_out, torch.from_numpy(labels).float()),0)
        # s += score
    # sil_score = s/len(batch)
    score = metrics.silhouette_score(total_batch, total_out, metric='euclidean')
    sil_sk.append(score)

plt.plot(k_num, sil_sk)
plt.xlabel("k (the number of centroids)")
plt.ylabel("Silhouette score")
plt.savefig("./image/fig2_skearn.kmeans.png")