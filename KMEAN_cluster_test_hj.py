import pdb
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import random
from Dataset import tr_cent_data, te_cent_data, tr_centroid, te_centroid
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

class KMEAN_Memory(nn.Module):
    def __init__(self, cnum_centroid, hidden_size, max_seq_length):
        super().__init__()
        self.num_centroid = num_centroid
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.centroid = nn.Embedding.from_pretrained(torch.normal(0.0, 0.39, size=(self.num_centroid,
                                          self.max_seq_length *
                                          self.hidden_size)))
    def forward(self, input):
        if self.centroid.weight.requires_grad == True:
            #pdb.set_trace()
            new_input = input.view(-1, self.max_seq_length*self.hidden_size)
            centroid = self.centroid.weight
            cluster_ids_input, new_centers = kmeans(new_input, num_clusters=self.num_centroid, cluster_centers=centroid, distance='euclidean')
            self.centroid = self.centroid.from_pretrained(new_centers)
            return self.centroid.weight, cluster_ids_input
        else:
            return self.centroid.weight, cluster_ids_input


def cent_table(centroid_updated, centroid_trg):
    dist = torch.cdist(centroid_updated, centroid_trg)  # (8,8)
    table = []
    for i in range(dist.size(0)):
        idx = torch.argmin(dist)
        row_idx = idx // dist.size(0)
        col_idx = idx % dist.size(0)
        table.append([row_idx, col_idx])
        dist = torch.cat(dist[:row_idx, :], dist[row_idx:, :])
        dist = torch.cat(dist[:, :col_idx], dist[:, col_idx:])
    return table


def mapping(idx, table):
    mapped_idx = table[idx.long()]  
    return mapped_idx


if __name__ == "__main__":
    num_centroid = 8
    max_seq_length = 128
    hidden_size = 768
    # device = 'cpu'
    # std = np.sqrt(0.2)
    tr_example_size = 3200
    te_example_size = 640
    batch_size = 32
    num_centroid = 8
    epochs = 5

    # with gzip.open('tr_data.pickle','rb') as f:
    #     tr_cent_data = pickle.load(f)

    model = KMEAN_Memory(num_centroid, hidden_size, max_seq_length).to(device)
    tr_data = tr_cent_data[:, :-1]  # (320,98304)
    tr_target = tr_cent_data[:, -1]  # (320)

    model.train()
    model.centroid.weight.requires_grad = True
    correct = 0

    print("clustering start")
    for epoch in range(epochs):  # unsupervised learning 이니까 아마도 에폭이 의미가 없는것 같다
        # For silhouette score
        tr_out = torch.ones(1).unsqueeze(0).to(device)
        model.update_lr(batch_size * (epoch))
        for b in range(0, len(tr_data), batch_size):
            batch = tr_data[b:b + batch_size, :].to(device)  # tensor(32,98304)
            centroid, result = model(batch)

            tr_out = torch.cat((tr_out, result.unsqueeze(1)), 0)
            # correct += (real_idx == batch_trg).sum().item()

        # mapping predict and real
        table = cent_table(centroid.cpu(), tr_centroid)  # table tensor([2,3,4,4,1,1,1,4])
        real_idx = mapping(tr_out[1:, :], table).squeeze()
        correct += (real_idx == tr_target).sum().item()

        # Get Silhouette Score
        tr_out_arr = tr_out[1:, :].squeeze().cpu().detach().numpy()
        tr_data_arr = tr_data.numpy()
        s_score = metrics.silhouette_score(tr_data_arr, tr_out_arr, metric='euclidean')
        print("Accuracy of the Cluster in training: %d epochs| %.3f %%" % (epoch, 100 * correct / tr_example_size))
        print("train Silhouette score: %.3f" % (s_score))
        correct = 0

    # with gzip.open('te_data.pickle','rb') as f:
    #     te_cent_data = pickle.load(f)

    model.eval()
    te_data = te_cent_data[:, :-1]  # (320,98304)
    te_target = te_cent_data[:, -1]  # (320)
    correct = 0
    te_out = torch.ones(1).unsqueeze(0).to(device)
    with torch.no_grad():
        model.centroid.weight.requires_grad = False
        for b in range(0, len(te_data), batch_size):
            batch = te_data[b:b + batch_size, :].to(device)
            batch_trg = te_target[b:b + batch_size].to(device)
            centroid, pred = model(batch)  # False

            te_out = torch.cat((te_out, pred.unsqueeze(1)), 0)

        # mapping predict and real
        table = cent_table(centroid.cpu(), te_centroid)  # table tensor([2,3,4,4,1,1,1,4])
        real_idx = mapping(te_out[1:, :], table).squeeze()
        correct += (real_idx == te_target).sum().item()

        # Get Silhouette Score
        te_out_arr = te_out[1:, :].squeeze().cpu().detach().numpy()
        te_data_arr = te_data.numpy()
        s_score = metrics.silhouette_score(te_data_arr, te_out_arr, metric='euclidean')
        print("Accuracy of the Cluster in testing: %.3f %%" % (100 * correct / te_example_size))
        print("test Silhouette score: %.3f" % (s_score))