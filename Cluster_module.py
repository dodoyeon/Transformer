import torch
import torch.nn as nn
import numpy as np
from kmeans_pytorch import kmeans

class cluster_SOM(nn.Module):
    def __init__(self, n_centroid, dim, sigma, lr=0.5):
        super().__init__()
        self.num_centroid =n_centroid
        self.lr = lr

    def euclidean_dist(self, input, cent):

    def cos_sim(self, input, cent):
        return torch.nn.CosineSimilarity(cent, input)

    def neighbor(self, , sigma):
        d = 2*np.pi*sigm*sigma
        h = torch.exp(cdist()/d)

    def forward(self, input):
        matmul(input, memory) # similarity
        memory = self.memory

        return loss, output, memory

class cluster(nn.Module):
    def __init__(self,n_centroid, epochs, distance='euclid'):
        self.num_centroid = n_centroid
        self.epochs = epochs
        self.distance = distance

    def attn(self,Q,K): # Q=input, K=memory
        weight = torch.matmul(Q,K) / K.size(1)**0.5
        mixture = torch.matmul(weight, K)
        return mixture, weight

    def euclid(self, centeroid, input): # input=tensor([0.658,0.473])
        if input.point.ndim == 1:
            changed_input = input.point.view(1,-1) # changed_input=tensor([[0.658,0.473]])
        return torch.cdist(centeroid, changed_input)

    def cos(self, centeroid, input):
        return torch.nn.CosineSimilarity(centeroid, input.point)

    def find_wining_center(self, input):
        #find BMU index
        neighbor_value = 0.0
        if self.distance == 'euclid': # input=tensor
            neighbor_value = self.euclid(self.centeroid, input)
        elif self.distance == 'cos':
            neighbor_value = self.cos(self.centeroid, input)
        bmu_idx = torch.argmin(neighbor_value) # about all centroids
        return bmu_idx

    def find_mixture(self, input):
        #find BMU index
        neighbor_value = 0.0
        if self.distance == 'euclid': # input=tensor
            neighbor_value = self.euclid(self.centeroid, input)
            prob_neigh = neighbor_value/torch.sum(neighbor_value)
            torch.matmul(prob_neigh)

        elif self.distance == 'cos':
            neighbor_value = self.cos(self.centeroid, input)

        bmu_idx = torch.argmin(neighbor_value) # about all centroids
        return bmu_idx

    def update_som(self, input, sigma): # input=exmples, sigma=just number?
        idx = self.find_wining_center(input)
        bmu = self.centeroid[idx] # found closest centroid
        neigh = self.neighbor_som(bmu, sigma)
        self.centeroid += neigh * self.lr * (input.point - self.centeroid) # centroid updates
        return

    def update_learning_rate_som(self):
        self.lr -= self.decay
        return

    def clustering_som(self, inputs):
        #find centeroid
        for i in range(len(inputs.example)): # input data
            if self.distance == 'euclid':
                dist = euclid(self.centeroid, inputs.example[i])
                id = torch.argmin(dist) # find the closest cneter of each datapoints
                inputs.example[i].idx_center(id) # assigned center

            elif self.distance == 'cos':
                dist = cos(self.centeroid, inputs[i])
                id = torch.argmin(dist)
                inputs.example[i].idx_center(id)

    def neighbor_som(self, bmu, sigma): # gaussian function/bmu= 1 winning center point, sigma=number
        d = 2 * np.pi * sigma * sigma
        bmu = bmu.unsqueeze(0) # tensor([0.2354,0.8374])->tensor([[0.2354,0.8374]])
        neighbor_value = torch.exp(torch.cdist(self.centeroid, bmu) / d) # distance of winning center and other centroids

        return neighbor_value # =tensor?

    def forward(self,input):
        distance = []
        for _ in range(self.epochs):
            for datapoint in enumerate(input):
                # dist = torch.nn.CosineSimilarity(datapoint, centroid)
                # distance.append(dist)
        # prob = distance/torch.sum(distance)
        # cent_dist= torch.tensor(distance)
        # prob = nn.Softmax(cent_dist)
        mem.append()
        loss = torch.mean()
        return loss, mem