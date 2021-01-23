import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
#from metrics.LayerWiseMetrics import cdist2
import numpy as np

class dataset():
    def __init__(self, num, point):
        self.num = num
        self.point = point
        self.x = point[0]
        self.y = point[1]
        self.center = None
    def idx_center(self, id):
        self.center = id

class examples():
    def __init__(self, example): # example=tensor
        self.example = []

        for num, point in enumerate(example): # num=0, (0.314, 0.934)
            self.example.append(dataset(num, point)) # dataset= dataset making function

    def shuffle(self):
        random.shuffle(self.example)

class SOM():
    def __init__(self, num_centeroid, dim, learning_rate, epoch, num_example, distance=True):

        self.centeroid = torch.rand(size=(self.num_centeroid, dim)) # dim = same as dataset_dim
        self.distance = distance
        self.lr = learning_rate
        self.decay = learning_rate / num_example / epoch
        self.pca_centeroid = None
        self.neigx = torch.arange(dim)
        self.neigy = torch.arange(dim)

    def find_BMU(self, input):
        #find BMU index ->to find closest centroid index about input
        neighbor_value = 0.0
        if self.distance: # input=tensor
            neighbor_value = self.eucli(self.centeroid, input)
        else:
            neighbor_value = self.cos(self.centeroid, input)
        bmu_idx = torch.argmin(neighbor_value) 
        return bmu_idx

    def eucli(self, centeroid, input): # input=tensor([0.658,0.473])
        if input.point.ndim == 1:
            changed_input = input.point.view(1,-1) # changed_input=tensor([[0.658,0.473]])
        return torch.cdist(centeroid, changed_input)

    def cos(self, centeroid, input):
        return torch.nn.CosineSimilarity(centeroid, input.point)

    def neighbor(self, bmu, sigma): # gaussian function/bmu= 1 winning center point, sigma=number
        d = 2 * np.pi * sigma * sigma
        bmu = bmu.unsqueeze(0) # tensor([0.2354,0.8374])->tensor([[0.2354,0.8374]])
        neighbor_value = torch.exp(-torch.cdist(self.centeroid, bmu) / d) # distance of winning center and other centroids
        # torch.cdist:
        return neighbor_value # =tensor?

    def update(self, input, sigma): # input=exmples, sigma=just number?
        idx = self.find_BMU(input)
        bmu = self.centeroid[idx] # found closest centroid
        neigh = self.neighbor(bmu, sigma)
        self.centeroid += neigh * self.lr * (input.point - self.centeroid) # centroid updates
        return

    def update_learning_rate(self):
        self.lr -= self.decay
        return

    def clustering(self, inputs):
        #find centeroid?/ using updated centroid
        for i in range(len(inputs.example)): # input data 
            if self.distance:
                dist = self.eucli(self.centeroid, inputs.example[i])
                id = torch.argmin(dist) # find the closest cneter of each datapoints
                inputs.example[i].idx_center(id) # mapping center with input points

            else:
                dist = self.cos(self.centeroid, inputs[i])
                id = torch.argmin(dist)
                inputs.example[i].idx_center(id)

    # def mapping(self, inputs):
    #     pca = PCA(n_components=5)
    #     pca_centeroid = pca.fit_transform(self.centeroid)
    #     pca_input = pca.fit_transform(inputs)
    # 
    #     explained_variance = pca.explained_variance_ratio_
    #     return pca_centeroid, pca_input, explained_variance

    def visualize(self, data):
        color = ['red', 'green', 'blue', 'yellow', 'black', 'c', 'm', ]
        #after update
        self.clustering(data) # input-centroid mapping
        #pca_centeroid, pca_input = self.mapping(input)
        pca_centeroid, pca_input = self.centeroid, data

        for i in range(len(pca_centeroid)):
            plt.scatter(pca_centeroid[i][0], pca_centeroid[i][1], c=color[i], marker='x', s=100)

        for example in pca_input.example: # plt.scatter(x-axis, y-axis):
            plt.scatter(example.x, example.y, c=color[example.center])

        plt.show()

if __name__ == "__main__":
    epoch = 3
    num_centroid = 7
    dim = 2
    learning_rate = 0.1
    num_example = 100
    sigma = 0.1

    data = torch.rand(size=(num_example, dim)) # tensor(100,2)
    data = examples(data)

    model = SOM(num_centroid, dim, learning_rate, epoch, num_example, True)

    for _ in range(epoch):
        data.shuffle()
        for example in data.example: # input each 1 datapoint (x,y)
            model.update(example, sigma) # update centroids using all inputs

    model.visualize(data) # 
