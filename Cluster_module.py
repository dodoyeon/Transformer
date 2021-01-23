import torch
import torch.nn as nn
import numpy as np
# from kmeans_pytorch import kmeans

##### Heejun - SOM reference ####
class cluster_som(nn.Module):
    def __init__(self,n_centroid, epochs, distance='euclid', learning_rate=0.1, sigma=0.1):
        super().__init__()
        self.num_centroid = n_centroid
        self.epochs = epochs
        self.lr = learning_rate
        self.sigma = sigma
        self.distance = distance

#### measuring distance (or similarity)####
    def euclid(self, input, memory): # input=tensor([0.658,0.473])
        # 2-way to get distance 1.size(7) 2.size(200,7,1)
        # 1. 웨이트가 그냥 7 사이즈이면 너무 웨이트가 단순해서 문제가 있을수 있다
        # loss를 그냥 distance를 더하면 합쳐지기 때문에 그냥 데이터 포인트들의 평균으로 모아진다.. 아마 로스가 그러면 여러 문제가 생길거같아
        distance = []
        for i in range(memory.size(0)):
            distance.append(torch.dist(input,memory[i,:,:]))
        distance = torch.tensor(distance) # distance.size=(7)
        # distance = torch.cdist(memory.transpose(0,1), input.unsqueeze(0).transpose(0,1))  # 2. mem(200,7,512)/input(200,1,512) -> (200,7,1)
        # torch.cdist(in1,in2):in1(s,b,e) in2(s,c,e) -> out(s,b,c)
        return distance

    def cos(self, input, memory):
        cos = torch.nn.CosineSimilarity(dim=0)
        # torch.nn.CosineSimilarity(input1, input2, dim) input1(b,s,e) input2(c,s,e) ->dim=0 output(s,e)
        return cos(memory, input.point)

    def attn(self,Q,K): # Q=input(200,512), K=memory(7,200,512)
        weight = torch.matmul(Q.unsqueeze(0),K.transpose(-2,-1)) / (K.size(1)**0.5) #(7,200,200)
        mixture = torch.matmul(weight, K) # (7,200,512)
        return weight, mixture

#### about Clustering ####
    def find_winning_center_mixture(self, input_datapoint, memory):
        #find BMU index or mixture
        neighbor_value = 0.0
        if self.distance == 'euclid': # input=tensor
            neighbor_value = self.euclid(input_datapoint, memory)
            # 2- way to make weight
            # prob_neigh = neighbor_value/torch.sum(neighbor_value) # 1
            prob_neigh = nn.Softmax(dim=0)(neighbor_value) # 2

            # 2-way to mult weight and mem
            prob_neigh = prob_neigh.unsqueeze(1).expand(-1, 7) # 1
            mem = memory.reshape((memory.size(0), -1))
            weighted_center = torch.matmul(prob_neigh, mem) / mem.size(0)
            weighted_center = weighted_center.reshape(-1, memory.size(1), memory.size(2)) # size(7,200,512)
            # weighted_center = torch.matmul(prob_neigh, memory)# 2. weight(200,7,1), memory size=(7,200,512)

        elif self.distance == 'cos':
            neighbor_value = self.cos(input_datapoint, memory)
            prob_neigh = neighbor_value / torch.sum(neighbor_value)
            weighted_center = torch.matmul(prob_neigh,memory)
            
        elif self.distance == 'attn':
            neighbor_value, weighted_center = self.attn(input_datapoint, memory)

        else:
            print("find winning center or mixture ERROR")
        # search argmin layer
        bmu_idx = torch.argmin(neighbor_value) # about all centroids
        weighted_center = weighted_center.sum(dim=0)  # size(200,512)
        return bmu_idx, weighted_center #=updated output in cluster module
 
    def neighbor_som(self, bmu, memory, sigma): # gaussian function/bmu= 1 winning center point, sigma=number
        d = 2 * np.pi * sigma * sigma
        # unsqueeze: tensor([0.2354,0.8374])->tensor([[0.2354,0.8374]]), size(1,200,512)
        neighbor_value = torch.exp(torch.cdist(memory.transpose(0,1), bmu.unsqueeze(1)) / d) # distance of winning center and other centroids
        return neighbor_value # =tensor?

    # using "for i in range(): self.centroid=set of centroids"
    def update_som(self, input_datapoint, memory): # input=exmples, sigma=just number?
        idx, mixture = self.find_winning_center_mixture(input_datapoint, memory)
        centroid = memory[idx,:,:] # found closest centroid /size(200,512)
        neigh = self.neighbor_som(centroid, memory, self.sigma) # size(200,7,1)
        dir = (mixture - centroid).unsqueeze(1)
        memory += (neigh * self.lr * dir).transpose(0,1)  # centroid(memory) updates
        return memory

    def update_learning_rate_som(self):
        self.lr -= self.decay
        return

    def clustering_som(self, inputs,memory):
        #find centeroid
        for i in range(inputs.size(0)): # input data
            if self.distance == 'euclid':
                dist = self.euclid(inputs[i,:,:], memory)
                prob = nn.Softmax(dim=0)(dist)

                mem = memory.reshape((memory.size(0), -1))
                output = torch.matmul(prob.unsqueeze(1).expand(-1, 7), mem) / mem.size(0)
                output = output.reshape(-1, memory.size(1), memory.size(2))  # size(7,200,512)

            elif self.distance == 'cos':
                dist = self.cos(centeroid, inputs[i])
                output = 0

            elif self.distance == 'attn':
                prob, output = self.attn(input_datapoint, memory)
                dist = prob
            else:
                print("clustering ERROR")

        return output, dist, prob

    def forward(self,input):
        # cluster centroid initialization
        memory = torch.rand(size=(self.num_centroid, input.size(1), input.size(2)))
        for _ in range(self.epochs):
            for j in range(input.size(0)):
                memory = self.update_som(input[j,:,:], memory)

        output, dist, prob = self.clustering_som(input,memory)
        loss = torch.mean(prob * dist)

        return output, loss, memory


# epoch = 3
# num_centroid = 7
# learning_rate = 0.1
# sigma = 0.1
#
# batch_size=32
# emb_dim=512
# max_len=200
#
# data = torch.rand(size=(batch_size, max_len, emb_dim)) # attn output
#
# model = cluster_som(num_centroid, epoch, distance='euclid', learning_rate=learning_rate, sigma=sigma)
#
# loss, memory, output = model(data)


#### 3D matmul ####
# x = torch.tensor([[[1,2],[3,4],[5,6]],[[6,5],[4,3],[2,1]]]) # size()
# y = torch.tensor([[[2,4,6,8],[10,12,14,16]],[[1,3,5,7],[9,11,13,15]]]) # size()
# res = torch.matmul(x,y) # size(
# print(res)

s = torch.ones(1)
print(s)
print(s.unsqueeze(1).size())