import pdb
import torch
from torch import nn
import numpy as np
import random
from Dataset import tr_cent_data, te_cent_data, tr_centroid
# from metrics.LayerWiseMetrics import cdist2
# from kmeans_pytorch import kmeans
from sklearn import metrics
from sklearn.cluster import KMeans
import gzip
import pickle
from sklearn.manifold import TSNE
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################SOM clustering Memory ########################
def cdist2(x, y, eps=1e-8): # dist between x='selected cent'(32,98304) and y='centroids'(8, 98304)
    if y.ndim == 1:
        y = y.view(1, -1)
    x_sq_norm = x.pow(2).sum(dim=-1, keepdim=True)# ^2->sum : magnitude^2 of x(98304->1) ->(32,1)
    y_sq_norm = y.pow(2).sum(dim=-1)
    if y_sq_norm.ndim == 1:
        y_sq_norm = y_sq_norm.unsqueeze(0) # (1,8)
    x_dot_y = x @ y.transpose(-1,-2) if len(y) > 1 else x @ y.transpose(0,1)
    sq_dist = x_sq_norm + y_sq_norm.unsqueeze(dim=-2) - 2 * x_dot_y
    sq_dist.clamp_(min=0.0)
    return torch.sqrt(sq_dist + eps)

class SOM_Memory(nn.Module):
    def __init__(self, num_centroid, hidden_size, max_seq_length, lr):
        super().__init__()
        self.num_centroid = num_centroid
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        # self.device = device
        self.centroid = nn.Embedding.from_pretrained(torch.normal(-0.07, 0.39, size=(num_centroid,
                                                    max_seq_length * hidden_size)), freeze=False)
        # self.centroid = torch.normal(-0.01, 0.44, size=(num_centroid,max_seq_length * hidden_size))
        self.lr = lr

    def forward(self, input): # requires_grad
        #new_centroid = self.centroid.clone().detach()
        #pdb.set_trace()
        if self.centroid.weight.requires_grad == True:
            centroid = self.centroid.weight # (8,98304)
            # centroid = self.centroid.to(device)
            bmu, prob, dist, idx = self.bmu(input, centroid) # 근데 왜 distance가 약193.9x-194.4x로 너무 서로 붙어있지..?
            neighbor_value = self.neighbor(bmu, centroid).squeeze()
            #pdb.set_trace()
            neighbor_value = neighbor_value.view(-1,1, self.num_centroid).transpose(-1,-2)
            neighbor = neighbor_value * self.lr * (input.view(-1,1,self.hidden_size * self.max_seq_length) - centroid)
            new_weights = torch.add(centroid, torch.mean(neighbor, dim=0)) # size(8,98304)
            # self.centroid = new_weights
            self.centroid = self.centroid.from_pretrained(new_weights)
            return prob, self.centroid.weight, idx # (prob, self.centroid)
        else:
            centroid = self.centroid.weight
            bmu, prob, dist, idx = self.bmu(input, centroid) # input(32,98304)/cent(8,98304)
            return prob, self.centroid.weight, idx # (prob, self.centroid)

    def bmu(self, input, centroid): # get dist=disance between input and centroid/ idx=smallest cent among centroids / bmu=select centroid /
        input = input.view(-1,self.hidden_size * self.max_seq_length) # 나는 굳이 필요없는 부분
        dist = -torch.cdist(input, centroid) # (32,8) -> distance=54358.4688???
        idx = torch.argmax(dist, dim=-1) # (32)
        bmu = centroid[idx].squeeze() # (32,98304)
        prob = nn.Softmax(dim=-1)(dist) # (32,8)  
        idx = idx.float()
        return bmu, prob, dist, idx

    def neighbor(self,bmu, centroid): # bmu=selected cent/centroid=cents
        dist = cdist2(bmu, centroid)
        value = torch.exp(-dist / 1000)
        return value

    def update_lr(self, length):
        if length == 0:
            self.lr = self.lr
        else:
            self.lr -= self.lr / length

def cent_table(centroid_updated, centroid_trg):
    dist = torch.cdist(centroid_updated, centroid_trg) # (8,8)
    table = []
    dist_sort = dist.view(1,-1) # (1,64)
    sort = torch.argsort(dist_sort,dim=1).squeeze()
    i=0
    idx = sort[i].item()
    row_idx = (idx // dist.size(0))  # 타겟에 대한 업데이트 센트로이드의 인덱스
    col_idx = (idx % dist.size(0))  # 업데이트에 대한 타겟의 인덱스
    table.append([row_idx, col_idx])
    i += 1
    while len(table) < dist.size(0):
        if i < (dist.size(0))**2:
            idx = sort[i].item()
            row_idx = (idx // dist.size(0)) # 타겟에 대한 업데이트 센트로이드의 인덱스
            col_idx = (idx % dist.size(0)) # 업데이트에 대한 타겟의 인덱스
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
            # break 를 쓰면 안된다..table을 7개만 만들고 멈출수 있음
        i += 1
    table = torch.tensor(table)

    # for i in range(dist.size(0)):
        # row_idx = torch.argmin(dist, dim=-1)
        # col_idx = torch.min(torch.argmin(dist,dim=0),dim=-1)
        # idx = torch.argmin(dist)
        # row_idx = idx//dist.size(0) # 타겟에 대한 업데이트 센트로이드의 인덱스
        # col_idx = idx % dist.size(0) # 업데이트에 대한 타겟의 인덱스
        # table.append([row_idx, col_idx])
        # dist = torch.cat((dist[:row_idx,:],dist[row_idx:,:]), dim=0)
        # dist = torch.cat((dist[:,:col_idx],dist[:,col_idx:]), dim=1)

    # idx_table = torch.argsort(dist,dim=-1) # (8,8)
    # table = torch.argmin(dist, dim=-1)  # FIXED (8)
    # for j in range(centroid_trg.size(0)):
    #     for i in range(centroid_trg.size(0)):
    #         check = (idx_table[:,j] == i) # 0,1 if True
    #         c = check.nonzero().squeeze()
    #         if c.size() != 1 && c.size() != 0:
    #             idx_table
    #         elif a.size() == 1:
    #             continue
    #         else:
    #             continue
    # table = torch.argmin(dist,dim=-1) # FIXED (8)
    return table

def mapping(idx,table): # idx:tensor table:tensor?or list?
    dict_table = {}
    mapped_idx = []
    for i in range(table.size(0)):
        dict_table[table[i, 0].item()] = table[i, 1].item()

    for j in range(idx.size(0)):
        map_idx = dict_table[idx[j].item()]
        mapped_idx.append(map_idx)
    mapped_idx = torch.tensor(mapped_idx)
    return mapped_idx

def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))[:n] # 주어진 범위 내의 정수를 랜덤하게 생성 (=indices)
    return data[perm], labels[perm]

if __name__=="__main__":
    num_centroid = 8
    max_seq_length = 128
    hidden_size = 768
    # device = 'cpu'
    # std = np.sqrt(0.2)
    tr_example_size = 3200
    te_example_size = 640
    batch_size = 32
    num_centroid = 8
    learning_rate = 5e-5
    epochs = 5

    # with gzip.open('tr_data.pickle','rb') as f:
    #     tr_cent_data = pickle.load(f)

    t_sne = TSNE(n_components=2, random_state=0)  # n_component= embedded space의 dim
    colors = ['r', 'b', 'g', 'y', 'k',
              'violet', 'springgreen', 'dodgerblue']

    model = SOM_Memory(num_centroid, hidden_size, max_seq_length, learning_rate).to(device)
    tr_data = tr_cent_data[:,:-1] # (320,98304)
    tr_target = tr_cent_data[:,-1] # (320)

    model.train()
    model.centroid.weight.requires_grad = True
    # correct = 0
    
    print("clustering start")
    cent_wb = torch.zeros(8, 98304)
    for epoch in range(epochs): # unsupervised learning 이니까 아마도 에폭이 의미가 없는것 같다
        # For silhouette score
        tr_out = torch.ones(1).unsqueeze(0).to(device)
        model.update_lr(batch_size*(epoch))
        for b in range(0, len(tr_data), batch_size):
            batch = tr_data[b:b + batch_size,:].to(device) # tensor(32,98304)
            batch_trg = tr_target[b:b + batch_size].to(device)
            output,centroid, result = model(batch)
            
            tr_out = torch.cat((tr_out, result.unsqueeze(1)), 0)
            interval = 50

            if b % interval == 0 and b > 0:
                # avg_loss = total_loss / interval
                # ppl = math.exp(avg_loss)

                # print("epoch: %d | b: %d | loss: %.3f " % (epoch + 1, b, avg_loss))  # | ppl: %.3f , ppl
                # writer.add_scalar('Loss/train', avg_loss, (epoch * 6400 + b))
                cent_wb = torch.cat((cent_wb, centroid.detach().cpu()), dim=0)
                # total_loss = 0
        
        # Get Silhouette Score
        tr_out_arr = tr_out[1:,:].squeeze().cpu().detach().numpy()
        tr_data_arr = tr_data.numpy()
        s_score = metrics.silhouette_score(tr_data_arr, tr_out_arr, metric='euclidean')
        print("train Silhouette score: %.3f" %(s_score))
        
    # mapping predict and real
    table = cent_table(centroid.cpu(), tr_centroid)
    real_idx = mapping(tr_out[1:, :].squeeze().cpu(), table)  # tr_out[1:,:].squeeze() :3200
    correct += (real_idx == tr_target).sum().item()
    print("Accuracy of the Cluster in training: %d epochs| %.3f %%" % (epoch, 100 * correct / tr_example_size))
    correct = 0

    # T-SNE visualization
    cat = torch.cat((tr_data[:500, :].cpu(), tr_centroid.cpu(), cent_wb[8:, :]),
                    dim=0)  # (500,98304) (8,98304) (8*36,98304)
    centroid_tr = t_sne.fit_transform(cat.numpy())

    for i in range((epoch+1)*7):
        # for j in range(num_centroid):
        plt.figure(i)
        plt.scatter(centroid_tr[:500, 0], centroid_tr[:500, 1], c='gray', marker='o') # tr_data
        plt.scatter(centroid_tr[500:508, 0], centroid_tr[500:508, 1], c=colors, marker='x') # tr_target
        plt.scatter(centroid_tr[508+(i*8):508+((i+1)*8), 0], centroid_tr[508+(i*8):508+((i+1)*8), 1], c=colors, marker='s') # cent_wb

        plt.title("t-SNE dataset+centroid epoch: %d b: %d" % ((i//7), (i%7)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    # epoch 설정없이
    # for b in range(0, len(tr_data), batch_size):
    #     batch = tr_data[b:b + batch_size, :].to(device)  # tensor(32,98305)
    #     batch_trg = tr_target[b:b + batch_size].to(device)
    #     output, result = model(batch)
    #
    #     correct += (result == batch_trg).sum().item()
    #
    # print("Accuracy of the Cluster in training: epochs| %.3f %%" % (100 * correct / tr_example_size))
    # correct = 0

    # with gzip.open('te_data.pickle','rb') as f:
    #     te_cent_data = pickle.load(f)

    model.eval()
    te_data = te_cent_data[:, :-1]  # (320,98304)
    te_target = te_cent_data[:, -1]  # (320)
    correct = 0
    te_out = torch.ones(1).unsqueeze(0).to(device)
    cent_wb = torch.zeros(8, 98304)
    with torch.no_grad():
        model.centroid.weight.requires_grad = False
        for b in range(0, len(te_data), batch_size):
            batch = te_data[b:b + batch_size, :].to(device)
            batch_trg = te_target[b:b + batch_size].to(device)
            output,centroid, pred = model(batch) # False

            te_out = torch.cat((te_out, pred.unsqueeze(1)), 0)
            # correct += (result == batch_trg).sum().item()

        # mapping predict and real
        table = cent_table(centroid.cpu(), te_centroid)  # table tensor([2,3,4,4,1,1,1,4])
        real_idx = mapping(te_out[1:, :], table).squeeze()
        correct += (real_idx == te_target).sum().item()
        print("Accuracy of the Cluster in testing: %.3f %%" % (100 * correct / te_example_size))

        # Get Silhouette Score
        te_out_arr = te_out[1:, :].squeeze().cpu().detach().numpy()
        te_data_arr = te_data.numpy()
        s_score = metrics.silhouette_score(te_data_arr, te_out_arr, metric='euclidean')
        print("test Silhouette score: %.3f" % (s_score))
    
    # k means clustering test
    kmeans_model = KMeans(n_clusters=8, ).fit(tr_data)
    labels = kmeans_model.labels_
    sil_score = metrics.silhouette_score(tr_data, labels, metric ='euclidean')
    print("kmeans Silhouette score: %.3f" %(sil_score))
    # cluster_ids_x, cluster_centers = kmeans(X=input, num_clusters=num_centroid, distance='euclidean', device=device)

    # a = np.arange(20)
    # b=np.random.randint(0,20,20)
    # print(a)
    # print(b)