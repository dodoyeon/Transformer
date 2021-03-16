import torch
from torch import nn
import torch.nn.functional as F
import math
from sklearn import metrics
import random
import gzip
import pickle
from Dataset import tr_centroid, tr_cent_data, te_cent_data
from torch.utils.tensorboard import SummaryWriter

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class loss_mod_Memory(nn.Module):
    def __init__(self, num_centroid, hidden_size, max_seq_length):
        super(loss_mod_Memory, self).__init__()
        self.num_centroid = num_centroid
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        # self.device = config.device
        # self.softmax = nn.Softmax()
        self.centroid = nn.Embedding.from_pretrained(torch.normal(0, 1, size=(self.num_centroid,
                                                                                    self.max_seq_length *
                                                                                    self.hidden_size)), freeze=False)
        # self.length = config.length
        # self.lr = config.learning_rate

    def entropy_loss(self, dist):
        self.prob = F.softmax(dist, dim=-1)
        # loss = -torch.sum(self.prob * torch.log(self.prob)) # 마이너스 추가
        H = self.prob * F.log_softmax(dist, dim=-1)
        loss = - torch.sum(H)
        # loss = -torch.sum(self.prob.entropy())
        return loss

    def silhoutte_loss(self, input, dist, idx): # 근데 sil loss는 배치 32개에 대해서만 고려할수밖에 없어서 결과가 안좋을거같다..
        sil = 0
        for i in range(idx.size(0)): # i는 기준이 되는 data point
            num = 1e-8
            d = 0
            b_i = 1e10
            for j in range(i+1, idx.size(0)): # j는 32개중 i와는 다른 data points
                if idx[j] == idx[i]: # 그 중 센트로이드가 같은 것들
                    num += 1
                    d += torch.dist(input[i,:], input[j,:])

                else: # 센트로이드가 다른 것들
                    d_i = torch.dist(input[i,:], input[j,:])
                    # if b_i >= d_i:# b_i=개체와 다른 군집 내 개체들간의 거리를 군집별로 구하고, 이중 가장 작은 값
                    #     break
                    # else:
                    #     b_i = d_i
                    b_i = min(b_i, d_i)
            a_i = d / num  # a_i=개체와 같은 군집에 속한 요소들 간 거리들의 평균
            sil += (b_i - a_i)/max(a_i, b_i)
        loss = (sil/idx.size(0))-1 # 평균
        return loss

    def forward(self, input):
        # 인풋과 센트로이드 사이 거리(loss)구하고 제일 가까운 센터 구함
        # 1. just euclidean dist metric
        self.dist = torch.cdist(input, self.centroid.weight)
        idx = torch.argmin(self.dist, dim=-1)

        # 2. Cos similarity
        # self.dist = F.cosine_similarity(input.unsqueeze(-1).expand(-1,-1,8),
        #                                 self.centroid.weight.unsqueeze(0).expand(32,-1,-1).transpose(1,-1))
        # idx = torch.argmax(self.dist, dim=-1)

        # 3. semantic metric using HSIC / cka
        centroid = self.centroid.weight.view(-1, self.max_seq_length, self.hidden_size)
        input = input.view(-1, self.max_seq_length, self.hidden_size)
        input = input.unsqueeze(1) # (32,1,128,768)
        cka_loss_matrix = self.linear_cka_loss(input, centroid)  # input(32,128,768) cent(8,128,768) -> (32,8)
        idx = torch.argmax(cka_loss_matrix, dim=-1)
        # get selected cent and loss
        selected_cetroid = self.centroid(idx)
        selected_centroid = selected_cetroid.view(-1, self.max_seq_length, self.hidden_size)
        cka_loss = self.linear_cka_loss(input, selected_centroid, matrix=False) # cka_loss(size=1) cka_loss_mat(size=32x8) idx(32)
        # q = torch.exp(self.dist) # for sharpening

        # i) probability
        # self.prob = self.dist/torch.sum(self.dist, dim=-1, keepdim=True).expand(-1, self.dist.size(-1))

        # ii) use 'softmax' to get prob
        # Tip) nn.Softmax(): 은 함수 그자체/ 쓰려면 self.softmax=nn.Softmax()하고, self를 써야한다 / F.softmax
        # cka_loss = self.entropy_loss(self.dist)

        # iii) "Learning to learn rare event" 논문 참조 k개 뽑아서 prob 얻음
        # k = 4  # centroid index
        # dist = torch.sort(self.dist)  # 그의 prob 값
        # cka_loss = self.entropy_loss(dist.values[:, :k - 1]) # cos similarity를 사용하기 위해서는 loss를 다시 마이너스를 취해줘야한다(gradient ascent)

        # 4. NEW distance loss
        # cka_loss = torch.sum(self.dist[:, :k - 1]) # 평균: /k 추가 & k개 뽑는 코드 추가
        # cka_loss = torch.sum(self.dist)

        # 5.
        # cka_loss = - self.silhoutte_loss(input, torch.min(self.dist,dim=1).values, idx) # -: 클수록 좋으므로

        # 6. 데이터와 cent간의 거리 합/centroid간의 거리 합
        # cent_dist = torch.cdist(self.centroid.weight, self.centroid.weight)
        # cka_loss = torch.sum(self.dist)-(torch.sum(cent_dist)/2)

        return cka_loss, self.dist, self.centroid.weight, idx

    # 논문 "Similarity of Neural Network Representations Revisited": Hilbert-Schmidt Independence Criterion
    # centered kernel alignment(CKA)
    def linear_cka_loss(self, x, y, matrix=True):
        hsic = self.linear_HSIC(x, y)
        var1 = torch.sqrt(self.linear_HSIC(x, x))
        var2 = torch.sqrt(self.linear_HSIC(y, y))
        if matrix:
            return -torch.log(torch.abs(torch.div(hsic, (var1 * var2))) + 1e-8)
        else:
            return -torch.log(torch.mean(torch.abs(torch.div(hsic, (var1 * var2)))) + 1e-8)

    def centering(self, input):
        n = input.size()[-1]
        unit = torch.ones(size=(n, n)).to('cuda')
        I = torch.eye(n).to('cuda')
        H = I - unit / n
        return torch.matmul(torch.matmul(H, input), H)

    def linear_HSIC(self, x, y):
        if x.dim() >= 3 and y.dim() >= 3:
            l_x = torch.matmul(x, x.transpose(-2, -1))  # (32,128,128)
            l_y = torch.matmul(y, y.transpose(-2, -1))  # (8,128,128)
            return torch.sum(torch.sum(torch.mul(self.centering(l_x), self.centering(l_y)), dim=-1),
                             dim=-1)  # torch.mul:element-wise mul
        else:
            l_x = torch.matmul(x, x.transpose(-2, -1))
            l_y = torch.matmul(y, y.transpose(-2, -1))
            return torch.sum(torch.mul(self.centering(l_x), self.centering(l_y)))


def cent_table(centroid_updated, centroid_trg):
    dist = torch.cdist(centroid_updated, centroid_trg.cpu())  # (8,8)
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
            # break 를 쓰면 안된다..table을 7개만 만들고 멈출수 있음
        i += 1
    table = torch.tensor(table)
    return table

def mapping(idx, table):  # idx:tensor table:tensor?or list?
    dict_table = {}
    mapped_idx = []
    for i in range(table.size(0)):
        dict_table[table[i, 0].item()] = table[i, 1].item()

    for j in range(idx.size(0)):
        map_idx = dict_table[idx[j].item()]
        mapped_idx.append(map_idx)
    mapped_idx = torch.tensor(mapped_idx)
    return mapped_idx

# TENSOR BOARD
writer = SummaryWriter('runs/1_Ann_test_1')

def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))[:n] # 주어진 범위 내의 정수를 랜덤하게 생성 (=indices)
    return data[perm], labels[perm]

if __name__ == "__main__":
    num_centroid = 8
    max_seq_length = 128
    hidden_size = 768
    tr_example_size = 6400
    te_example_size = 640
    batch_size = 32
    learning_rate = 5e-5
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_sne = TSNE(n_components=2, random_state=0)  # n_component= embedded space의 dim
    colors = ['r', 'b', 'g', 'y', 'k',
              'violet', 'springgreen', 'dodgerblue']

    # with gzip.open('tr_data.pickle','rb') as f:
    #     tr_cent_data = pickle.load(f)

    model = loss_mod_Memory(num_centroid, hidden_size, max_seq_length).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    tr_data = tr_cent_data[:, :-1]  # (320,98304)
    tr_target = tr_cent_data[:, -1]  # (320)

    model.train()
    model.centroid.weight.requires_grad = True
    total_loss = 0
    correct = 0
    cent_wb = torch.zeros(8,98304)
    for epoch in range(epochs): # unsupervised learning 이니까 아마도 에폭이 의미가 없는것 같다
        # For silhouette score
        tr_out = torch.ones(1).unsqueeze(0).to(device)
        for b in range(0, len(tr_data), batch_size):
            tr_batch = tr_data[b:b + batch_size, :].to(device)  # tensor(32,98305)
            tr_batch_trg = tr_target[b:b + batch_size].to(device)

            optimizer.zero_grad()
            cka_loss, cka_loss_matrix, centroid, result = model(tr_batch)  # loss(0.1553) loss_mat(32,8) cent(8,98304) res(32)

            tr_out = torch.cat((tr_out, result.unsqueeze(1).float()), 0)

            # 잘못 생각한 cross entropy 백프롭
            # cross_loss = criterion(result.view(-1, result.size(-1)), tr_batch_trg)
            # cross_loss.backward()

            # cka_loss는 grad_fn이 없어서 optimizer와 backprop() 식으로는 안된다 -> no
            # cka_loss.requires_grad = True # 이걸 직접 여기에 쓰면 안될거같다
            cka_loss.backward()
            optimizer.step()
            total_loss += cka_loss.item()

            interval = 50
            # print(i)
            if b % interval == 0 and b > 0:
                # print(b)
                avg_loss = total_loss / interval
                # ppl = math.exp(avg_loss)

                print("epoch: %d | b: %d | loss: %.3f " % (epoch + 1, b, avg_loss)) # | ppl: %.3f , ppl
                writer.add_scalar('Loss/train', avg_loss, (epoch*6400+b))
                cent_wb = torch.cat((cent_wb, centroid.detach().cpu()), dim=0)
                total_loss = 0

    table = cent_table(centroid.cpu(), tr_centroid)
    real_idx = mapping(tr_out[1:, :].squeeze().cpu(), table)
    correct += (real_idx == tr_target.cpu()).sum().item()
    print("Accuracy of the Cluster in training: %.3f %%" % (100 * correct / tr_example_size))
    correct = 0

    # Get Silhouette Score
    tr_out_arr = tr_out[1:, :].detach().squeeze().cpu().numpy()
    tr_data_arr = tr_data.cpu().numpy()
    s_score = metrics.silhouette_score(tr_data_arr, tr_out_arr, metric='euclidean')
    print("train Silhouette score: %.3f" % (s_score))

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

    model.eval()

    # with gzip.open('te_data.pickle', 'rb') as f:
    #     te_cent_data = pickle.load(f)

    te_data = te_cent_data[:, :-1]  # (320,98304)
    te_target = te_cent_data[:, -1]  # (320)
    total_loss = 0
    te_out = torch.ones(1).unsqueeze(0).to(device)
    correct = 0
    cent_wb = torch.zeros(8, 98304)
    with torch.no_grad():
        model.centroid.weight.requires_grad = False
        for b in range(0, len(te_data), batch_size):
            te_batch = te_data[b:b + batch_size, :].to(device)
            te_batch_trg = te_target[b:b + batch_size].to(device)
            cka_loss, cka_loss_matrix, centroid, pred = model(te_batch)
            total_loss += cka_loss.item()

            te_out = torch.cat((te_out, pred.unsqueeze(1).float()), 0)

            interval = 10  # batch 단위로 프린트 해야함 jwp
            if b % interval == 0 and b > 0:
                avg_loss = total_loss / interval
                # ppl = math.exp(avg_loss)
                cent_wb = torch.cat((cent_wb, centroid.detach().cpu()), dim=0)
                print("b: %d | loss: %.3f" % (b, avg_loss)) # | ppl: %.3f , ppl
                writer.add_scalar('Loss/test', avg_loss, b)
                total_loss = 0
    writer.close()

    # Get Silhouette Score
    te_out_arr = te_out[1:, :].squeeze().cpu().detach().numpy()
    te_data_arr = te_data.cpu().numpy()
    s_score = metrics.silhouette_score(te_data_arr, te_out_arr, metric='euclidean')
    print("test Silhouette score: %.3f" % (s_score))

    # mapping predict and real
    table = cent_table(centroid.cpu(), tr_centroid)
    real_idx = mapping(te_out[1:, :].squeeze().cpu(), table)
    correct += (real_idx == te_target.cpu()).sum().item()
    print("Accuracy of the Cluster in testing: %.3f %%" % (100 * correct / te_example_size))

    # t-sne
    cat = torch.cat((te_data[:500, :].cpu(), tr_centroid.cpu(), cent_wb[8:, :]),
                    dim=0)  # (500,98304) (8,98304) (8*36,98304)
    centroid_te = t_sne.fit_transform(cat.numpy())

    for i in range(3):
        # for j in range(num_centroid):
        plt.figure(i)
        plt.scatter(centroid_te[:500, 0], centroid_te[:500, 1], c='gray', marker='o') # tr_data
        plt.scatter(centroid_te[500:508, 0], centroid_te[500:508, 1], c=colors, marker='x') # tr_target
        plt.scatter(centroid_te[508+(i*8):508+((i+1)*8), 0], centroid_te[508+(i*8):508+((i+1)*8), 1], c=colors, marker='s') # cent_wb

        plt.title("t-SNE dataset+centroid b: %d" %i)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
