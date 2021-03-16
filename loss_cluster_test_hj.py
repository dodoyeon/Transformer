import pdb
import torch
from torch import nn
import math
# from metrics.LayerWiseMetrics import cdist2, linear_CKA_loss
from sklearn import metrics
import random
from Dataset import tr_cent_data, te_cent_data

class loss_Memory(nn.Module):
    def __init__(self, num_centroid, hidden_size, max_seq_length):
        super(loss_Memory, self).__init__()
        self.num_centroid = num_centroid
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        #self.device = config.device
        self.centroid = nn.Embedding.from_pretrained(torch.normal(-0.01, 0.2, size=(self.num_centroid,
                                          self.max_seq_length *
                                          self.hidden_size)), freeze= False)
        #self.length = config.length
        #self.lr = config.learning_rate
        
    def forward(self, input):
        # 인풋과 센트로이드 사이 거리(loss)구하고 제일 가까운 센터 구함
        centroid = self.centroid.weight.view(-1, self.max_seq_length, self.hidden_size)
        input = input.view(-1,self.max_seq_length, self.hidden_size)
        input = input.unsqueeze(1)
        cka_loss_matrix = self.linear_cka_loss(input, centroid) # input(32,128,768) cent(8,128,768)
        idx = torch.argmax(cka_loss_matrix, dim=-1)
        
        # get selected cent and loss
        selected_cetroid = self.centroid(idx)
        selected_centroid = selected_cetroid.view(-1, self.max_seq_length, self.hidden_size)
        cka_loss = self.linear_cka_loss(input, selected_centroid, matrix=False)
        return cka_loss, cka_loss_matrix, self.centroid, idx

    # 논문 "Similarity of Neural Network Representations Revisited": Hilbert-Schmidt Independence Criterion
    # centered kernel alignment(CKA)
    def linear_cka_loss(self, x, y, matrix=True):
        hsic = self.linear_HSIC(x, y)
        var1 = torch.sqrt(self.linear_HSIC(x, x))
        var2 = torch.sqrt(self.linear_HSIC(y, y))
        if matrix:
            return -torch.log(torch.abs(torch.div(hsic,(var1 * var2))) + 1e-8)
        else:
            return -torch.log(torch.mean(torch.abs(torch.div(hsic, (var1 * var2)))) + 1e-8)

    def centering(self, input):
        n = input.size()[-1]
        unit = torch.ones(size=(n,n)).to('cuda')
        I = torch.eye(n).to('cuda')
        H = I - unit / n
        return torch.matmul(torch.matmul(H, input), H)
    
    def linear_HSIC(self, x, y):
        if x.dim() >= 3 and y.dim() >= 3:
            l_x = torch.matmul(x, x.transpose(-2, -1)) # (32,128,128)
            l_y = torch.matmul(y, y.transpose(-2, -1)) # (8,128,128)
            return torch.sum(torch.sum(torch.mul(self.centering(l_x), self.centering(l_y)), dim=-1), dim=-1) # torch.mul:element-wise mul
        else:
            l_x = torch.matmul(x, x.transpose(-2, -1))
            l_y = torch.matmul(y, y.transpose(-2, -1))
            return torch.sum(torch.mul(self.centering(l_x), self.centering(l_y)))
        
if __name__ == "__main__":
    num_centroid = 8
    max_seq_length = 128
    hidden_size = 768
    tr_example_size = 320
    te_example_size = 64
    batch_size = 32
    learning_rate = 1e-5
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = loss_Memory(num_centroid, hidden_size, max_seq_length).to(device)
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    tr_data = tr_cent_data[:, :-1]  # (320,98304)
    tr_target = tr_cent_data[:, -1]  # (320)

    model.train()
    model.centroid.weight.requires_grad = True
    total_loss = 0
    correct = 0
    for epoch in range(epochs):  # unsupervised learning 이니까 아마도 에폭이 의미가 없는것 같다
        # For silhouette score
        tr_out = torch.ones(1).unsqueeze(0).to(device)
        for b in range(0, len(tr_data), batch_size):
            tr_batch = tr_data[b:b + batch_size, :].to(device)  # tensor(32,98305)
            tr_batch_trg = tr_target[b:b + batch_size].to(device)

            # optimizer.zero_grad()
            cka_loss, cka_loss_matrix, centroid, result = model(tr_batch) # loss(0.1553) loss_mat(32,8) cent(8,98304) res(32)
            tr_out = torch.cat((tr_out, result.unsqueeze(1).float()), 0)
            
            # 잘못 생각한 cross entropy 백프롭
            # cross_loss = criterion(result.view(-1, result.size(-1)), tr_batch_trg)
            # cross_loss.backward()

            # cka_loss는 grad_fn이 없어서 optimizer와 backprop() 식으로는 안된다
            cka_loss.backward()
            optimizer.step()
            total_loss += cka_loss.item()

            interval = 500  # batch가 100개
            # print(i)
            if b % interval == 0 and b > 0:
                # print(b)
                avg_loss = total_loss / interval
                ppl = math.exp(avg_loss)

                print("epoch: %d | b: %d | loss: %.3f | ppl: %.3f" % (epoch + 1, b, avg_loss, ppl))
                total_loss = 0

        correct += (tr_out[1:, :].squeeze().cpu() == tr_target).sum().item()
        print("Accuracy of the Cluster in training: %d epochs| %.3f %%" % (epoch, 100 * correct / tr_example_size))
        correct = 0

        # Get Silhouette Score
        tr_out_arr = tr_out[1:, :].squeeze().cpu().detach().numpy()
        tr_data_arr = tr_data.numpy()
        s_score = metrics.silhouette_score(tr_data_arr, tr_out_arr, metric='euclidean')
        print("train Silhouette score: %.3f" % (s_score))
        

    model.eval()
    te_data = te_cent_data[:, :-1]  # (320,98304)
    te_target = te_cent_data[:, -1]  # (320)
    total_loss = 0
    te_out = torch.ones(1).unsqueeze(0).to(device)
    correct = 0
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
                ppl = math.exp(avg_loss)
                print("b: %d | loss: %.3f | ppl: %.3f" % (b, avg_loss, ppl))
                total_loss = 0

        # Get Silhouette Score
        te_out_arr = te_out[1:, :].squeeze().cpu().detach().numpy()
        te_data_arr = te_data.numpy()
        s_score = metrics.silhouette_score(te_data_arr, te_out_arr, metric='euclidean')
        print("train Silhouette score: %.3f" % (s_score))

        # mapping predict and real
        correct += (te_out[1:, :].squeeze().cpu() == te_target).sum().item()
        print("Accuracy of the Cluster in testing: %.3f %%" % (100 * correct / te_example_size))
