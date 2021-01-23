import pdb
import torch
from torch import nn
# from metrics.LayerWiseMetrics import cdist2, linear_CKA_loss

import random
from Dataset import tr_cent_data, te_cent_data

class Memory(nn.Module):
    def __init__(self, num_centroid, hidden_size, max_seq_length):
        super().__init__()
        self.num_centroid = num_centroid
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        #self.device = config.device
        self.centroid = nn.Embedding.from_pretrained(torch.normal(-0.01, 0.2, size=(self.num_centroid,
                                          self.max_seq_length *
                                          self.hidden_size)))
        #self.length = config.length
        #self.lr = config.learning_rate
        
    def forward(self, input):
        #new_centroid = self.centroid.clone().detach()
        #pdb.set_trace()
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
        #print("CKA loss : {}".format(cka_loss.item()))
        #print("CKA loss matrix \n")
        #print(cka_loss_matrix)
        #print("CKA idx \n")
        #print(idx)
        #print(cka_loss, cka_loss_matrix, self.centroid)
        return (cka_loss, cka_loss_matrix, self.centroid), idx
    
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

    model = Memory(num_centroid, hidden_size, max_seq_length).to(device)
    tr_data = tr_cent_data[:, :-1]  # (320,98304)
    tr_target = tr_cent_data[:, -1]  # (320)

    model.train()
    model.centroid.weight.requires_grad = True
    correct = 0
    for epoch in range(epochs):  # unsupervised learning 이니까 아마도 에폭이 의미가 없는것 같다
        for b in range(0, len(tr_data), batch_size):
            batch = tr_data[b:b + batch_size, :].to(device)  # tensor(32,98305)
            batch_trg = tr_target[b:b + batch_size].to(device)
            output, result = model(batch)

            correct += (result == batch_trg).sum().item()

        print("Accuracy of the Cluster in training: %d epochs| %.3f %%" % (
        epoch, 100 * correct / tr_example_size))
        correct = 0

    model.eval()
    te_data = te_cent_data[:, :-1]  # (320,98304)
    te_target = te_cent_data[:, -1]  # (320)
    correct = 0
    with torch.no_grad():
        model.centroid.weight.requires_grad = False
        for b in range(0, len(te_data), batch_size):
            batch = te_data[b:b + batch_size, :].to(device)
            batch_trg = te_target[b:b + batch_size].to(device)
            output, pred = model(batch)

            correct += (result == batch_trg).sum().item()
        print("Accuracy of the Cluster in testing: %.3f %%" % (100 * correct / tr_example_size))
