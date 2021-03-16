import matplotlib.pyplot as plt
from Dataset import Bert_feature_dataset
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn

dir = os.path.abspath(".")
dataset = Bert_feature_dataset(dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

num_centroid = 8
max_seq_len = 128
hidden_size = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################FAILED#####################
class Memory(nn.Module):
    def __init__(self, num_centroid, hidden_size, max_seq_len):
        super().__init__()
        self.num_centroid = num_centroid
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        #self.centroid = nn.Embedding.from_pretrained(torch.normal(0.0, 0.39, size=(self.num_centroid,
        #                                  self.max_seq_length *
        #                                  self.hidden_size)))
        self.centroid = torch.normal(-0.01, 0.39, size=(self.num_centroid,
                                                        self.hidden_size * self.max_seq_len),
                                     requires_grad=True).to('cuda')
        self.idx_dict = {}
        self.count = {}
        for i in range(self.num_centroid):
            self.idx_dict[i] = []
            self.count[i] = 0

teacher_memory = Memory(num_centroid, hidden_size, max_seq_len)
teacher_centroid = teacher_memory.centroid.clone().detach()
#centroid 0으로 초기화 왜냐하면 0으로 초기화 하지 않으면 처음 업데이트 시 원래 값이 같이 계산되므로 초기화 함.
teacher_memory.centroid = torch.zeros_like(teacher_memory.centroid).to(device)
# iteration : epoch

k_max = 20
sil = []
k_num = []
train_iterator = 2

cl_out = torch.ones(1) #.to(device)
cl_data = torch.ones(1, 128*768) #.to(device) # start_time = time.time()
for iteration in range(train_iterator): # train_iterator?
    #step은 example index와 동일
    for step, teacher_feature in enumerate(dataloader):
        #teacher_feature = example: [1, 128 * 768]
        teacher_feature = teacher_feature.view(-1, 128*768).to(device)
        cl_data = torch.cat((cl_data, teacher_feature.cpu()), 0)
        sample_var, sample_mean = 0.0, 0.0
        # iteration : epoch : 첫번째 epoch에서는 mean,var만 구하는 단계 업데이트 X
        if iteration == 0:
            print("0: mean,var만 구하는 단계")
            #step : iteration
            #smaple_var, sample_mean : 매 iter 마다 새롭게 업데이트 그냥 신경 안쓰고 사용하면 됨
            sample_var = (((step + 1) * torch.var(teacher_feature, dim=-1) + sample_var) / (step + 2)).to(device)
            sample_mean = (((step + 1) * torch.mean(teacher_feature, dim=-1) + sample_mean) / (step + 2)).to(device)
            #관측된 dataset example에 대해 새롭게 centroid 초기화
            teacher_centroid = torch.normal(sample_mean.item(), torch.sqrt(sample_var).item(),
                                            size=(teacher_centroid.size()[0],
                                                  teacher_centroid.size()[1])).to(device)
        #두 번째 epoch에서는 centroid에 nearest neighbor example index 저장
        elif iteration % 2 == 1 :
            print("%d: centroid example index 저장", iteration)
            #dist: example과 centroid 간의 거리 계산
            dist = torch.cdist(teacher_feature, teacher_centroid)
            # centroid_idx : dist를 통해 가장 가까운 centroid index
            centroid_idx = torch.argmin(dist, dim=-1)
            # Memory module self.idx_dict에 idx_dict[centroid_dix] = i_th example 번호 저장
            for i, idx in enumerate(centroid_idx):
                cl_out = torch.cat((cl_out, idx.cpu().unsqueeze(0).float()), 0)
                teacher_memory.idx_dict[int(idx.item())].append(i + step)
        #세번째 epoch에서 centroid 업데이트
        elif iteration % 2 == 0 and iteration != 0 :
            print("%d: centroid 업데이트", iteration)
            #i : 0 ~ centroid 갯수-1
            for i in range(teacher_centroid.size()[0]):
                #step : example idx 순서와 동일
                #Memory module idx_dict에 해당 example idx가 존재하면 centroid 바로바로 업데이트
                if step in teacher_memory.idx_dict[i]:
                    #pdb.set_trace()
                    teacher_memory.centroid[i] = ((teacher_memory.count[i] + 1)
                                                      * teacher_memory.centroid[i]
                                                      + teacher_feature) / (teacher_memory.count[i] + 2)
                    #centroid 평균을 위해 매 iter마다 선택된 centroid_idx에 해당하는 key에 숫자 카운팅
                    teacher_memory.count[i] += 1
                    #계산된 example id 제거
                    teacher_memory.idx_dict[i].remove(step)
            teacher_centroid = teacher_memory.centroid
            # 모든 카운트 초기화
        for i in range(teacher_centroid.size()[0]):
            teacher_memory.count[i] = 0

# teacher_memory.idx_dict.values()
cl_out_arr = cl_out[1:].squeeze().detach().numpy()
cl_data_arr = cl_data[1:, :].numpy()
s_score = metrics.silhouette_score(cl_data_arr, cl_out_arr, metric='euclidean')
print("k: %d Silhouette score: %.3f" % (k, s_score))