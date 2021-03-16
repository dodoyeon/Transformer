import os
import torch
import torch.nn as nn
import torchtext
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import gzip

# from process_Samlynn import batch_size_fn, read_data, create_dataset, create_fields
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######Samlynn_train dataset preprocessing#####
# max_seq_len = 200
# batch_size = 1500
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Loading Data on variables
# src_data = os.path.abspath("data/Samlynn_data/english.txt")
# trg_data = os.path.abspath("data/Samlynn_data/french.txt")
# src_data, trg_data = read_data(src_data, trg_data)
# SRC, TRG = create_fields(src_lang='en', trg_lang='fr')
#
# train_data, trian_len, src_pad, trg_pad = create_dataset(src_data, trg_data, batch_size, device, max_seq_len, SRC, TRG)  # = train_iter(in Process.py)
# # train, test = train_test_split(data, test_size = 0.2)
# src_n_tokens = len(SRC.vocab)
# trg_n_tokens = len(TRG.vocab)

######Europarl-v7.fr-en dataset preprocessing######
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

####pytorch tutorial: torchtext translation reference####
batch_size = 32

SRC = Field(tokenize="spacy",
            tokenizer_language="en",
            lower= True)
TRG = Field(tokenize="spacy",
            tokenizer_language="de",
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data, vaild_data, test_data = Multi30k.splits(exts = ('.en', '.de'),
                                                    fields = (SRC, TRG))
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

train_iterator, vaild_iterator, test_iterator = BucketIterator.splits(
    (train_data, vaild_data, test_data),
    batch_size= batch_size,
    device = device
)

#### pytorch tutorial: nn.Transformer torchtext seq2seq modeling reference####
# from torchtext.datasets import WikiText2
# from torchtext.data.utils import get_tokenizer
# 
# TEXT = Field(tokenize=get_tokenizer("basic_english"),
#              init_token='<sos>',
#              eos_token='<eos>',
#              lower = True)
# train_txt, val_txt, test_txt = WikiText2.splits(TEXT)
# TEXT.build_vocab(train_txt)
# 
# def batchify(data, bsz):
#     data = TEXT.numericalize([data.examples[0].text])
#     # numericalize = 번호순으로 정렬하다
#     # Divide the dataset into bsz parts.
#     nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     return data.to(device)
# 
# batchsize = 32
# eval_batchsize= 32
# train_data = batchify(train_txt, batchsize)
# val_data = batchify(val_txt, eval_batchsize)
# test_data = batchify(test_txt, eval_batchsize)

######## SOM_clustering_test_hj ########
# 랜덤 disatibution 으로 데이터를 제작
class cluster_test_dataset(nn.Module):
    def __init__(self, num_centroid, max_seq_length, hidden_size, tr_example_size, te_example_size, batch_size):
        super(cluster_test_dataset, self).__init__()
        # self.centroid = centroid
        self.num_centroid = num_centroid
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.tr_example_size = tr_example_size
        self.te_example_size = te_example_size
        self.batch_size = batch_size
        
    def forward(self, centroid): # centroid(8,98304)
        tr_datas = torch.ones(self.max_seq_length * self.hidden_size).unsqueeze(0).to(device)  # (1,98704)
        tr_trg = torch.ones(1).unsqueeze(0).to(device) # (1,1)
        for i in range(centroid.size(0)):
            data = torch.normal(centroid[i, :], 0.1*torch.ones(self.tr_example_size//centroid.size(0), self.max_seq_length * self.hidden_size).to(device)) # tensor(800,98304)
            tr_datas = torch.cat((tr_datas, data), dim=0) # (801,98304)
            tr_trg = torch.cat((tr_trg, i * torch.ones(self.tr_example_size//centroid.size(0)).to(device).unsqueeze(1))) # (801,1)

        tr_cent_data = torch.cat((tr_datas, tr_trg), dim=1)
        tr_cent_data = tr_cent_data[1:,:] # (6400,98305)
        tr_cent_data = tr_cent_data[torch.randperm(tr_cent_data.size(0)), :]
        
        # with gzip.open('tr_data.pickle','wb') as f: # torch.save
        #     pickle.dump(tr_cent_data.cpu(), f)

        te_datas = torch.ones(self.max_seq_length * self.hidden_size).unsqueeze(0).to(device)  # tensor(98304)
        te_trg = torch.ones(1).unsqueeze(0).to(device)
        for j in range(centroid.size(0)):
            data = torch.normal(centroid[j, :],
                                0.1*torch.ones(self.te_example_size // centroid.size(0), self.max_seq_length * self.hidden_size).to(device)) # tensor(98304)
            te_datas = torch.cat((te_datas, data))
            te_trg = torch.cat(
                (te_trg, j * torch.ones(self.te_example_size // centroid.size(0)).to(device).unsqueeze(1)))

        te_cent_data = torch.cat((te_datas, te_trg), dim=1)
        te_cent_data = te_cent_data[1:, :]
        te_cent_data = te_cent_data[torch.randperm(te_cent_data.size(0)), :]

        # with gzip.open('te_data.pickle','wb') as f:
        #     pickle.dump(te_cent_data.cpu(), f)
        return tr_cent_data, te_cent_data
# print("training dataset over")

# print("test dataset over")

    # def three_D_visualization(self, input):
    #     color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown']
    #     # fig = plt.figure()
    #     # ax = fig.gca(projection='3d')
    #     input = input[:, :-1]
    #     input_trg = input[:, -1]
    #     input = input.view(-1, 128, 768)
    # 
    #     centroid = self.centroid.view(-1, 128, 768)
    #     for i in range(centroid.size(0)):  # 8
    #         center = centroid[i, :, :]
    #         plt.scatter(center.size()[0], center.size()[1], c=color[i], marker='x')
    # 
    #     for j in range(input.size(0)):
    #         datum = input[j:, :]
    #         plt.scatter(datum.size()[0], datum.size()[1], c=color[input_trg[j]], maker='o')
    # 
    #     plt.set_xlabel('max seg length')
    #     plt.set_ylabel('hidden state')
    #     plt.show()
    
# Dataset: BERT feaure hj 
# from torch.utils.data import Dataset, DataLoader
# class Bert_feature_dataset(Dataset): # Dataset 상속 해야하는가?
#     def __init__(self, file_dir):
#         self.data_dir = file_dir
#
#     def __len__(self):
#         return 8544
#
#     def __getitem__(self, idx):
#         file_dir = os.path.join(
#             self.data_dir + '/cola_dataset/'+'cola_data'+ str(idx+1) + '.pt'
#         )
#         data = torch.load(file_dir)
#         return data
#
# # .pt 32x128x768 텐서파일을 열어서 1x128x768로 다시 저장하는 함수
# def tr():
#     for idx in range(267):
#         file_name = 'BERT_feature/cola_tensors' + '/cola' + str(idx + 1) + '.pt'
#         s = torch.load(file_name)
#         for i in range(s.size(0)):
#             f = s[i,:,:]
#             load_name = 'cola_dataset/'+'cola_data'+ str((idx * 32) + i) + '.pt'
#             torch.save(f, load_name)
# tr()

# # 멀티 지피유(노드4개)를 다 써서 데이터셋 만들고 싶다
# # -> AssertionError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
# import os
# import sys
# import tempfile
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
#
# def setup(rank, world_size):
#     if sys.platform == 'win32': # 윈도우 32bit 일때
#         init_method="file:///{your local file path}"
#
#         # initialize the process group
#         dist.init_process_group(
#             "gloo", # 고정
#             init_method=init_method,
#             rank=rank, # 디바이스 넘버 ex) cuda:0 같은
#             world_size=world_size # mp(멀티 프로세스)를 몇개 생성할 것인가(4,8 등)
#         )
#
#     else: # 리눅스 등의 os는 환경변수를 사용
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '12355'
#
#         # initialize the process group
#         dist.init_process_group("gloo", rank=rank, world_size=world_size)
#
# # 멀티 프로세스를 종료하는 역할
# def cleanup():
#     dist.destroy_process_group()
#
# def demo_basic(rank, world_size):
#     num_centroid = 8
#     max_seq_length = 128
#     hidden_size = 768
#     tr_example_size = 6400
#     te_example_size = 640
#     batch_size = 32
#
#     print(f"Running basic DDP example on rank {rank}.")
#     setup(rank, world_size)
#     tr_centroid = torch.distributions.Uniform(-1, +1).sample((num_centroid, max_seq_length * hidden_size)).to(rank)
#
#     # create model and move it to GPU with id rank
#     model = cluster_test_dataset(num_centroid, max_seq_length, hidden_size, tr_example_size, te_example_size, batch_size).to(rank)
#     ddp_model = DDP(model, device_ids=[rank])
#
#     outputs = ddp_model(tr_centroid)
#
#     cleanup()
#     return outputs
#
# def run_demo(demo_fn, world_size):
#     mp.spawn(demo_fn,
#              args=(world_size,),
#              nprocs=world_size,
#              join=True)


# # if __name__ == "__main__": # dataset.py를 직접 실행 시키겠다는 뜻 OTL

num_centroid = 8
max_seq_length = 128
hidden_size = 768
# device = 'cuda'
# std = np.sqrt(0.2)
tr_example_size = 6400
te_example_size = 640
batch_size = 32

tr_centroid = torch.distributions.Uniform(0, +1).sample((num_centroid, max_seq_length * hidden_size)).to(device)
# tr_centroid = torch.FloatTensor(num_centroid, max_seq_length * hidden_size).uniform_(-1,1).to(device) # 여기까지는 문제 없음

# with gzip.open('tr_centroid.pickle', 'wb') as f:
#     pickle.dump(tr_centroid.cpu(), f)
# torch.save(tr_centroid.cpu(),'tr_cent')

model = cluster_test_dataset(num_centroid, max_seq_length, hidden_size, tr_example_size, te_example_size, batch_size)# .to(device)
tr_cent_data, te_cent_data = model(tr_centroid)

print("Data making over")

# multi gpu main 문
# if __name__ == "__main__":
#     n_gpus = torch.cuda.device_count()
#     print(f"{n_gpus}!")
#     run_demo(demo_basic, 4)