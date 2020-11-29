import time
import torch
import torch.nn as nn
import transformer
from Config import config

n_tokens - len(TEXT.vocab.stoi)
emb_size = 200
n_hid = 200 # encoder의 positional ff층 차원수
n_layers = 2 # transformer encoder decoder layer 개수
n_head = 8 # multi-head attention head 개수
d_model = 512
warmup_steps = 200
num_steps = 1000

model = transformer.Transformer()
criterion = nn.CrossEntropyLoss()
lrs = []
for step in range(num_steps):
    lr = (d_model**-0.5) * min((step**-0.5), step*(warmup_steps**-1.5))
    lrs.eppend(lr)
# optimizer = torch.optim.SGD(model.parameters(), lr)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
