import time
import math
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from transformer import Transformer

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"), init_token='<sos>', eos_token='<eos>', lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    nbatch = data.size(0) // bsz
    data = data.narrow(0,0,nbatch*bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

bptt = 35 # 이게 뭐야?
def get_batch(source, i):
    seq_len = min(bptt, len(source)-1-i)
    data = source[i:i+seq_len]
    target = source[i+1,i+1+seq_len].view(-1)
    return data, target


n_tokens = len(TEXT.vocab.stoi) # vocabulary dictionary size
emb_size = 200 # emb_dim
n_hid = 200 # encoder의 positional ff층 차원수
n_layers = 2 # transformer encoder decoder layer 개수
n_head = 8 # multi-head attention head 개수
d_model = 512
# warmup_steps = 200
# num_steps = 1000
epochs = 3

model = Transformer(voca_size= n_tokens, emb_dim= emb_size, d_ff= n_hid, n_layers= n_layers, n_head= n_head).to(device)
criterion = nn.CrossEntropyLoss()
# lrs = [] # 논문에 나와있는 learning rate 식 구현-> 어떻게 쓸지는 좀 더 알아보자
# for step in range(num_steps): #
#     lr = (d_model**-0.5) * min((step**-0.5), step*(warmup_steps**-1.5))
#     lrs.append(lr)
# optimizer = torch.optim.SGD(model.parameters(), lr)
lr = 5.0
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    # n_tokens = len(TEXT.vocab.stoi)

    for batch, i in enumerate(range(0, train_data.size(0)-1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, n_tokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200

        if batch% log_interval == 0 and batch>0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch: {:3d} | {:5d}/{:5d} batches |'
                  'lr = {:02.2f} | ms/batch {:5.2f} |'
                  'loss = {:5.2f} | ppl = {:8.2f}'.format(
                epoch, batch, len(train.data) // bptt, scheduler.get_lr()[0],
                elapsed*1000 / log_interval,
                cur_loss, math.exp(cur_loss)
            ))

            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    n_tokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0)-1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, n_tokens)
            total_loss += len(data)*criterion(output_flat, targets).item()
        return total_loss/(len(data_source)-1)

# 이게 train?
best_val_loss = float("inf")
best_model = None # validation loss가 지금까지 관찰한 모델중 최적일때, 저장

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

