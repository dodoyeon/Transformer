import time
import math
import torch
import torch.nn as nn
<<<<<<< HEAD
from Dataset import *
from transformer import Transformer

# warmup_steps = 4000 # 논문 설정
# num_steps = 1000

=======
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
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


n_tokens = len(TEXT.vocab.stoi) # vocabulary dictionary size
emb_size = 200 # emb_dim
n_hid = 200 # encoder의 positional ff층 차원수
n_layers = 2 # transformer encoder decoder layer 개수
n_head = 8 # multi-head attention head 개수
d_model = 512
# warmup_steps = 4000 # 논문 설정
# num_steps = 1000
epochs = 3

model = Transformer(voca_size= n_tokens, emb_dim= emb_size, d_ff= n_hid, n_layers= n_layers, n_head= n_head).to(device)
criterion = nn.CrossEntropyLoss()
>>>>>>> 21d308b3c97a14ff26fea12bf0264dfbd336a3f4
# lrs = [] # 논문에 나와있는 learning rate 식 구현-> 어떻게 쓸지는 좀 더 알아보자
# for step in range(num_steps): #
#     lr = (d_model**-0.5) * min((step**-0.5), step*(warmup_steps**-1.5))
#     lrs.append(lr)
<<<<<<< HEAD

def train(model, train_iterator, optimizer, criterion, epochs):
    model.train()
    total_loss = 0.
    start_time = time.time()
    train_len = len(train_iterator)

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_iterator):
            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)
            trg_input = trg[:,:-1]
            ys = trg[:,1:].contiguous().view(-1)

            optimizer.zero_grad()
            pred, _, _, _ = model(src, trg_input)
            loss = criterion(pred.view(-1, pred.size(-1)), ys)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) -> RNN
            optimizer.step()

            total_loss += loss.item()

            p = int(100 * (i + 1) / train_len)
            avg_loss = total_loss / batch_size
            print("time= %dm: epoch %d iter %d [%s%s]  %d%%  loss = %.3f" %
                ((time.time() - start_time) // 60, epoch + 1, i + 1, "".join('#' * (p // 5)),
                    "".join(' ' * (20 - (p // 5))),
                    p, avg_loss), end='\r')
            total_loss = 0

    # print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % (
    # (time.time() - start_time) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100, avg_loss,
    # epoch + 1, avg_loss))


def evaluate(model, test_iterator, criterion):
    model.eval()
    total_loss = 0
    test_len = len(test_iterator)
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            src = batch.src
            trg = batch.trg

            trg_input = trg[:, :-1]
            ys = trg[:, 1:].contiguous().view(-1)

            pred, _, _, _ = model(src, trg_input)
            loss = criterion(pred.view(-1, pred.size(-1)), ys)
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                p = int(100 * (i + 1) / test_len)
                avg_loss = total_loss / 100
                print("time= %dm: epoch %d iter %d [%s%s]  %d%%  loss = %.3f" %
                      ((time.time() - start_time) // 60, i + 1, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))),
                       p, avg_loss), end='\r')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_src_tokens = len(SRC.vocab.stoi)  # vocabulary dictionary size
    n_trg_tokens = len(TRG.vocab.stoi)
    epochs = 6
    emb_size = 512  # emb_dim
    n_hid = 200  # encoder의 positional ff층 차원수
    n_layers = 2  # transformer encoder decoder layer 개수
    n_head = 8  # multi-head attention head 개수
    d_model = 512
    max_seq_len = 200
    batch_size = 1500
    lr = 0.0001

    model = Transformer(src_voca_size=n_src_tokens, trg_voca_size=n_trg_tokens, emb_dim=emb_size, d_ff=n_hid,
                        n_layers=n_layers, n_head=n_head).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=lr)

    print("start training..")
    train(model, train_iterator, optimizer, criterion, epochs)
    print("start testing..")
    evaluate(model, test_iterator, criterion)

if __name__ == '__main__':
    main()
=======
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

>>>>>>> 21d308b3c97a14ff26fea12bf0264dfbd336a3f4
