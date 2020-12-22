import time
import math
import torch
import torch.nn as nn
from Dataset import *
from mask import create_padding_mask, create_attn_decoder_mask
from transformer import Transformer

# warmup_steps = 4000 # 논문 설정
# num_steps = 1000

# lrs = [] # 논문에 나와있는 learning rate 식 구현-> 어떻게 쓸지는 좀 더 알아보자
# for step in range(num_steps): #
#     lr = (d_model**-0.5) * min((step**-0.5), step*(warmup_steps**-1.5))
#     lrs.append(lr)

def train(model, train_iterator, optimizer, criterion, epochs):
    model.train()
    pad_idx = 0
    start_time = time.time()
    train_len = len(train_iterator) # 227

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_iterator):
            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)

            trg_input = trg.clone().detach()
            trg_input[trg_input == 3] = 1 # if trg_input.data = 3 ->convert 1(<pad>)
            trg_input = trg_input[:,:-1]
            # trg_input = torch.empty((trg.size(0),trg.size(1)-1))
            # for j in range(trg.size(0)): # row
            #     trg_row = trg[j,:]
            #     for k in range(trg.size(1)): # col
            #         if trg_row[k] == 3: # <eos>=3
            #             trg_k = torch.cat((trg_row[:k],trg_row[k+1:]))
            #             trg_input.cat(trg_k, dim=1)

            ys = trg[:,1:].contiguous().view(-1)

            enc_pad_mask = create_padding_mask(src, src, pad_idx).to(device)

            self_pad_mask = create_padding_mask(trg_input, trg_input, pad_idx).to(device)
            self_attn_mask = create_attn_decoder_mask(trg_input).to(device)
            self_dec_mask = torch.gt((self_pad_mask + self_attn_mask),0).to(device)
            enc_dec_pad_mask = create_padding_mask(trg_input, src, pad_idx).to(device)
            
            # enc_pad_mask = create_padding_mask(enc_input, enc_input, self.pad_idx)
            # 
            # self_pad_mask = create_padding_mask(dec_input, dec_input, self.pad_idx)
            # self_attn_mask = create_attn_decoder_mask(dec_input)
            # self_dec_mask = torch.gt((self_pad_mask + self_attn_mask), 0)  # 첫번째 인풋에 broadcastable한 두번째 아규먼트사용, input>2nd이면 true
            # enc_dec_pad_mask = create_padding_mask(dec_input, enc_input, self.pad_idx)  # ERROR POINT

            optimizer.zero_grad()
            pred, _, _, _ = model(src, trg_input, enc_pad_mask, self_dec_mask, enc_dec_pad_mask)
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

        torch.save({'epoch': epoch,
                    'model state_dist': model.state_dict(),# dict!!!
                    'optimizer state_dict': optimizer.state_dict(),
                    'loss': avg_loss}, 'weight/train_weight.pth') # new file'train_weight.pth' if not exists

        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % (
        (time.time() - start_time) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100, avg_loss,
        epoch + 1, avg_loss))


def evaluate(model, test_iterator, criterion, max_seq_len):
    checkpoint = torch.load('weight/train_weight.pth')
    model.load_state_dict(checkpoint['model state_dist']) # dict!!
    # optimizer.load_state_dict(checkpoint['optimizer state_dict'])#->optimizer aren't used in test?

    model.eval()
    total_loss = 0
    test_len = len(test_iterator) # test_len=8(-> the number of batch?/not change)
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            src = batch.src.transpose(0,1) # (128,12) -> batch_size = 128
            trg = batch.trg.transpose(0,1) # (128,11)

            ys = trg[:, 1:].contiguous().view(-1) # size:(128x10=1280)
            test_input = torch.ones(src.size(0), max_seq_len) # (1,11)
            test_input[:,0] = 2
            
            # for j in range(src.size(0)):
            for k in range(max_seq_len):
                pred, _, _, _ = model(src, test_input.to(device).long()) # consider get model's mask function out
                test_input = pred
                
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
    lr = 0.0001

    model = Transformer(src_voca_size=n_src_tokens, trg_voca_size=n_trg_tokens, emb_dim=emb_size, d_ff=n_hid,
                        n_layers=n_layers, n_head=n_head).to(device)
    criterion = nn.CrossEntropyLoss() # optimizer,loss->to(device) x
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=lr)

    # print("model's state_dict:")  # 모델의 학습가능한 parameter는 model.parameters()로 접근한다
    # # state_dict: 각 레이어를 파라미터 tensor와 매핑하는 PYTHON dict objects/ cnn, linear 등이나 registered buffer(batchnorm)등 저장
    # # optimizer도 state와 hyper parameter의 state_dict 가짐
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # 
    # print("optimier's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    # train
    print("start training..")
    train(model, train_iterator, optimizer, criterion, epochs)
    # test
    print("start testing..")
    evaluate(model, test_iterator, criterion, max_seq_len)

if __name__ == '__main__':
    main()
