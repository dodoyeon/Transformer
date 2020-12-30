import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data.metrics
from Dataset import *
from mask import create_padding_mask, create_attn_decoder_mask
from Attention import PositionalEncoding
from exist_Trmodel import Trmodel


def train(model, train_iterator, optimizer, criterion, epochs):
    model.train()
    for p in model.parameters():
        if p.dim() > 1:  # why?
            nn.init.xavier_uniform_(p)
    pad_idx = 0
    start_time = time.time()
    train_len = len(train_iterator)  # 227

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_iterator):
            src = batch.src
            trg = batch.trg
            
            trg_input = trg.clone().detach()
            trg_input[trg_input == 3] = 1  # if trg_input.data = 3 ->convert 1(<pad>)
            trg_input = trg_input[:-1, :]

            ys = trg[1:, :].contiguous().view(-1)
            
            optimizer.zero_grad()
            pred = model(src, trg_input)
            loss = criterion(pred.view(-1, pred.size(-1)), ys)
            loss.backward()
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
                    'model state_dict': model.state_dict(),
                    'optimizer state_dict': optimizer.state_dict(),
                    'loss': avg_loss}, 'weight/Tr_train_weight.pth')  # new file'train_weight.pth' if not exists

        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % (
            (time.time() - start_time) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))),
            100, avg_loss,epoch + 1, avg_loss))


def evaluate(model, test_iterator, max_seq_len):
    checkpoint = torch.load('weight/Tr_train_weight.pth')
    model.load_state_dict(checkpoint['model state_dict'])

    model.eval()
    total_loss = 0
    test_len = len(test_iterator)

    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            src = batch.src
            trg = batch.trg
            trg_mask = torch.ones(trg.size(0), 201 - trg.size(1)).long().to(device)  # (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            trg_test = torch.cat((trg, trg_mask), dim=1)
            ys = trg_test[:, 1:].contiguous().view(-1)  # size:(128x10=1280)
            dec_input = 2 * torch.ones(src.size(0), 1).long().to(device)  # (batch=32,1)

            for k in range(max_seq_len):
                # https://github.com/eagle705/pytorch-transformer-chatbot/blob/master/inference.py
                pred = model(src, dec_input)  # we have to shape pred(32,1,512)->but no..
                prediction = pred[:, -1, :].unsqueeze(1)
                pred_ids = prediction.max(dim=-1)[1]  # dim=-1(=512): [1]=index (32,1)->argmax?

                dec_input = torch.cat([dec_input, pred_ids.long()], dim=1)  # dec_input.unsqueeze(1),/pred_ids[0,-1].unsqueeze(0).unsqueeze(0)
            loss = F.cross_entropy(pred.view(-1, pred.size(-1)), ys)
            total_loss += loss.item()

    avg_loss = total_loss / test_len
    ppl = math.exp(avg_loss)

    print("loss = %.3f  perplexity = %.3f" % (avg_loss, ppl))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab = SRC.vocab # vocabulary dictionary size
    trg_vocab = TRG.vocab
    epochs = 6
    max_seq_len = 200
    lr = 0.0001

    model = Trmodel(src_vocab=src_vocab, trg_vocab=trg_vocab).to(device)
    criterion = nn.CrossEntropyLoss()  # optimizer,loss->to(device) x
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=lr)

    # train
    print("start training..")
    train(model, train_iterator, optimizer, criterion, epochs)
    # test
    print("start testing..")
    evaluate(model, test_iterator, max_seq_len)


if __name__ == '__main__':
    main()