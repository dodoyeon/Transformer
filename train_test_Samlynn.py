import time
import torch
import torch.nn as nn
from transformer import Transformer

n_tokens = len(TEXT.vocab.stoi)  # vocabulary dictionary size
emb_size = 200  # emb_dim
n_hid = 200  # encoder의 positional ff층 차원수
n_layers = 2  # transformer encoder decoder layer 개수
n_head = 8  # multi-head attention head 개수
d_model = 512
model = Transformer(voca_size=n_tokens, emb_dim=emb_size, d_ff=n_hid, n_layers=n_layers, n_head=n_head).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=lr)

def train_model(model):
    print("training model...")
    model.train()
    start = time.time()
    epochs = 2

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(opt.train):
            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)
            trg_input = trg[:, :-1]
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            optimizer.zero_grad()
            loss = criterion(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

if __name__ == "__main__":
    main()