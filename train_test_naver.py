import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import sentencepiece as spm
import json
from transformer import Transformer
########### "␝" 에러#############
# text = open("./web-crawler/kowiki/kowiki_20201202.csv", "r",encoding='UTF-8')
# text = ''.join([i for i in text]).replace("␝", ",")
# x = open("./web-crawler/kowiki/kowiki_20201202_re.csv","w",encoding='UTF-8')
# x.writelines(text)
# x.close()
######################한국어 데이터#########################
import pandas as pd
#
in_file = "./web-crawler/kowiki/kowiki_20201202.csv"
out_file = "./web-crawler/kowiki/kowiki.txt"
SEPARATOR = u"\u241D" # ␝ = U-241D
df = pd.read_csv(in_file, sep=SEPARATOR, engine="python")
with open(out_file, "w") as f:
  for index, row in df.iterrows():
    f.write(row["text"]) # title 과 text를 중복 되므로 text만 저장 함
    f.write("\n\n\n\n") # 구분자

# corpus = "./web-crawler/kowiki/kowiki.txt"
# prefix = "kowiki"
# vocab_size = 8000
# spm.SentencePieceTrainer.train(
#     f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
#     " --model_type=bpe" +
#     " --max_sentence_length=999999" + # 문장 최대 길이
#     " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
#     " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
#     " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
#     " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
#     " --user_defined_symbols=[SEP],[CLS],[MASK]") # 사용자 정의 토큰
##########돌리고나면 다시돌릴필요없으므로 그냥 주석처리(한번만 돌리면 됨)#############

##그다음에 만들어지는 파일: kowiki.model, kowiki.vocab ##
# vocab_file = "<path of vocab>/kowiki.model"
# vocab = spm.SentencePieceProcessor()
# vocab.load(vocab_file)
#
# class MovieClassification(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#
#         self.transformer = Transformer(self.config)
#         self.projection = nn.Linear(self.config.d_hidn, self.config.n_output, bias=False)
#
#     def forward(self, enc_inputs, dec_inputs):
#         # (bs, n_dec_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
#         dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs,
#                                                                                                      dec_inputs)
#         # (bs, d_hidn)
#         dec_outputs, _ = torch.max(dec_outputs, dim=1)
#         # (bs, n_output)
#         logits = self.projection(dec_outputs)
#         # (bs, n_output), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
#         return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
#
#
# class MovieDataSet(torch.utils.data.Dataset):
#     def __init__(self, vocab, infile):
#         self.vocab = vocab
#         self.labels = []
#         self.sentences = []
#
#         line_cnt = 0
#         with open(infile, "r") as f:
#             for line in f:
#                 line_cnt += 1
#
#         with open(infile, "r") as f:
#             for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
#                 data = json.loads(line)
#                 self.labels.append(data["label"])
#                 self.sentences.append([vocab.piece_to_id(p) for p in data["doc"]])
#
#     def __len__(self):
#         assert len(self.labels) == len(self.sentences)
#         return len(self.labels)
#
#     def __getitem__(self, item):
#         return (torch.tensor(self.labels[item]),
#                 torch.tensor(self.sentences[item]),
#                 torch.tensor([self.vocab.piece_to_id("[BOS]")]))
#
# def movie_collate_fn(inputs):
#     labels, enc_inputs, dec_inputs = list(zip(*inputs))
#
#     enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)
#     dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)
#
#     batch = [
#         torch.stack(labels, dim=0),
#         enc_inputs,
#         dec_inputs,
#     ]
#     return batch
#
# batch_size = 128
# train_dataset = MovieDataSet(vocab, "<path of data>/ratings_train.json")
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=movie_collate_fn)
# test_dataset = MovieDataSet(vocab, "<path of data>/ratings_test.json")
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=movie_collate_fn)
#
# def eval_epoch(config, model, data_loader):
#     matchs = []
#     model.eval()
#
#     n_word_total = 0
#     n_correct_total = 0
#     with tqdm(total=len(data_loader), desc=f"Valid") as pbar:
#         for i, value in enumerate(data_loader):
#             labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)
#
#             outputs = model(enc_inputs, dec_inputs)
#             logits = outputs[0]
#             _, indices = logits.max(1)
#
#             match = torch.eq(indices, labels).detach()
#             matchs.extend(match.cpu())
#             accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0
#
#             pbar.update(1)
#             pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
#     return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0
#
# def train_epoch(config, epoch, model, criterion, optimizer, train_loader):
#     losses = []
#     model.train()
#
#     with tqdm(total=len(train_loader), desc=f"Train {epoch}") as pbar:
#         for i, value in enumerate(train_loader):
#             labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)
#
#             optimizer.zero_grad()
#             outputs = model(enc_inputs, dec_inputs)
#             logits = outputs[0]
#
#             loss = criterion(logits, labels)
#             loss_val = loss.item()
#             losses.append(loss_val)
#
#             loss.backward()
#             optimizer.step()
#
#             pbar.update(1)
#             pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
#     return np.mean(losses)
#
# config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# config.n_output = 2
# print(config)
#
# learning_rate = 5e-5
# n_epoch = 10
#
# model = MovieClassification(config)
# model.to(config.device)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# losses, scores = [], []
# for epoch in range(n_epoch):
#     loss = train_epoch(config, epoch, model, criterion, optimizer, train_loader)
#     score = eval_epoch(config, model, test_loader)
#
#     losses.append(loss)
#     scores.append(score)
#
# # table
# data = {
#     "loss": losses,
#     "score": scores
# }
# df = pd.DataFrame(data)
# display(df)
#
# # graph
# plt.figure(figsize=[12, 4])
# plt.plot(losses, label="loss")
# plt.plot(scores, label="score")
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.show()