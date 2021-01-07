import os
import torch
import torchtext
from sklearn.model_selection import train_test_split
# from process_Samlynn import batch_size_fn, read_data, create_dataset, create_fields
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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