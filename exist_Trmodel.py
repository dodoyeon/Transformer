import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data.metrics
from Dataset import *
from mask import create_padding_mask, create_attn_decoder_mask
from Attention import PositionalEncoding

class Trmodel(nn.Module):
    def __init__(self, src_vocab, trg_vocab, emb_dim=512, max_seq_len=200, pos_dropout=0.1):
        super().__init__()
        src_voca_size = len(src_vocab.stoi)
        trg_voca_size = len(trg_vocab.stoi)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_emb = nn.Embedding(src_voca_size, emb_dim)
        self.trg_emb = nn.Embedding(trg_voca_size, emb_dim)
        self.pos_emb = PositionalEncoding(emb_dim, max_seq_len, pos_dropout)
        self.Tr = nn.Transformer()
        self.proj_vocab_layer = nn.Linear(2048, trg_voca_size)  # d_ff=2048

    def forward(self, src, trg):
        enc_input = self.src_emb(src)
        enc_input = self.pos_emb(enc_input)
        dec_input = self.trg_emb(trg)
        dec_input = self.pos_emb(dec_input)

        # masking
        src_pad_mask = enc_input == self.src_vocab.stoi["<pad>"]  # to get value in the dict!!
        trg_pad_mask = dec_input == self.trg_vocab.stoi["<pad>"]
        trg_dec_mask = self.Tr.generate_square_subsequent_mask(dec_input.size(1))

        # enc_input = torch.einsum('ijk->jik', enc_in)# index i,j,k->j,i,k(change the order)
        # dec_input = torch.einsum('ijk->jik', dec_in)
        # torch.einsum(equation,operand): 

        pred = self.Tr(enc_input, dec_input, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
                       memory_key_padding_mask=src_pad_mask, tgt_mask=trg_dec_mask)

        pred_fin = self.proj_vocab_layer(pred)

        return pred_fin