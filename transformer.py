import torch
import torch.nn as nn
from Attention import Embedding, PositionalEncoding, MultiHeadAttention, PositionalFeedForward
from mask import create_padding_mask, create_attn_decoder_mask

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, n_head, dropout_attn, dropout_multi, d_ff, dropout_ff, layernorm_epsilon):
        super().__init__()
        self.multiheadattn = MultiHeadAttention(emb_dim, n_head, dropout_attn, dropout_multi)
        self.normal_layer1 = nn.LayerNorm(eps= layernorm_epsilon)
        self.normal_layer2 = nn.LayerNorm(eps= layernorm_epsilon)
        self.posffn_layer = PositionalFeedForward(emb_dim, d_ff, dropout_ff)

    def forward(self, enc_input):
        input_q = enc_input
        input_k = enc_input
        input_v = enc_input
        residual = enc_input
        attn_out, attn_prob = self.multiheadattn(input_q, input_k, input_v)
        res_out = residual + attn_out
        out = self.normal_layer1(res_out)

        pos_output = self.posffn_layer(out)
        enc_output = self.normal_layer2(pos_output)

        return enc_output, attn_prob

class Encoder(nn.Module):
    def __init__(self, voca_size, emb_dim, max_seq_len, pos_dropout,dropout_attn, dropout_multi, d_ff, dropout_ff, n_layers, n_head, pad_idx, layernorm_epsilon):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding_layer = Embedding(voca_size, emb_dim)
        self.position_layer = PositionalEncoding(max_seq_len, emb_dim, pos_dropout)
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, n_head, dropout_attn, dropout_multi, d_ff, dropout_ff, layernorm_epsilon) for _ in range(n_layers)])

    def forward(self, enc_input):
        emb_enc = self.embedding_layer(enc_input)
        posemb_enc = self.position_layer(emb_enc)
        enc_output = posemb_enc

        enc_pad_mask = create_padding_mask(enc_input, enc_input, self.pad_idx)

        enc_attn_prob = []

        for layer in self.layers:
            enc_output, attn_prob = layer(enc_output, enc_pad_mask)
            enc_attn_prob.append(attn_prob)
        return enc_output, enc_attn_prob


class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, n_head, dropout_attn, dropout_multi,d_ff, dropout_ff, layernorm_epsilon):
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(emb_dim, n_head, dropout_attn, dropout_multi)
        self.normal_layer1 = nn.LayerNorm(eps = layernorm_epsilon)
        self.masked_attn = MultiHeadAttention(emb_dim, n_head, dropout_attn, dropout_multi)
        self.normal_layer2 = nn.LayerNorm(eps = layernorm_epsilon)
        self.posffn_layer = PositionalFeedForward(emb_dim, d_ff, dropout_ff)
        self.normal_layer3 = nn.LayerNorm(eps=layernorm_epsilon)

    def forward(self, dec_input, enc_output, self_attn_mask, enc_dec_mask):
        input_q = dec_input
        input_k = dec_input
        input_v = dec_input
        residual_1 = dec_input
        attn_out, attn_prob = self.maksed_attn(input_q, input_k, input_v, self_attn_mask)
        res_out1 = residual_1 + attn_out
        masked_output = self.normal_layer1_layer(res_out1)

        residual_2 = masked_output
        mask_out, enc_dec_prob = self.multi_head_attn(masked_output, enc_output, enc_output, enc_dec_mask)
        res_out2 = residual_2 + mask_out
        enc_dec_output = self.normal_layer2(res_out2)

        pos_output = self.posffn_layer(enc_dec_output)
        dec_output = self.normal_layer3(pos_output)

        return dec_output, attn_prob, enc_dec_prob

class Decoder(nn.Module):
    def __init__(self, voca_size, emb_dim, max_seq_len, pos_dropout, dropout_attn, dropout_multi, d_ff, dropout_ff, n_layers, n_head, pad_idx, layernorm_epsilon):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding_layer = Embedding(voca_size, emb_dim)
        self.position_layer = PositionalEncoding(max_seq_len, emb_dim, pos_dropout)
        self.layers = nn.ModuleList([DecoderLayer(emb_dim, n_head, dropout_attn, dropout_multi,d_ff, dropout_ff, layernorm_epsilon) for _ in range(n_layers)])
        self.total_linear = nn.Linear()
        self.softmax = nn.Softmax()

    def forward(self, dec_input, enc_input, enc_output): # memory = seq of output of the last layer of encoder
        emb_dec = self.embedding_layer(dec_input)
        posemb_dec = self.position_layer(emb_dec)
        dec_output = posemb_dec

        self_pad_mask = create_padding_mask(dec_input, dec_input, self.pad_idx)
        self_attn_mask = create_attn_decoder_mask(dec_input)
        self_mask = torch.gt((self_pad_mask+self_attn_mask), 0) # 첫번째 인풋에 broadcastable한 두번째 아규먼트사용, input>2nd이면 true
        enc_dec_pad_mask = create_padding_mask(dec_input, enc_input, self.pad_idx)

        dec_self_attn_prob = []
        enc_dec_attn_prob = []

        for layer in self.layers:
            dec_output, self_attn, enc_dec_attn = layer(dec_output, enc_output, self_mask, enc_dec_pad_mask)
            dec_self_attn_prob.append(self_attn)
            enc_dec_attn_prob.append(enc_dec_attn)

        lin_out = self.total_inear(dec_output)
        dec_output_prob = self.softmax(lin_out)
        return dec_output_prob, dec_self_attn_prob, enc_dec_attn_prob

class Transformer(nn.Module): # encoder와 decoder의 emb_dim등은 다르려나?
    def __init__(self, voca_size, emb_dim, max_seq_len, pos_dropout,dropout_attn,
                 dropout_multi, d_ff, dropout_ff, n_layers, n_head, pad_idx, layernorm_epsilon):
        super().__init__()
        self.encoder = Encoder(voca_size, emb_dim, max_seq_len, pos_dropout,dropout_attn,
                               dropout_multi, d_ff, dropout_ff, n_layers, n_head, pad_idx, layernorm_epsilon)
        self.decoder = Decoder(voca_size, emb_dim, max_seq_len, pos_dropout, dropout_attn,
                               dropout_multi, d_ff, dropout_ff, n_layers, n_head, pad_idx, layernorm_epsilon)

    def forward(self, enc_input, dec_input):

        enc_output, self_enc_attn_prob = self.encoder(enc_input)

        dec_output, self_dec_attn_prob, dec_enc_attn_prob = self.decoder(dec_input, enc_output)
        return dec_output, self_enc_attn_prob, self_dec_attn_prob, dec_enc_attn_prob