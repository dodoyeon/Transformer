import torch.nn as nn
from Attention import Embedding, PositionalEncoding, MultiHeadAttention, PositionalFeedForward

class Encoder_layer(nn.Module):
    def __init__(self, embed_dim, n_head, dropout_attn, dropout_multi, d_ff, dropout_ff, layernorm_epsilon):
        super().__init__()
        self.multiheadattn = MultiHeadAttention(embed_dim, n_head, dropout_attn, dropout_multi)
        self.normal_layer1 = nn.LayerNorm(eps= layernorm_epsilon)
        self.normal_layer2 = nn.LayerNorm(eps= layernorm_epsilon)
        self.posffn_layer = PositionalFeedForward(embed_dim, d_ff, dropout_ff)

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

class Decoder_layer(nn.Module):
    def __init__(self, embed_dim, n_head, dropout_attn, dropout_multi,d_ff, dropout_ff, layernorm_epsilon):
        super().__init__()
        self.multiheadattn = MultiHeadAttention(embed_dim, n_head, dropout_attn, dropout_multi)
        self.maskedattn = MultiHeadAttention(embed_dim, n_head, dropout_attn, dropout_multi)
        self.normal_layer1 = nn.LayerNorm(eps = layernorm_epsilon)
        self.total_inear = nn.Linear()
        self.normal_layer2 = nn.LayerNorm(eps = layernorm_epsilon)
        self.posffn_layer = PositionalFeedForward(embed_dim, d_ff, dropout_ff)

    def forward(self, dec_input, enc_output):
        input_q = dec_input
        input_k = dec_input
        input_v = dec_input
        residual_1 = dec_input
        attn_out, attn_prob = self.maksedattn(input_q, input_k, input_v)
        res_out1 =residual_1 + attn_out
        masked_output = self.regularize_layer(res_out1)
        residual_2 = masked_output
        mask_out, mask_prob = self.multiheadattn(input_q, enc_output, enc_output)
        res_out2 = residual_2 + mask_out
        pos_output = self.posffn_layer(res_out2)
        dec_output = self.regularize_layer(pos_output)
        lin_out = self.total_inear(dec_output)
        output_prob = nn.Softmax(lin_out)
        return output_prob,

class Encoder(nn.Module):
    def __init__(self, voca_size, emb_dim, max_seq_len, pos_dropout, n_layers):
        super().__init__()
        self.embedding_layer = Embedding(voca_size, emb_dim)
        self.position_layer = PositionalEncoding(max_seq_len, emb_dim, pos_dropout)
        self.layers = nn.ModuleList([Encoder_layer for _ in range(n_layers)])
    def forward(self):
        return

class Decoder(nn.Module):
    def __init__(self, voca_size, emb_dim, max_seq_len, pos_dropout, n_layers):
        super().__init__()
        self.embedding_layer = Embedding(voca_size, emb_dim)
        self.position_layer = PositionalEncoding(max_seq_len, emb_dim, pos_dropout)
        self.layers = nn.ModuleList([Encoder_layer for _ in range(n_layers)])
    def forward(self):
        return

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, enc_input, dec_input):
        emb_enc = self.embedding_layer(enc_input)
        posemb_enc = self.position_layer(emb_enc)
        enc_out = self.encoder(posemb_enc)
        emb_dec = self.embedding_layer(dec_input)
        posemb_dec = self.position_layer(emb_dec)
        output = self.decoder(posemb_dec, enc_out)
        return output