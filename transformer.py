import torch.nn as nn
from Attention import Embedding, PositionalEncoding, MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = Embedding()
        self.position_layer = PositionalEncoding()
        self.multiheadattn = MultiHeadAttention()
        self.regularize_layer = nn.LayerNorm()
    def forward(self, x):
        emb_x = self.embedding_layer(x)
        posemb_x = self.position_layer(emb_x)
        input_q = posemb_x
        input_k = posemb_x
        input_v = posemb_x
        residual = posemb_x
        out = residual + self.multiheadattn(input_q, input_k, input_v)
        attn_output = self.regularize_layer(out)
        pos_output = self.position_layer(attn_output)
        enc_output = self.regularize_layer(pos_output)
        return enc_output

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = Embedding()
        self.position_layer = PositionalEncoding()
        self.multiheadattn = MultiHeadAttention()
        self.maskedattn = MultiHeadAttention()
        self.regularize_layer = nn.LayerNorm()
        self.total_inear = nn.Linear()

    def forward(self,x, enc_output):
        emb_x = self.embedding_layer(x)
        posemb_x = self.position_layer(emb_x)
        input_q = posemb_x
        input_k = posemb_x
        input_v = posemb_x
        residual_1 = posemb_x
        out_1 = residual_1 + self.maksedattn(input_q, input_k, input_v)
        masked_output = self.regularize_layer(out_1)
        residual_2 = masked_output
        attn_out = residual_2 + self.multiheadattn(input_q, enc_output, enc_output)
        pos_output = self.position_layer(attn_out)
        dec_output = self.regularize_layer(pos_output)
        out_2 = self.total_inear(dec_output)
        output_prob = nn.Softmax(out_2)
        return output_prob

class Transformer(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, enc_input):
        enc_out = self.encoder(enc_input)
        output = self.decoder(dec_input, enc_out)
        return output