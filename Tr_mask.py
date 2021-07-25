import torch
# def create_mask(src: torch.Tensor,
#                 trg: torch.Tensor,
#                 src_pad_idx: int,
#                 trg_pad_idx: int):
#     src_mask = _create_padding_mask(src, src_pad_idx)
#     trg_mask = None
#     if trg is not None:
#         trg_mask = _create_padding_mask(trg, trg_pad_idx)  # (256, 1, 33)
#         nopeak_mask = _create_nopeak_mask(trg)  # (1, 33, 33)
#         trg_mask = trg_mask & nopeak_mask  # (256, 33, 33)
#     return src_mask, trg_mask

# paul-hyun reference
def create_padding_mask(seq_q, seq_k, pad_idx):  # pad_idx = 0
    """
    seq 형태를  (256, 33) -> (256, 1, 33) 이렇게 변경합니다.

    아래와 같이 padding index부분을 False로 변경합니다. (리턴 tensor)
    아래의 vector 하나당 sentence라고 보면 되고, True로 되어 있는건 단어가 있다는 뜻.
    tensor([[[ True,  True,  True,  True, False, False, False]],
            [[ True,  True, False, False, False, False, False]],
            [[ True,  True,  True,  True,  True,  True, False]]])
    """
    batch_size = seq_q.size(0)
    len_q = seq_q.size(1)
    len_k = seq_k.size(1)

    pad_attn_mask = seq_k.data.eq(pad_idx) # if same as pad_idx -> True
    pad_attn_mask = pad_attn_mask.unsqueeze(1) # increase dim
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k) # expand()
    return pad_attn_mask

def create_attn_decoder_mask(seq):
    """
    Attention Decoder MASK
    Target의 경우 그 다음 단어를 못보게 가린다
    """
    decoder_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1)) # ones_like: filled with 1 with same size of input
    decoder_mask = decoder_mask.triu(diagonal=1) # torch.triu: nxn 행렬에서 위쪽 삼각을 리턴(2D) down is all 0/ diagonal= return amount(1: not contain the diag-self)

    # decoder_mask = torch.triu(decoder_mask, diagonal=1)
    return decoder_mask

# # Example
# sentences = torch.Tensor([[2., 4., 6., 2., 7., 1., 1.],
#                           [2., 2., 4., 1., 1., 1., 1.],
#                           [1., 2., 8., 2., 9., 3., 1.]])
# src_mask = create_padding_mask(sentences, sentences, 1)
# trg_mask = create_attn_decoder_mask(sentences)
# 
# print('sentences:', sentences.shape)
# print('src_mask :', src_mask.shape)
# print('src_mask :', src_mask)
# print('trg_mask :', trg_mask.shape)
# print('trg_mask :', trg_mask)