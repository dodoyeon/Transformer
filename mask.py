import torch
def create_mask(src: torch.Tensor,
                trg: torch.Tensor,
                src_pad_idx: int,
                trg_pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    src_mask = _create_padding_mask(src, src_pad_idx)
    trg_mask = None
    if trg is not None:
        trg_mask = _create_padding_mask(trg, trg_pad_idx)  # (256, 1, 33)
        nopeak_mask = _create_nopeak_mask(trg)  # (1, 33, 33)
        trg_mask = trg_mask & nopeak_mask  # (256, 33, 33)
    return src_mask, trg_mask

def _create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    seq 형태를  (256, 33) -> (256, 1, 31) 이렇게 변경합니다.

    아래와 같이 padding index부분을 False로 변경합니다. (리턴 tensor)
    아래의 vector 하나당 sentence라고 보면 되고, True로 되어 있는건 단어가 있다는 뜻.
    tensor([[[ True,  True,  True,  True, False, False, False]],
            [[ True,  True, False, False, False, False, False]],
            [[ True,  True,  True,  True,  True,  True, False]]])
    """
    return (seq != pad_idx).unsqueeze(-2)

def _create_nopeak_mask(trg) -> torch.Tensor:
    """
    NO PEAK MASK
    Target의 경우 그 다음 단어를 못보게 가린다
    """
    batch_size, seq_len = trg.size()
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device=trg.device), diagonal=1)).bool()
    return nopeak_mask
