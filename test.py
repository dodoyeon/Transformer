import time
import math
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from transformer import Transformer
from train import test_data

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"), init_token='<sos>', eos_token='<eos>', lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bptt = 35 # 이게 뭐야?
def get_batch(source, i):
    seq_len = min(bptt, len(source)-1-i)
    data = source[i:i+seq_len]
    target = source[i+1,i+1+seq_len].view(-1)
    return data, target

criterion = nn.CrossEntropyLoss()
# best_model 은 train에서 학습한 모델을 가져와야 하는데 그거 어케하지?

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    n_tokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0)-1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, n_tokens)
            total_loss += len(data)*criterion(output_flat, targets).item()
        return total_loss/(len(data_source)-1)

# test
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)