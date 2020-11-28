import time
import torch
import torch.nn as nn
import transformer

model = transformer.Transformer()
criterion = nn.CrossEntropyLoss()
lr = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr)
