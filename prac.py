import torch

a = torch.randn(10).unsqueeze(0)

# b = a <= 0
# indices = b.nonzero()
# print(indices.size())

# b= torch.argsort(a)
# print(b.size()) # (1,10)

a = [[1,3],[4,5]]
b=torch.tensor(a)[:,1]
print(len(a))
print(b)