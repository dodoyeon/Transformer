# import torch
# import torch.nn.functional as F

from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

matplotlib.rc('font', family='AppleGothic') # 한글 출력
plt.rcParams['axes.unicode_minus'] = False # 축 - 설정

digits = load_digits()

tsne = TSNE(random_state=0)
digits_tsne = tsne.fit_transform(digits.data)
colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
            '#A83683', '#4E655E', '#853541'] # , '#3A3120', '#535D8E'
for i in range(len(digits.data)): # 0부터  digits.data까지 정수
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), # x, y , 그룹
             color=colors[digits.target[i]], # 색상
             fontdict={'weight': 'bold', 'size':9}) # font

plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max()) # 최소, 최대
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max()) # 최소, 최대

plt.xlabel('t-SNE 특성0') # x축 이름
plt.ylabel('t-SNE 특성1') # y축 이름
plt.show() # 그래프 출력
# a = torch.randn(10).unsqueeze(0)

# b = a <= 0
# indices = b.nonzero()
# print(indices.size())

# b= torch.argsort(a)
# print(b.size()) # (1,10)

# a = [[1,3],[4,5]]
# b=torch.tensor(a)[:,1]
# print(len(a))
# print(b)

# table = []
# dist_sort = torch.randn(1,64)  # (1,64)
# sort = torch.argsort(dist_sort, dim=1).squeeze()
# i = 0
# idx = sort[i].item()
# row_idx = (idx // 8)  # 타겟에 대한 업데이트 센트로이드의 인덱스
# col_idx = (idx % 8)  # 업데이트에 대한 타겟의 인덱스
# table.append([row_idx, col_idx])
# while len(table) < 8:
#     i += 1
#     if i < (8) ** 2:
#         idx = sort[i].item()
#         row_idx = (idx // 8) # 몫: 타겟에 대한 업데이트 센트로이드의 인덱스 0
#         col_idx = (idx % 8) # 나머지: 업데이트에 대한 타겟의 인덱스 6
#         ten_table = torch.tensor(table)[:, 0]
#         ten_table1= torch.tensor(table)[:, 1]
# 
#         if (row_idx in ten_table.tolist()) or (col_idx in ten_table1.tolist()):
#             if len(table) == 8:
#                 break
#             else:
#                 continue
#         else:
#             table.append([row_idx, col_idx])
# 
#     else:
#         ten_table2 = torch.tensor(table)
#         check = torch.arange(8)
#         for m in range(8):
#             if check[m] not in ten_table2[:,0]:
#                 s0 = check[m]
#             if check[m] not in ten_table2[:,1]:
#                 s1 = check[m]
#         table.append([s0, s1])
#         # break 를 쓰면 안된다..table을 7개만 만들고 멈출수 있음
#     # if len(table) == 8:
#     #     break
#     i += 1
# 
# print(torch.tensor(table))

# dict_table = {}
# r = torch.randint(7,(64,))
# table = torch.tensor(([0,6],[7,7],[5,4],[2,0],[4,3],[1,2],[6,5],[3,1]))
#
# for i in range(table.size(0)):
#     dict_table[table[i,0].item()] = table[i,1].item()

# m = torch.sort(table,dim=0)

# print(dict_table)

# a = torch.rand(3,9,2)
# b = torch.rand(3,9,2)
# d1 = F.cosine_similarity(a,b)
#
# d2 = F.cosine_similarity(a[0,:,0].unsqueeze(0).unsqueeze(-1), b[0,:,0].unsqueeze(0).unsqueeze(-1)) # d1[0,0]
# d3 = F.cosine_similarity(a[1,:,0].unsqueeze(0).unsqueeze(-1), b[1,:,0].unsqueeze(0).unsqueeze(-1)) # d1[1,0]
# d4 = F.cosine_similarity(a[2,:,0].unsqueeze(0).unsqueeze(-1), b[2,:,0].unsqueeze(0).unsqueeze(-1)) # d1[2,0]
# d5 = F.cosine_similarity(a[0,:,1].unsqueeze(0).unsqueeze(-1), b[0,:,1].unsqueeze(0).unsqueeze(-1)) # d1[0,1]
# d6 = F.cosine_similarity(a[1,:,1].unsqueeze(0).unsqueeze(-1), b[1,:,1].unsqueeze(0).unsqueeze(-1)) # d1[1,1]
# d7 = F.cosine_similarity(a[2,:,1].unsqueeze(0).unsqueeze(-1), b[2,:,1].unsqueeze(0).unsqueeze(-1)) # d1[2,1]
# print(d1)
#
# print(d2)
# print(d3)
# print(d4)
#
# print(d5)
# print(d6)
# print(d7)

a =  torch.rand(3)
b = torch.sum(a)
print(b)