import torch
from adv_model import Net
import csv
import Bio.SeqIO
import pandas as pd
import numpy as np
from itertools import islice
import random


device = torch.device('cuda:4')

model = Net()
model.load_state_dict(torch.load('model_params.pkl', map_location=device)) # 18环境
model.to(device)
model.eval()

def seq2tensor(seq, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    encode_seq = []
    for element in seq.upper():
        encode_seq.append(one_hot[element])
    return torch.tensor(encode_seq, dtype=torch.float, device=device).t().unsqueeze(dim=0)


# dict_res = []
# df1 = pd.read_csv(r'stremep001_pos_ara.csv')
# for row in df1.index.values:
#     dict_res.append([df1.iloc[row, 0], df1.iloc[row, 1], df1.iloc[row, 2], df1.iloc[row, 3], df1.iloc[row, 4]])


# def shuffleMotif(s, s_seed=123, n=3):
#     random.seed(s_seed)
#     res = []
#     for _ in range(n):
#         t = list(s)
#         random.shuffle(t)
#         res.append("".join(t))
#     return res

dict_promoter = {} # 读一个序列
with open(r'/home/yxs/Diffusion/promoter_arabidopsis/data/Arabidopsis_seq.csv', 'r') as f:
    for line in islice(f, 1, None):
        tmp = line[:-1].split(',')
        dict_promoter[tmp[0]] = tmp[11]



seq, idx, motif = 0 # idx 插入的位置，插入的 motif string

ori_v = model(seq2tensor(seq)).item() # seq预测出的值

tmp_seq = seq[:idx] + motif + seq[idx+len(motif):] # tmp_seq 插入后的序列
tmp_res =model(seq2tensor(tmp_seq)).item() # tmp_res是插入后预测的值

