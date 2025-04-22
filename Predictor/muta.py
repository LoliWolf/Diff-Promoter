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
model.load_state_dict(torch.load('model_params.pkl', map_location=device)) # 选六种环境 18
model.to(device)
model.eval()

def seq2tensor(seq, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    encode_seq = []
    for element in seq.upper():
        encode_seq.append(one_hot[element])
    return torch.tensor(encode_seq, dtype=torch.float, device=device).t().unsqueeze(dim=0)

dict_promoter = {"gen_0":"ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT"}
for gene, seq in dict_promoter.items():
    predict_v = model(seq2tensor(seq)).item() # predict_v 对应seq预测的活性值
    print(gene, predict_v) # 写到文件
