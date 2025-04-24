# D:\code\git-project\Diff-Promoter\.venv\Scripts\python.exe D:\code\git-project\Diff-Promoter\motif_implantion\muta.py -task_id 12 -env WDM -gene_name name1 -sequence ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT -position 164 -species maize -motif_seq ATGC
# python motif_implantion/muta.py -task_id 12 -env WDM -gene_name name1 -sequence ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT -position 164 -species maize -motif_seq ATGC
import argparse
import os

import torch
from adv_model import Net
import csv
import Bio.SeqIO
import pandas as pd
import numpy as np
from itertools import islice
import random

parser = argparse.ArgumentParser()
parser.add_argument('-task_id', type=int, help='后端task_id')
parser.add_argument('-env', type=str, help='使用环境')
parser.add_argument('-gene_name', type=str, help='基因名字(fasta)')  # 没用 随便输入
parser.add_argument('-sequence', type=str, help='基因序列(fasta)')
parser.add_argument('-position', type=int, help='指定位置')
parser.add_argument('-motif_seq', type=str, help='插入的序列')
parser.add_argument('-species', type=str, help='maize | sorghum | arabidopsis')

args = parser.parse_args()
task_id = args.task_id
env = args.env
gene_name = args.gene_name
sequence = args.sequence
position = args.position
motif_seq = args.motif_seq

device = None
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    try:
        device = torch.device('cuda:0')
        # 简单测试CUDA是否真的可用
        torch.zeros(1).to(device)
    except:
        device = None

if device is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')

if device is None:
    device = torch.device('cpu')
print(f"Using device: {device}")

# 环境选择
species = args.species
file_path_species = ''
if species == 'maize':
    file_path_species = 'maize'
elif species == 'sorghum':
    file_path_species = 'sorghum'
elif species == 'arabidopsis':
    file_path_species = 'arabidopsis'
else:
    raise ValueError("species参数错误")
file_path = f'Predictor-Guidance/{file_path_species}/'
if env == 'WDM':
    filename = file_path + "wdm/model_params.pkl"
elif env == 'NDM':
    filename = file_path + "ndm/model_params.pkl"
elif env == 'NDT':
    filename = file_path + "ndt/model_params.pkl"
elif env == 'NLT':
    filename = file_path + "nlt/model_params.pkl"
elif env == 'WDT':
    filename = file_path + "wdt/model_params.pkl"
elif env == 'WLT':
    filename = file_path + "wlt/model_params.pkl"
else:
    raise ValueError("env参数错误")

model = Net()
model.load_state_dict(torch.load(filename, map_location=device))  # 18环境
model.to(device)
model.eval()

def seq2tensor(seq, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    encode_seq = []
    for element in seq.upper():
        encode_seq.append(one_hot[element])
    return torch.tensor(encode_seq, dtype=torch.float32, device=device).t().unsqueeze(dim=0)


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

dict_promoter = {gene_name: sequence}  # 读一个序列
# with open(r'/home/yxs/Diffusion/promoter_arabidopsis/data/Arabidopsis_seq.csv', 'r') as f:
#     for line in islice(f, 1, None):
#         tmp = line[:-1].split(',')
#         dict_promoter[tmp[0]] = tmp[11]


seq, idx, motif = sequence, position, motif_seq  # idx 插入的位置，插入的 motif string

ori_v = model(seq2tensor(seq)).item()  # seq预测出的值

tmp_seq = seq[:idx] + motif + seq[idx + len(motif):]  # tmp_seq 插入后的序列
tmp_res = model(seq2tensor(tmp_seq)).item()  # tmp_res是插入后预测的值

os.makedirs(f'motif_implantion/res/{task_id}', exist_ok=True)
with open(f'motif_implantion/res/{task_id}/result.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gene', 'base_sequence', 'base_pred', 'motif', 'motif_pred'])
    csv_writer.writerow([gene_name, seq, ori_v, tmp_seq, tmp_res])
