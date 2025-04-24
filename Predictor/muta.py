# D:\code\git-project\Diff-Promoter\.venv\Scripts\python.exe D:\code\git-project\Diff-Promoter\Predictor\muta.py -task_id 1 -env NDM -species maize -gene gene_0:ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT,gene_1:ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT
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
parser.add_argument('-species', type=str, help='maize | sorghum | arabidopsis')
parser.add_argument('-gene', type=str,required=True, help='启动子序列，格式为"基因名:序列"，多个启动子用逗号分隔')

args = parser.parse_args()
task_id = args.task_id
env = args.env

device = None
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    try:
        device = torch.device('cuda:0')
        # 简单测试CUDA是否真的可用
        torch.zeros(1).to(device)
    except:
        device = None

# if device is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')

if device is None:
    device = torch.device('cpu')

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
model.load_state_dict(torch.load(filename, map_location=device)) # 选六种环境 18
model.to(device)
model.eval()

def seq2tensor(seq, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    encode_seq = []
    for element in seq.upper():
        encode_seq.append(one_hot[element])
    return torch.tensor(encode_seq, dtype=torch.float, device=device).t().unsqueeze(dim=0)

# 解析启动子参数
dict_promoter = {}
if args.gene:
    for item in args.gene.split(','):
        gene, seq = item.split(':')
        dict_promoter[gene] = seq.upper()

# dict_promoter = {"gen_0":"ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT"}
dict_res = {}
for gene, seq in dict_promoter.items():
    predict_v = model(seq2tensor(seq)).item() # predict_v 对应seq预测的活性值
    dict_res[gene] = predict_v
    print(gene, predict_v) # 写到文件

os.makedirs(f'Predictor/res/{task_id}', exist_ok=True)
with open(f'Predictor/res/{task_id}/pred_v.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gene', 'pred_v'])
    for gene, pred_v in dict_res.items():
        csv_writer.writerow([gene, pred_v])



