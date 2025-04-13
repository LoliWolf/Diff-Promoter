# D:\code\git-project\Diff-Promoter\.venv\Scripts\python.exe D:\code\git-project\Diff-Promoter\Diff-Promoter\generate.py -task_id 1
# python Diff-Promoter/generate.py -task_id 1
import argparse
import sys
import time

sys.path.append('..')
import torch
from model import *
from utils import *
# from dataset import *
import numpy as np
import os
from itertools import islice

# if os.path.exists('res'):
#     import shutil
#     shutil.rmtree('res')

parser = argparse.ArgumentParser()
parser.add_argument('-task_id', type=int, default=1, help='backend task id')
args = parser.parse_args()

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

timesteps = 1000
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

os.makedirs('res', exist_ok=True)
# model = UNetModel()
model = torch.load('params/model.pkl', map_location=device,  weights_only=False)
model.to(device)
model.eval()
g_seqs = generate_gen(model, gaussian_diffusion, 1) # 输入 生成几条
with open('Diff-Promoter/res/' + f"{args.task_id}" + '.fasta', 'w') as f: # 生成 fasta 文件
    for idx, seq in enumerate(g_seqs):
        f.write('>gen_' + str(idx) + '\n')
        f.write(seq + '\n')