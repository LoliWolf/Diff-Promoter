# D:\code\git-project\Diff-Promoter\.venv\Scripts\python.exe D:\code\git-project\Diff-Promoter\Diff-Promoter\generate.py -task_id 1 -species maize
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
parser.add_argument('-species', type=str, help='maize | sorghum | arabidopsis')
args = parser.parse_args()

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
print(f"Using device: {device}")

species = args.species
model_name = ''
if species == 'maize':
    model_name = 'maize'
elif species == 'sorghum':
    model_name = 'sorghum'
elif species == 'arabidopsis':
    model_name = 'arabidopsis'
else:
    raise ValueError("species参数错误")

timesteps = 1000
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

os.makedirs('res', exist_ok=True)
# model = UNetModel()
model = torch.load('params/' + model_name + '.pkl', map_location=device, weights_only=False)
if species == 'sorghum' or species == 'arabidopsis':
    model = UNetModel()  # 先创建模型实例
    state_dict = torch.load('params/' + model_name + '.pkl', map_location=device)  # 加载状态字典
    model.load_state_dict(state_dict)
model.to(device)
model.eval()
g_seqs = generate_gen(model, gaussian_diffusion, 1)  # 输入 生成几条
with open('Diff-Promoter/res/' + f"{args.task_id}" + '.fasta', 'w') as f:  # 生成 fasta 文件
    for idx, seq in enumerate(g_seqs):
        f.write('>gen_' + str(idx) + '\n')
        f.write(seq + '\n')
