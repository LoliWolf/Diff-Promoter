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

device = torch.device('cpu')

timesteps = 1000
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)


# model = UNetModel()
model = torch.load('../params/model.pkl', map_location=device)
model.to(device)
model.eval()
g_seqs = generate_gen(model, gaussian_diffusion, 1) # 输入 生成几条
with open('./res/' + "" + '_' + time.time().__str__() + '_gene_seqs.fasta', 'w') as f: # 生成 fasta 文件
    for idx, seq in enumerate(g_seqs):
        f.write('>gen_' + str(idx) + '\n')
        f.write(seq + '\n')