# D:\code\git-project\Diff-Promoter\.venv\Scripts\python.exe D:\code\git-project\Diff-Promoter\Predictor-Guidance\eva.py -task_id 1 -env WDM -target 0.5
# python Predictor-Guidance/eva.py -task_id 1 -env WDM -target 0.5
import argparse
import io
import os
import sys

import torch
import torch.nn as nn
from sympy.strategies.core import switch
from torchgen.api.types import doubleT

from model import *
from utils import *
import numpy as np
from adv_model import Net as advNet
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-task_id', type=int, help='后端task_id')
parser.add_argument('-env', type=str, help='使用环境')
parser.add_argument('-target', type=float, help='目标活性')
parser.add_argument('-species', type=str, help='maize | sorghum | arabidopsis')

args = parser.parse_args()
task_id = args.task_id
env = args.env
target_input = args.target

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

model = torch.load(f'{file_path}model.pkl', map_location=device, weights_only=False)
if species == 'sorghum' or species == 'arabidopsis':
    model = UNetModel()  # 先创建模型实例
    state_dict = torch.load(f'{file_path}model.pkl', map_location=device, weights_only=False)  # 加载状态字典
    model.load_state_dict(state_dict)
model.to(device)
model.eval()

adv_model = advNet().to(device)
adv_model.load_state_dict(torch.load(filename, map_location=device))  # 选择环境
adv_model.eval()


def cond_fn(x, target=target_input, guidance_loss_scale=10000):  # target:目标活性 用户输入
    loss_fn = nn.MSELoss()
    tar = torch.tensor([target] * x.size(0), dtype=torch.float32, device=x.device)
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        y = adv_model(x_in)
        # loss = torch.abs(target - y).mean()
        # loss = torch.square(target - y).mean()
        loss = loss_fn(y, tar)
        return -torch.autograd.grad(loss, x_in)[0] * guidance_loss_scale


def seqs2tensor(seqs, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}

    seqs_tensor = []
    for seq in seqs:
        encode_seq = []
        for element in 'NNN' + seq.upper() + 'NNN':
            encode_seq.append(one_hot[element])
        seqs_tensor.append(torch.tensor(encode_seq, dtype=torch.float32, device=device).t().unsqueeze(dim=0))
    return torch.cat(seqs_tensor, 0)


timesteps = 1000
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

hot_one = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

ori_seqs, last_seqs = [], []
ori_v, last_v = [], []
for iters in range(1):  # 运行次数
    x_start = torch.randn(1, 4, 176, device=device)  # 1：次数

    generate_gen = gaussian_diffusion.sample(model, 176, batch_size=1, channels=4, cond=False,
                                             x_start=x_start)  # batch_size

    tmp_ori_seqs = []
    for seq in generate_gen[-1]:
        res = ''
        for a in seq.T[3:-3]:
            index = np.argmax(a)
            res += hot_one[index]
        tmp_ori_seqs.append(res)
    ori_seqs += tmp_ori_seqs
    ori_v += adv_model(seqs2tensor(tmp_ori_seqs)).detach().cpu().numpy().tolist()

    generate_gen = gaussian_diffusion.sample(model, 176, batch_size=1, channels=4, cond=True, cond_fn=cond_fn,
                                             x_start=x_start)  # batch_size
    tmp_last_seqs = []
    for seq in generate_gen[-1]:
        res = ''
        for a in seq.T[3:-3]:
            index = np.argmax(a)
            res += hot_one[index]
        tmp_last_seqs.append(res)
    last_seqs += tmp_last_seqs
    last_v += adv_model(seqs2tensor(tmp_last_seqs)).detach().cpu().numpy().tolist()

os.makedirs(f'Predictor-Guidance/res/{task_id}', exist_ok=True)
with open(f'Predictor-Guidance/res/{task_id}/ori_gene.txt', 'w') as f:
    for i, seq in enumerate(ori_seqs):
        f.write('>gen_' + str(i) + '\n')
        f.write(seq + '\n')

with open(f'Predictor-Guidance/res/{task_id}/last_gene.txt', 'w') as f:
    for i, seq in enumerate(last_seqs):
        f.write('>gen_' + str(i) + '\n')
        f.write(seq + '\n')

with open(f'Predictor-Guidance/res/{task_id}/ori_pred_v.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gene', 'pred_v'])

    for i, tv in enumerate(ori_v):
        csv_writer.writerow(['gen_' + str(i), tv])

with open(f'Predictor-Guidance/res/{task_id}/last_pred_v.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gene', 'pred_v'])

    for i, tv in enumerate(last_v):
        csv_writer.writerow(['gen_' + str(i), tv])
sys.exit(0)
'''
sss_cc = seqs2tensor(ori_seqs)
tensor2seq_v = adv_model(sss_cc).detach().cpu().numpy().tolist()

with open('res/ori_pred_v.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gene', 'tensor_v', 'tensor2seq_v'])

    for i, (tv, tsv) in enumerate(list(zip(tensor_v, tensor2seq_v))):
        csv_writer.writerow(['gen_' + str(i), tv, tsv])


#################### has guidance ####################

generate_gen = gaussian_diffusion.sample(model, 176, batch_size=2000, channels=4, cond=True, cond_fn=cond_fn, x_start=x_start)

# for idx, seq in enumerate(generate_gen[-1]):
#     np.savetxt("res/" + str(idx) + ".csv", seq, delimiter=',')

np.save("res/last.npy", generate_gen[-1])

hot_one = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

last_seqs = []
for seq in generate_gen[-1]:
    res = ''
    for a in seq.T[3:-3]:
        index = np.argmax(a)
        res += hot_one[index]
    last_seqs.append(res)

with open('res/last_gene.txt', 'w') as f:
    for i, seq in enumerate(last_seqs):
        f.write('>gen_' + str(i) + '\n')
        f.write(seq + '\n')

sss = torch.tensor(generate_gen[-1], dtype=torch.float, device=device)
tensor_v = adv_model(sss).detach().cpu().numpy().tolist()

sss_cc = seqs2tensor(last_seqs)
tensor2seq_v = adv_model(sss_cc).detach().cpu().numpy().tolist()

with open('res/last_pred_v.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gene', 'tensor_v', 'tensor2seq_v'])

    for i, (tv, tsv) in enumerate(list(zip(tensor_v, tensor2seq_v))):
        csv_writer.writerow(['gen_' + str(i), tv, tsv])


#################### end guidance ####################

np.save("res/input.npy", x_start.detach().cpu().numpy())
'''
