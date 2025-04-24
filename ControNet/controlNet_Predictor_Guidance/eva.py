# 示例指令
# D:\code\git-project\Diff-Promoter\.venv\Scripts\python.exe D:\code\git-project\Diff-Promoter\ControNet\controlNet_Predictor_Guidance\eva.py -task_id 12 -env WDM -gene_name name1 -sequence ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT -position 164 -target 3.5 -species maize
# python ControNet/controlNet_Predictor_Guidance/eva.py -task_id 12 -env WDM -gene_name name1 -sequence ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT -position 164 -target 3.5 -species maize
import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm
import sys

original_init = tqdm.__init__
def new_init(self, *args, **kwargs):
    if 'file' not in kwargs:
        kwargs['file'] = sys.stdout
    original_init(self, *args, **kwargs)
tqdm.__init__ = new_init

from controlnet import *
from utils import *
import numpy as np
from adv_model import Net as advNet
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-task_id', type=int, help='后端task_id')
parser.add_argument('-env', type=str, help='使用环境')
parser.add_argument('-gene_name', type=str, help='基因名字(fasta)') # 没用 随便输入
parser.add_argument('-sequence', type=str, help='基因序列(fasta)')
parser.add_argument('-position', type=int, help='指定位置')
parser.add_argument('-target', type=float, help='指定目标活性')
parser.add_argument('-species', type=str, help='maize | sorghum | arabidopsis')

args = parser.parse_args()
task_id = args.task_id
env = args.env
gene_name = args.gene_name
sequence = args.sequence
position = args.position
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

model = ControlNet().to(device)
model.load_state_dict(torch.load('ControNet/controlNet_Predictor_Guidance/ControlNet_param.pkl', map_location=device))
model.eval()

adv_model = advNet().to(device)
adv_model.load_state_dict(torch.load(filename, map_location=device)) # 选择环境
adv_model.eval()

# target 输入[1,15]
def cond_fn(x, target=target_input, guidance_loss_scale=10000):
    loss_fn = nn.MSELoss()
    tar = torch.tensor([target] * x.size(0), dtype=torch.float32, device=x.device)
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        y = adv_model(x_in)
        # loss = torch.abs(target - y).mean()
        # loss = torch.square(target - y).mean()
        loss = loss_fn(y, tar)
        return -torch.autograd.grad(loss, x_in)[0] * guidance_loss_scale

timesteps = 1000
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)


def seqs2tensor(seqs, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}

    seqs_tensor = []
    for seq in seqs:
        encode_seq = []
        for element in 'NNN' + seq.upper() + 'NNN':
            encode_seq.append(one_hot[element])
        seqs_tensor.append(torch.tensor(encode_seq, dtype=torch.float32, device=device).t().unsqueeze(dim=0))
    return torch.cat(seqs_tensor, 0)

hot_one = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

import pandas as pd
# 用户要自己输入序列
# dict_seqs = {"gen_0":"ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT"}
dict_seqs = {gene_name:sequence}
# df = pd.read_excel(r'gene_change_20bp_from_135.xlsx')
# for row in df.index.values:
#     dict_seqs[df.iloc[row, 1]] = df.iloc[row, 2].upper()

# dict_seqs 一个序列一个元素 让用户输入
for gene, seq in dict_seqs.items():
    x_start = torch.randn(100, 4, 176, device=device)

    idx = position
    # for idx in range(len(seq) - 6):
    c_seqs = [seq[:idx] + 'N'*6 + seq[idx+6:]] * 100 # 拿输入的 前后和这里比较 一样的保留 作为结果
    c = seqs2tensor(c_seqs)

    #####ori#####
    generate_gen = gaussian_diffusion.sample(model, c, 176, batch_size=100, channels=4, cond=False, x_start=x_start)
    os.makedirs(f'ControNet/controlNet_Predictor_Guidance/res/{task_id}/', exist_ok=True)
    np.save(f"ControNet/controlNet_Predictor_Guidance/res/{task_id}/"+ gene +  "_" + str(idx) + "_ori.npy", generate_gen[-1])

    ori_seqs = []
    for seq in generate_gen[-1]:
        res = ''
        for a in seq.T[3:-3]:
            index = np.argmax(a)
            res += hot_one[index]
        ori_seqs.append(res)

    with open(f'ControNet/controlNet_Predictor_Guidance/res/{task_id}/' + gene + '_' + str(idx) + '_ori_gene.txt', 'w') as f:
        for i, seq in enumerate(ori_seqs):
            f.write('>gen_' + str(i) + '\n')
            f.write(seq + '\n')

    sss = torch.tensor(generate_gen[-1], dtype=torch.float32, device=device)
    tensor_v = adv_model(sss).detach().cpu().numpy().tolist()

    sss_cc = seqs2tensor(ori_seqs)
    tensor2seq_v = adv_model(sss_cc).detach().cpu().numpy().tolist()

    with open(f'ControNet/controlNet_Predictor_Guidance/res/{task_id}/' + gene + '_' + str(idx) + '_ori_pred_v.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['gene', 'tensor_v', 'tensor2seq_v'])

        for i, (tv, tsv) in enumerate(list(zip(tensor_v, tensor2seq_v))):
            csv_writer.writerow(['gen_' + str(i), tv, tsv])

    #####last#####
    generate_gen = gaussian_diffusion.sample(model, c, 176, batch_size=100, channels=4, cond=True, cond_fn=cond_fn, x_start=x_start)

    np.save(f"ControNet/controlNet_Predictor_Guidance/res/{task_id}/"+ gene + "_" + str(idx) + "_last.npy", generate_gen[-1])

    last_seqs = []
    for seq in generate_gen[-1]:
        res = ''
        for a in seq.T[3:-3]:
            index = np.argmax(a)
            res += hot_one[index]
        last_seqs.append(res)

    with open(f'ControNet/controlNet_Predictor_Guidance/res/{task_id}/'+ gene + '_' + str(idx) + '_last_gene.txt', 'w') as f:
        for i, seq in enumerate(last_seqs):
            f.write('>gen_' + str(i) + '\n')
            f.write(seq + '\n')

    sss = torch.tensor(generate_gen[-1], dtype=torch.float32, device=device)
    tensor_v = adv_model(sss).detach().cpu().numpy().tolist()

    sss_cc = seqs2tensor(last_seqs)
    tensor2seq_v = adv_model(sss_cc).detach().cpu().numpy().tolist()

    with open(f'ControNet/controlNet_Predictor_Guidance/res/{task_id}/'+ gene + '_' + str(idx) + '_last_pred_v.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['gene', 'tensor_v', 'tensor2seq_v'])

        for i, (tv, tsv) in enumerate(list(zip(tensor_v, tensor2seq_v))):
            csv_writer.writerow(['gen_' + str(i), tv, tsv])

    np.save(f"ControNet/controlNet_Predictor_Guidance/res/{task_id}/" + gene + "_input.npy", x_start.detach().cpu().numpy())


'''
#################### same input ####################
x_start = torch.randn(1800, 4, 176, device=device)
# np.save("input.npy", x_start.detach().cpu().numpy())
# x_start = np.load(r'/home/yxs/Diffusion/maize_promoter/v6_epoch1000_valid/008_evalution/m2/res/input.npy')
# x_start = torch.tensor(x_start, device=device)

import pandas as pd
c_seqs = []
df = pd.read_excel(r'gene_change_20bp_from_135.xlsx')
for row in df.index.values:
    tmp = df.iloc[row, 2].upper()
    c_seqs += [tmp[:135] + 'N' * 20 + tmp[155:]] * 200

c = seqs2tensor(c_seqs)


#################### no guidance ####################

generate_gen = gaussian_diffusion.sample(model, c, 176, batch_size=1800, channels=4, cond=False, x_start=x_start)

# for idx, seq in enumerate(generate_gen[-1]):
#     np.savetxt("res/" + str(idx) + ".csv", seq, delimiter=',')

np.save("res/ori.npy", generate_gen[-1])

hot_one = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

ori_seqs = []
for seq in generate_gen[-1]:
    res = ''
    for a in seq.T[3:-3]:
        index = np.argmax(a)
        res += hot_one[index]
    ori_seqs.append(res)

with open('res/ori_gene.txt', 'w') as f:
    for i, seq in enumerate(ori_seqs):
        f.write('>gen_' + str(i) + '\n')
        f.write(seq + '\n')

sss = torch.tensor(generate_gen[-1], dtype=torch.float, device=device)
tensor_v = adv_model(sss).detach().cpu().numpy().tolist()

sss_cc = seqs2tensor(ori_seqs)
tensor2seq_v = adv_model(sss_cc).detach().cpu().numpy().tolist()

with open('res/ori_pred_v.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gene', 'tensor_v', 'tensor2seq_v'])

    for i, (tv, tsv) in enumerate(list(zip(tensor_v, tensor2seq_v))):
        csv_writer.writerow(['gen_' + str(i), tv, tsv])


#################### has guidance ####################

generate_gen = gaussian_diffusion.sample(model, c, 176, batch_size=1800, channels=4, cond=True, cond_fn=cond_fn, x_start=x_start)

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
