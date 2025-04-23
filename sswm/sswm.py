# D:\code\git-project\Diff-Promoter\.venv\Scripts\python.exe D:\code\git-project\Diff-Promoter\sswm\sswm.py -task_id 12 -env WDM -species maize -gene_name name1 -sequence ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT
import argparse
import os

import torch
from adv_model import Net
import Bio.SeqIO
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-task_id', type=int, help='后端task_id')
parser.add_argument('-env', type=str, help='使用环境')
parser.add_argument('-gene_name', type=str, help='基因名字(fasta)')  # 没用 随便输入
parser.add_argument('-sequence', type=str, help='基因序列(fasta)')
parser.add_argument('-species', type=str, help='maize | sorghum | arabidopsis')

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

# 环境选择
args = parser.parse_args()
task_id = args.task_id
species = args.species
env = args.env
gene_name = args.gene_name
sequence = args.sequence
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

model = Net().to(device)
model.load_state_dict(torch.load(filename, map_location=device))  # 3*6 18
model.eval()


def seq2tensor(seq, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encode_seq = []
    for element in seq.upper():
        encode_seq.append(one_hot[element])
    return torch.tensor(encode_seq, dtype=torch.float, device=device).t().unsqueeze(dim=0)


def findMax(seq, v):
    maxSeq, maxV = seq, v

    for i in range(len(seq)):
        for base in ['A', 'C', 'G', 'T']:
            if base != seq[i]:
                new_seq = seq[:i] + base + seq[i + 1:]
                new_seq_tensor = seq2tensor(new_seq)
                new_v = model(new_seq_tensor).item()
                if new_v > maxV:
                    maxSeq = new_seq
                    maxV = new_v

    return maxSeq, maxV


def findMin(seq, v):
    minSeq, minV = seq, v

    for i in range(len(seq)):
        for base in ['A', 'C', 'G', 'T']:
            if base != seq[i]:
                new_seq = seq[:i] + base + seq[i + 1:]
                new_seq_tensor = seq2tensor(new_seq)
                new_v = model(new_seq_tensor).item()
                if new_v < minV:
                    minSeq = new_seq
                    minV = new_v

    return minSeq, minV


# dict_gene = {
#     "gen_0": "ACTATCGCCCGTACCGCCCCCCAGCCCCCACACATAAAAATAAACCCGACACTCTCGTCCCGCTTCCCCCCACCTAATCTCATGTAAGCGCCCAACACGTAAAAAGTGACTAAACCCGTCTCGTGATACCGAATACCCCAGCCTGACTCCACACTCATCTTTAACCAATG",
#     "gen_1": "TCTTCTTTTCCTCCGCCCGTGACAAAAAAAACGTCACACAACCAAGTCTCTTTTCGTCTCATTTTATGAGGCGTTTACGAAATCGTAAAAACGCCGTACGCGTTTAACGTTCCTCTGTAGCTTGGCCCACACGCGCGTAAAGTAGTTGATAGTCTACATATATATAAAAC"
# }
# for x in Bio.SeqIO.parse(r'./data/flower_gene_use.fasta', 'fasta'): # 序列dict input 10
#     dict_gene[x.id] = str(x.seq).upper()
dict_gene = {gene_name: sequence}
dcit_predV = {}
for name in dict_gene.keys():
    seq = dict_gene[name]
    seq_tensor = seq2tensor(seq)

    v = model(seq_tensor).item()
    dcit_predV[name] = v

dict_sswm_max, dict_sswm_min = {}, {}
dict_sswm_max[0] = (dict_gene, dcit_predV)
dict_sswm_min[0] = (dict_gene, dcit_predV)

for itertion in range(1, 16):

    res_max_seq, res_max_v = {}, {}
    gene_max, v_max = dict_sswm_max[itertion - 1]
    for gene in gene_max.keys():
        findSeq, findV = findMax(gene_max[gene], v_max[gene])
        res_max_seq[gene] = findSeq
        res_max_v[gene] = findV

    dict_sswm_max[itertion] = (res_max_seq, res_max_v)

    res_min_seq, res_min_v = {}, {}
    gene_min, v_min = dict_sswm_min[itertion - 1]
    for gene in gene_min.keys():
        findSeq, findV = findMin(gene_min[gene], v_min[gene])
        res_min_seq[gene] = findSeq
        res_min_v[gene] = findV

    dict_sswm_min[itertion] = (res_min_seq, res_min_v)

gene_names = list(dict_gene.keys())

os.makedirs(f'sswm/res/{task_id}', exist_ok=True)
with open(f'sswm/res/{task_id}/max_seq_change.csv', 'w', newline='') as f:  # 最大迭代 每轮序列 16行10列 每个元素是一条序列
    csv_writer = csv.writer(f)
    csv_writer.writerow(gene_names)  # 15轮 每轮换一个位置的碱基

    for itertion in range(0, 16):
        seq_max = dict_sswm_max[itertion][0]
        res = []
        for gene in gene_names:
            res.append(seq_max[gene])
        csv_writer.writerow(res)

with open(f'sswm/res/{task_id}/max_v_change.csv', 'w', newline='') as f:  # 最大迭代 每轮预测的活性值 16行10列 每个元素是一个值
    csv_writer = csv.writer(f)
    csv_writer.writerow(gene_names)

    for itertion in range(0, 16):
        v_max = dict_sswm_max[itertion][1]
        res = []
        for gene in gene_names:
            res.append(v_max[gene])
        csv_writer.writerow(res)

with open(f'sswm/res/{task_id}/min_seq_change.csv', 'w', newline='') as f:  # 最小迭代 每轮序列 16行10列 每个元素是一条序列
    csv_writer = csv.writer(f)
    csv_writer.writerow(gene_names)

    for itertion in range(0, 16):
        seq_min = dict_sswm_min[itertion][0]
        res = []
        for gene in gene_names:
            res.append(seq_min[gene])
        csv_writer.writerow(res)

with open(f'sswm/res/{task_id}/min_v_change.csv', 'w', newline='') as f:  # 最小迭代 每轮预测的活性值 16行10列 每个元素是一个值
    csv_writer = csv.writer(f)
    csv_writer.writerow(gene_names)

    for itertion in range(0, 16):
        v_min = dict_sswm_min[itertion][1]
        res = []
        for gene in gene_names:
            res.append(v_min[gene])
        csv_writer.writerow(res)

# 呈现 每条序列的值的变化 16条序列每个和初始的diff 画图 最大最小 两组

# asfddasf 1
# asfddasf 2
# asfddasf 3
