# D:\code\git-project\Diff-Promoter\.venv\Scripts\python.exe D:\code\git-project\Diff-Promoter\grads\scores.py -task_id 1 -env NDM -species maize -gene gene_0:ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT,gene_1:ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAATTGGCCAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCATCGATGCTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT
# python grads/scores.py -task_id 1 -env NDM -species maize -gene gene_0:ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAACCTTAAAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCACTGTCGTTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT,gene_1:ACCTTGAAAGTATTTTTCACTGTATTTTGACGTCAGCCCATCACAATCTCGAAATTGGCCAGCTTATCGCGGCTTGCCCCGCCCACCACACGCACTGCCATGAATCCCCGCGCACTGATCATGCTCAGCATCGATGCTTTCAGTGGGGGTGGCCAGAAAAGAGACCAGCT
import argparse
import os

import matplotlib
import numpy as np
import Bio.SeqIO
import torch
from captum.attr import DeepLiftShap
from matplotlib import pyplot as plt

from adv_model import Net
from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence
from itertools import islice

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

matplotlib.use('Agg')

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

if device is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')

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
model.load_state_dict(torch.load(filename, map_location=device)) # 3x6 18
model.to(device)
model.eval()


def seq2tensor(seq, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    encode_seq = []
    for element in seq.upper():
        encode_seq.append(one_hot[element])
    return torch.tensor(encode_seq, dtype=torch.float32, requires_grad=True, device=device).t().unsqueeze(dim=0)


def seqs2tensor(seqs, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}

    seqs_tensor = []
    for seq in seqs:
        encode_seq = []
        for element in seq.upper():
            encode_seq.append(one_hot[element])
        seqs_tensor.append(torch.tensor(encode_seq, dtype=torch.float32, device=device).t().unsqueeze(dim=0))
    return torch.cat(seqs_tensor, 0)


# 解析启动子参数
dict_promoter = {}
if args.gene:
    for item in args.gene.split(','):
        gene, seq = item.split(':')
        dict_promoter[gene] = seq.upper()


os.makedirs(f'grads/res/{task_id}', exist_ok=True)
# 处理每个启动子序列
for gene, seq in dict_promoter.items():
    seq_tensor = seq2tensor(seq)

    model.zero_grad()
    dls = DeepLiftShap(model, multiply_by_inputs=False)
    grads = dls.attribute(seq_tensor,
                          baselines=seqs2tensor(dinuc_shuffle(seq, num_shufs=100, rng=np.random.RandomState(123))))
    list_grads = (grads * seq_tensor).detach().cpu().squeeze(dim=0).numpy()
    # viz_sequence.plot_weights_yxs(list_grads, subticks_frequency=20, savewhere=f'grads/res/{task_id}/' + gene + '.png')
    viz_sequence.plot_weights(list_grads, subticks_frequency=20)
    fig = plt.gcf()
    outpath = f'grads/res/{task_id}/' + gene + '.png'
    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close(fig)

# list_res = [] # 放任意条 170bp
# with open(r'../sswm_res/max_seq_change.csv', 'r') as f: # 换成序列dict
#     # for line in islice(f, 1, None):
#     for line in f:
#         list_res.append(line[:-1].split(','))
#
#
# seqs = np.array(list_res).T
#
#
# for res in seqs:
#     gene = res[0]
#
#     for iters, seq in enumerate(res[1:]):
#         seq = str(seq)
#         seq_tensor = seq2tensor(seq)
#
#         model.zero_grad()
#         dls = DeepLiftShap(model, multiply_by_inputs=False)
#         grads = dls.attribute(seq_tensor, baselines=seqs2tensor(dinuc_shuffle(seq, num_shufs=100, rng=np.random.RandomState(123))))
#         list_grads = (grads * seq_tensor).detach().cpu().squeeze(dim=0).numpy()
#         viz_sequence.plot_weights_yxs(list_grads, subticks_frequency=20, savewhere='./res/'+gene+'_iter_'+ str(iters) + '.png')


'''
list_res = []
for gene in list_promoter:
    list_res.append((gene, dict_wdm[gene]))
list_res.sort(key=lambda x: x[1], reverse=True)

for gene, v in list_res[:20]:
    seq = dict_promoter[gene]
    seq_tensor = seq2tensor(seq)

    model.zero_grad()
    dls = DeepLiftShap(model, multiply_by_inputs=False)
    grads = dls.attribute(seq_tensor, baselines=seqs2tensor(dinuc_shuffle(seq, num_shufs=100, rng=np.random.RandomState(12301))))
    list_grads = (grads * seq_tensor).detach().cpu().squeeze(dim=0).numpy()

    np.savetxt('./res/high_'+gene+'.csv', list_grads, delimiter=',', fmt='%s')
    viz_sequence.plot_weights_yxs(list_grads, subticks_frequency=20, savewhere='./res/high_'+gene+'.png')

for gene, v in list_res[-20:]:
    seq = dict_promoter[gene]
    seq_tensor = seq2tensor(seq)

    model.zero_grad()
    dls = DeepLiftShap(model, multiply_by_inputs=False)
    grads = dls.attribute(seq_tensor, baselines=seqs2tensor(dinuc_shuffle(seq, num_shufs=100, rng=np.random.RandomState(12301))))
    list_grads = (grads * seq_tensor).detach().cpu().squeeze(dim=0).numpy()

    np.savetxt('./res/low_'+gene+'.csv', list_grads, delimiter=',', fmt='%s')
    viz_sequence.plot_weights_yxs(list_grads, subticks_frequency=20, savewhere='./res/low_'+gene+'.png')

mid = int(len(list_res) / 2)
for gene, v in list_res[mid-10:mid+10]:
    seq = dict_promoter[gene]
    seq_tensor = seq2tensor(seq)

    model.zero_grad()
    dls = DeepLiftShap(model, multiply_by_inputs=False)
    grads = dls.attribute(seq_tensor, baselines=seqs2tensor(dinuc_shuffle(seq, num_shufs=100, rng=np.random.RandomState(12301))))
    list_grads = (grads * seq_tensor).detach().cpu().squeeze(dim=0).numpy()

    np.savetxt('./res/mid_'+gene+'.csv', list_grads, delimiter=',', fmt='%s')
    viz_sequence.plot_weights_yxs(list_grads, subticks_frequency=20, savewhere='./res/mid_'+gene+'.png')
'''
