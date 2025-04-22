import torch
from adv_model import Net
import Bio.SeqIO
import csv

device = torch.device('cuda:3')

model = Net().to(device)
model.load_state_dict(torch.load('model_params.pkl', map_location=device)) # 3*6 18
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
                new_seq = seq[:i] + base + seq[i+1:]
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
                new_seq = seq[:i] + base + seq[i+1:]
                new_seq_tensor = seq2tensor(new_seq)
                new_v = model(new_seq_tensor).item()
                if new_v < minV:
                    minSeq = new_seq
                    minV = new_v
    
    return minSeq, minV



dict_gene = {}
for x in Bio.SeqIO.parse(r'./data/flower_gene_use.fasta', 'fasta'): # 序列dict input 10
    dict_gene[x.id] = str(x.seq).upper()

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

with open('res/max_seq_change.csv', 'w', newline='') as f: # 最大迭代 每轮序列 16行10列 每个元素是一条序列
    csv_writer = csv.writer(f)
    csv_writer.writerow(gene_names)# 15轮 每轮换一个位置的碱基

    for itertion in range(0, 16):
        seq_max = dict_sswm_max[itertion][0]
        res = []
        for gene in gene_names:
            res.append(seq_max[gene])
        csv_writer.writerow(res)
        
with open('res/max_v_change.csv', 'w', newline='') as f: # 最大迭代 每轮预测的活性值 16行10列 每个元素是一个值
    csv_writer = csv.writer(f)
    csv_writer.writerow(gene_names)

    for itertion in range(0, 16):
        v_max = dict_sswm_max[itertion][1]
        res = []
        for gene in gene_names:
            res.append(v_max[gene])
        csv_writer.writerow(res)


with open('res/min_seq_change.csv', 'w', newline='') as f: # 最小迭代 每轮序列 16行10列 每个元素是一条序列
    csv_writer = csv.writer(f)
    csv_writer.writerow(gene_names)

    for itertion in range(0, 16):
        seq_min = dict_sswm_min[itertion][0]
        res = []
        for gene in gene_names:
            res.append(seq_min[gene])
        csv_writer.writerow(res)
        
with open('res/min_v_change.csv', 'w', newline='') as f: # 最小迭代 每轮预测的活性值 16行10列 每个元素是一个值
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

   
    

