import numpy as np
import Bio.SeqIO
import torch
from captum.attr import DeepLiftShap
from adv_model import Net
from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence
from itertools import islice

device = torch.device('cuda:1')

model = Net()
model.load_state_dict(torch.load('model_params.pkl')) # 3x6 18
model.to(device)
model.eval()


def seq2tensor(seq, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    encode_seq = []
    for element in seq.upper():
        encode_seq.append(one_hot[element])
    return torch.tensor(encode_seq, dtype=torch.float, requires_grad=True, device=device).t().unsqueeze(dim=0)


def seqs2tensor(seqs, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}

    seqs_tensor = []
    for seq in seqs:
        encode_seq = []
        for element in seq.upper():
            encode_seq.append(one_hot[element])
        seqs_tensor.append(torch.tensor(encode_seq, dtype=torch.float, device=device).t().unsqueeze(dim=0))
    return torch.cat(seqs_tensor, 0)


list_res = [] # 放任意条 170bp
with open(r'../sswm_res/max_seq_change.csv', 'r') as f: # 换成序列dict
    # for line in islice(f, 1, None):
    for line in f:
        list_res.append(line[:-1].split(','))


seqs = np.array(list_res).T


for res in seqs:
    gene = res[0]

    for iters, seq in enumerate(res[1:]):
        seq = str(seq)
        seq_tensor = seq2tensor(seq)

        model.zero_grad()
        dls = DeepLiftShap(model, multiply_by_inputs=False)
        grads = dls.attribute(seq_tensor, baselines=seqs2tensor(dinuc_shuffle(seq, num_shufs=100, rng=np.random.RandomState(123))))
        list_grads = (grads * seq_tensor).detach().cpu().squeeze(dim=0).numpy()
        viz_sequence.plot_weights_yxs(list_grads, subticks_frequency=20, savewhere='./res/'+gene+'_iter_'+ str(iters) + '.png')


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
