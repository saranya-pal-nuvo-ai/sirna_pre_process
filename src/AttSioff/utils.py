import pdb
import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import product
import math
import RNA      # Error
import sklearn
import subprocess
import os




def calculate_Tm(sequence: str, ct_molar: float = 1e-6, na_molar: float = 0.15):  # 1, 

    NN = {
    'AA': (6.82, 19.0),  'AU': (9.38, 26.7),  'UA': (7.69, 20.5),  'CA': (10.44, 26.9),
    'CU': (10.48, 27.1), 'GA': (12.44, 32.5), 'GC': (15.37, 41.2), 'GG': (15.37, 41.2),
    'GU': (11.36, 29.5), 'UG': (11.36, 29.5), 'UU': (6.82, 19.0),  'AG': (12.44, 32.5),
    'AC': (10.44, 26.9), 'CG': (14.88, 36.9), 'UC': (10.48, 27.1), 'CC': (15.37, 41.2)
    }

    R = 1.987  # cal mol-1 K-1
    INIT_H, INIT_S = 3.61, 11.6           # initiation
    AU_END_H, AU_END_S = 0.45, 1.6        # AU/UA end penalty


    seq = sequence.upper().replace('T', 'U')
    dh = INIT_H  # kcal
    ds = INIT_S  # cal/K

    # stacking contributions
    for dinuc in (seq[i:i+2] for i in range(len(seq)-1)):
        h, s = NN[dinuc]
        dh += h
        ds += s

    # AU / UA at each end
    for end in (seq[0:2], seq[-2:]):
        if end in ('AU', 'UA'):
            dh += AU_END_H
            ds += AU_END_S

    dh *= 1000  # → cal
    ln_ct = math.log(ct_molar/4.0)     # duplex, non-self-compl.
    tm_k = dh / (ds + R*ln_ct)
    tm_c = tm_k - 273.15
    # monovalent-salt correction
    tm_c += 16.6 * math.log10(na_molar)
    return round(tm_c, 2)



def readFaRNAFOLD(fa):
	'''
	加载预处理好的mrna全长的rnafold预测结果
	'''
	with open(fa,'r') as FA:
		seqName,seq='',''
		while 1:
			line=FA.readline()
			line=line.strip('\n')
			if (line.startswith('>') or not line) and seqName:
				yield ((seqName, seq, min_energy))
			if line.startswith('>'):
				line=line.split(' ')
				seqName, min_energy = line[1].strip(), line[2].strip()
				seq = ''
			else:
				seq += line
			if not line:break


def load_fasta(filename):
    """Loads sequence from a FASTA file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    return ''.join([line.strip() for line in lines if not line.startswith('>')])


def get_gc_sterch(seq):  # 1,
    max_len, tem_len = 0, 0
    for i in range(len(seq)):
        if seq[i] == 'G' or seq[i] == 'C':
            tem_len += 1
            max_len = max(max_len, tem_len)
        else:
            tem_len = 0

    result = round((max_len / len(seq)), 3)
    return np.array([result])[:, np.newaxis]

def get_gc_percentage(seq):  # 1,
    result = round(((seq.count('C') + seq.count('G')) / len(seq)), 3)
    return np.array([result])[:, np.newaxis]

def get_single_comp_percent(seq):  # 4,
    nt_percent = []
    for base_i in list(['A', 'G', 'C', 'U']):
        nt_percent.append(round((seq.count(base_i) / len(seq)), 3))
    return np.array(nt_percent)[:, np.newaxis]

def get_di_comp_percent(seq):  # 16,
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=2))
    di_nt_percent = []
    for pmt_i in pmt:
        di_nt = pmt_i[0] + pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt) / (len(seq) - 1)), 3))
    return np.array(di_nt_percent)[:, np.newaxis]

def get_tri_comp_percent(seq):  # 64,
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=3))
    tri_nt_percent = []
    for pmt_i in pmt:
        tri_nt = pmt_i[0] + pmt_i[1] + pmt_i[2]
        tri_nt_percent.append(round((seq.count(tri_nt) / (len(seq) - 2)), 3))
    return np.array(tri_nt_percent)[:, np.newaxis]


def secondary_struct(seq):  # 2+1
   
    def _percentage(if_paired):
        paired_percent = (if_paired.count('(') + if_paired.count(')')) / len(if_paired)
        unpaired_percent = (if_paired.count('.')) / len(if_paired)
        return np.array([[paired_percent], [unpaired_percent]])

    paired_seq, min_free_energy = RNA.fold(seq)
    return _percentage(paired_seq), np.array([min_free_energy])[:, np.newaxis]


def score_seq_by_pssm(pssm, seq):  # 1,
    nt_order = {'A': 0, 'G': 1, 'C': 2, 'U': 3}
    ind_all = list(range(0, len(seq)))
    scores = [pssm[nt_order[nt], i] for nt, i in zip(seq, ind_all)]
    log_score = sum([-math.log2(i) for i in scores])
    return np.array([log_score])[:, np.newaxis]

def gibbs_energy(seq):  # 20 
    energy_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'table': np.array(
        [[-0.93, -2.24, -2.08, -1.1],
         [-2.11, -3.26, -2.36, -2.08],
         [-2.35, -3.42, -3.26, -2.24],
         [-1.33, -2.35, -2.11, -0.93]])}

    result = []
    for i in range(len(seq)-1):
        index_1 = energy_dict.get(seq[i])
        index_2 = energy_dict.get(seq[i + 1])
        result.append(energy_dict['table'][index_1, index_2])

    result.append(np.array(result).sum().round(3))
    result.append((result[0] - result[-2]).round(3)) 

    result = np.array(result)[:, np.newaxis]
    return result  # / abs(result).max()


def create_pssm(train_seq):
    # train_seq = [seq.split('!') for seq in train_seq]  #通过split方式将字符串整体转为list，如果用list的话会分割字符串为单个字符
    train_seq = [list(seq.upper()) for seq in train_seq]
    train_seq = np.array(train_seq)

    nr, nc = np.shape(train_seq)
    pseudocount = nr ** 0.5  # Introduce a pseudocount (sqrt(N)) to make sure that we do not end up with a score of 0
    bases = ['A', 'G', 'C', 'U']
    pssm = []
    for c in range(0, nc):
        col_c = train_seq[:, c].tolist()
        f_A = round(((col_c.count('A') + pseudocount) / (nr + pseudocount)), 3)
        f_G = round(((col_c.count('G') + pseudocount) / (nr + pseudocount)), 3)
        f_C = round(((col_c.count('C') + pseudocount) / (nr + pseudocount)), 3)
        f_U = round(((col_c.count('U') + pseudocount) / (nr + pseudocount)), 3)
        pssm.append([f_A, f_G, f_C, f_U])
    pssm = np.array(pssm)
    pssm = pssm.transpose()
    return pssm