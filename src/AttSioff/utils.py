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
from typing import Tuple, Optional



def calculate_Tm(sequence: str,
                 strand_conc: float = 2.0e-4,
                 start_pos: int = 0,
                 end_pos: Optional[int] = None) -> float:
    """
    Melting temperature (°C) of an RNA duplex segment using the
    INN-HB nearest-neighbor model (Xia et al., 1998; 1 M NaCl, pH 7).

    Parameters
    ----------
    sequence   : full 5'→3' RNA (A,U,G,C; T→U is accepted)
    strand_conc: total single-strand concentration (M), default 0.2 mM
    start_pos  : 0-based index of first base to include (default 0)
    end_pos    : index *after* the last base (Python-slice style).
                 Default = len(sequence).

    Notes
    -----
    • Validated lengths: 4–10 nt (paper); practical use up to 23 nt
      is common but unverified.
    • For internal windows the result is for an isolated mini-duplex.

    Warning
    -------
    The INN-HB parameters are rock-solid for 4–10 bp duplexes in 1 M NaCl.
    For full-length siRNA (19–23 bp) or for internal 6–8 bp windows you obtain a reasonable first approximation, but there is no experimental proof the ±1.3 °C error still holds.
    
    """


    seq_full = sequence.upper().replace('T', 'U')
    end_pos = len(seq_full) if end_pos is None else end_pos

    if not 0 <= start_pos < end_pos <= len(seq_full):
        raise ValueError("start_pos/end_pos indices out of range")
    seq = seq_full[start_pos:end_pos]
    if len(seq) < 4:
        raise ValueError("Sub-duplex must be at least 4 nt long")
    if any(b not in "AUGC" for b in seq):
        raise ValueError("Invalid nucleotide in sequence")



    R = 1.987  # cal K⁻¹ mol⁻¹
    NN = {     # ΔH (kcal mol⁻¹), ΔS (cal K⁻¹ mol⁻¹)
        "AA/UU": (-6.82, -19.0), "AU/UA": (-9.38, -26.7),
        "UA/AU": (-7.69, -20.5), "CU/GA": (-10.48, -27.1),
        "CA/GU": (-10.44, -26.9), "GU/CA": (-11.40, -29.5),
        "GA/CU": (-12.44, -32.5), "CG/GC": (-10.64, -26.7),
        "GG/CC": (-13.39, -32.7), "GC/CG": (-14.88, -36.9),
    }
    INIT   = (+3.61,  -1.5)   
    TERM_AU = (+3.72, +10.5) 


    comp = {"A":"U","U":"A","G":"C","C":"G"}

    def dimer_key(x: str, y: str) -> str:
        """Return a key that is guaranteed to be in NN."""
        key = f"{x}{y}/{comp[x]}{comp[y]}"
        if key in NN:
            return key
     
        key_rc = f"{comp[y]}{comp[x]}/{y}{x}"
        if key_rc in NN:
            return key_rc
        raise ValueError(f"Unrecognised dimer {x}{y}")
 


    def terminal_au_penalty(s: str) -> int:
        return int(s[0] in "AU") + int(s[-1] in "AU")


    dH = dS = 0.0
    for i in range(len(seq) - 1):
        key = dimer_key(seq[i], seq[i+1])
        try:
            h, s = NN[key]
        except KeyError:
            raise ValueError(f"Unrecognized dimer {key}")
        dH += h; dS += s

   

    dH += INIT[0] + terminal_au_penalty(seq) * TERM_AU[0]
    dS += INIT[1] + terminal_au_penalty(seq) * TERM_AU[1]
    Tm_K = (dH*1000.0) / (dS + R*math.log(strand_conc/4.0))

    return Tm_K - 273.15




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


def gibbs_energy(seq):  # 20    #    

    energy_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'table': np.array(
        [[-0.93, -2.24, -2.08, -1.1],
         [-2.11, -3.26, -2.36, -2.08],
         [-2.35, -3.42, -3.26, -2.24],
         [-1.33, -2.35, -2.11, -0.93]])}
    

    result = []
    for i in range(len(seq) - 1): 
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