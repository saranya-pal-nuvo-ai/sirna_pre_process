import os
import subprocess
import pandas as pd
import itertools
import math

# ─── Utility: RNAplfold wrapper ───────────────────────────────────────────────

def run_rnaplfold(fasta_file, window=80, span=40, max_unpaired=None, out_dir=None):
    """
    Runs RNAplfold and captures generated .lunp and .ps files.
    Moves and renames them into out_dir (or cwd if None).
    Returns paths to (lunp_path, ps_path).
    """
    cmd = ['RNAplfold', '-W', str(window), '-L', str(span)]
    if max_unpaired is not None:
        cmd += ['-u', str(max_unpaired)]
    print(f"▶️ Running: {' '.join(cmd)} < {fasta_file}")

    # ensure working directory
    proc = subprocess.run(
        cmd,
        stdin=open(fasta_file, 'r'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"RNAplfold failed (code {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # locate generated files
    cwd = os.getcwd()
    items = os.listdir(cwd)
    lunp_file = next((f for f in items if f.endswith('_lunp') or f.endswith('.lunp')), None)
    ps_file = next((f for f in items if f.endswith('_dp.ps') or f == 'dp.ps'), None)

    if not lunp_file:
        raise FileNotFoundError("Expected RNAplfold .lunp file not found.")
    if not ps_file:
        raise FileNotFoundError("Expected RNAplfold dp.ps file not found.")

    # define output directory
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    target_dir = out_dir or cwd

    base = os.path.splitext(os.path.basename(fasta_file))[0]
    lunp_target = os.path.join(target_dir, f"{base}_lunp")
    ps_target = os.path.join(target_dir, f"{base}_dp.ps")

    os.replace(lunp_file, lunp_target)
    os.replace(ps_file, ps_target)
    print(f"✅ Moved '{lunp_file}' → '{lunp_target}'")
    print(f"✅ Moved '{ps_file}' → '{ps_target}'")

    return lunp_target, ps_target

# ─── Parsing accessibility into DataFrame ─────────────────────────────────────


def get_complementary_rna(dna_seq):
    complement_map = str.maketrans("AUGC", "UACG")

    rna = dna_seq.replace("T", "U")
    return rna.translate(complement_map)[::-1]



def extract_accessibility_df(fasta_file, siRNA_length, mode, out_dir=None):
    """
    Runs plfold, moves outputs into out_dir, parses .lunp, returns DataFrame.
    """
    # lunp_path, _ = run_rnaplfold(fasta_file, max_unpaired=siRNA_length, out_dir=out_dir)
    lunp_path= '/home/saranya/Cleaned_Up_pipelines/sirna_pre_process/data/NM_000371.4_lunp'
    with open(fasta_file) as f:
        seq = ''.join(line.strip() for line in f if not line.startswith('>'))

    records = []
    with open(lunp_path) as fh:
        for line in fh:
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            if len(parts) < siRNA_length + 1:
                continue
            pos0 = int(parts[0]) - 1
            if pos0 + siRNA_length > len(seq):
                continue
            if mode == 'single':
                val = parts[siRNA_length]
                if val == 'NA': continue
                prob = float(val)
            else:
                nums = [float(x) for x in parts[1:siRNA_length+1] if x != 'NA']
                if not nums: continue
                prob = math.prod(nums)
            sense = seq[pos0:pos0+siRNA_length]
            antisense = get_complementary_rna(sense)
            records.append({
                'Accessibility_Prob': prob,
                'Start_Position': pos0+1,
                'Sense': sense,
                'Antisense': antisense
            })
    return pd.DataFrame(records)

# ─── Filters class ────────────────────────────────────────────────────────────

class Filters:
    def __init__(self, df, weights=None):
        self.df = df.copy()
        self.weights = weights or {'ui_tei': 3, 'reynolds': 2, 'amarzguioui': 1}
    @staticmethod
    def reynolds_score(sense):
        score=0
        gc=sense.count('G')+sense.count('C')
        if 6<=gc<=10: score+=1
        at_end=sense[14:19].count('A')+sense[14:19].count('U')+sense[14:19].count('T')
        score+=at_end
        if any(s*4 in sense for s in ['A','U','T','C','G']): score-=1
        if sense[18]=='A': score+=1
        if sense[2]=='A': score+=1
        if sense[9] in ('T','U'): score+=1
        if sense[18] in ('G','C'): score-=1
        if sense[12]=='G': score-=1
        return score
    @staticmethod
    def ui_tei_score(antisense):
        score=0
        if antisense[0] in ('A','U'): score+=1
        if antisense[18] in ('G','C'): score+=1
        if sum(b in ('A','U') for b in antisense[:7])>=4: score+=1
        gc_stretch=any(k and l>=10 for k,grp in itertools.groupby(antisense,lambda x:x in('G','C')) for l in[sum(1 for _ in grp)])
        score+= -1 if gc_stretch else 1
        return score
    @staticmethod
    def amarzguioui_score(sense):
        score=0
        gc_ratio=(sense.count('G')+sense.count('C'))/len(sense)
        if 0.32<=gc_ratio<=0.58: score+=1
        half=len(sense)//2
        if sum(b in ('A','U','T') for b in sense[half:])>sum(b in('A','U','T') for b in sense[:half]): score+=1
        if sense[0] in ('G','C'): score+=1
        if sense[5]=='A': score+=1
        if sense[18] in ('A','U','T'): score+=1
        return score
    def compute_confidence(self):
        out=[]
        tw=sum(self.weights.values())
        for _,r in self.df.iterrows():
            ui= self.ui_tei_score(r['Antisense'])
            re= self.reynolds_score(r['Sense'])
            am= self.amarzguioui_score(r['Sense'])
            ui_n=(ui+1)/5
            re_n=(re+3)/12
            am_n=am/5
            comb=(self.weights['ui_tei']*ui_n+self.weights['reynolds']*re_n+self.weights['amarzguioui']*am_n)/tw
            out.append({
                'Accessibility_Prob':r['Accessibility_Prob'],
                'Start_Position':r['Start_Position'],
                'Sense':r['Sense'],
                'Antisense':r['Antisense'],
                'Ui-Tei_Norm':ui_n,
                'Reynolds_Norm':re_n,
                'Amarzguioui_Norm':am_n,
                'Confidence_Score':comb
            })
        return pd.DataFrame(out)

# ─── Main: prompt fasta and optionally save ────────────────────────────────────



def combine():
    data_folder="/home/somya/drugdiscovery/siRNA/siRNA_Pipeline/data/"
    fasta_file=input("Enter the FASTA filename (e.g. TTR_mrna.fasta): ").strip()
    fasta_path=os.path.join(data_folder,fasta_file)
    if not os.path.isfile(fasta_path): raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    N=int(input("siRNA length (e.g.19,21,23): ").strip())
    mode=input("Mode ('single' or 'joint'): ").strip().lower()
    if mode not in('single','joint'): raise ValueError("Mode must be 'single' or 'joint'")

    # extract and place plfold outputs in data_folder
    df_access=extract_accessibility_df(fasta_path,N,mode,out_dir=data_folder)
    df_final=Filters(df_access).compute_confidence()

    save=input("Save output CSV? (yes/no): ").strip().lower()
    if save in('yes','y'):
        out_csv=os.path.join(data_folder,'filter_score.csv')
        df_final.to_csv(out_csv,index=False)
        print(f"✅ Saved CSV to {out_csv} ({len(df_final)} rows)")
    else:
        # print(df_final.to_string(index=False))
        return df_final  # return DataFrame for further processing if needed
    
    

if __name__=="__main__":
    combine()
    
