import RNA
from sirna_pre_process.utils import load_fasta


def main():
    # Set sequence and parameters
    seq_path = '/home/saranya/Cleaned_Up_pipelines/sirna_pre_process/data/TTR_mrna.fasta'
    seq = load_fasta(seq_path)

    # print(seq)

    md = RNA.md()
    md.window_size = 100        # -W
    md.max_bp_span  = 50        # -L
    fc = RNA.fold_compound(seq, md, RNA.OPTION_WINDOW)
    print(fc)

    # # (Optional) rescale energies for numeric stability
    # ss, mfe = fc.mfe()
    # fc.exp_params_rescale(mfe)

    # # Key step!  build partition-function matrices
    # fc.pf()

    # # --- unpaired probabilities for all 5-nt windows (-u 5) ---------------------
    # ups = fc.pu_probs(5, RNA.PU_PROBS_DEFAULT)   # list of floats, length = len(seq)-4

    # # Pretty-print
    # for i, p in enumerate(ups, start=1):
    #     print(f"{i:>3}-{i+4:<3}  {p:.4f}")



if __name__ == '__main__':
    main()
