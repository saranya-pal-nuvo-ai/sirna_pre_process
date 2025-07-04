import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from src.preprocess.Preprocessing import extract_accessibility_df
from src.preprocess.Preprocessing import Filters
from src.AttSioff.inference import perform_inference


def load_fasta(filename):
    """Loads sequence from a FASTA file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    return ''.join([line.strip() for line in lines if not line.startswith('>')])



if __name__ == '__main__':

    load_dotenv()

    FASTA_FILE_PATH = os.getenv("FASTA_FILE_PATH")
    MODELS_DIR = os.getenv("MODELS_DIR")
    DATA_DIR = os.getenv("DATA_DIR")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    CACHE_PATH = os.getenv("CACHE_PATH")


    parser = argparse.ArgumentParser()

    parser.add_argument('--N', type=int, default=19)
    parser.add_argument('--mode', type=str, default='joint', choices=['single', 'joint'])

    args = parser.parse_args()

    mRNA_seq = load_fasta(FASTA_FILE_PATH)
    N = args.N
    mode = args.mode

    df_pre = extract_accessibility_df(FASTA_FILE_PATH, N, mode, OUTPUT_DIR)
    df_pre_process = Filters(df_pre).compute_confidence()
    df = perform_inference(df_pre_process, mRNA_seq, MODELS_DIR, CACHE_PATH)
    df = df.reset_index(drop=True)

    print(df.head(10))

    df.to_csv(OUTPUT_DIR + "/" + "SERPINC1_v7.csv", index=False)