import argparse
import pandas as pd
from src.preprocess.Preprocessing import extract_accessibility_df
from src.preprocess.Preprocessing import Filters
from src.AttSioff.inference import perform_inference



if __name__ == '__main__':
    fasta_path = '/home/saranya/Cleaned_Up_pipelines/sirna_pre_process/data/TTR_mrna.fasta'


    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=19, type=int)
    parser.add_argument('--mode', default='joint', type=str, choices=['single', 'joint'])

    args = parser.parse_args()

    N = args.N
    mode = args.mode
    data_folder = '/home/saranya/Cleaned_Up_pipelines/sirna_pre_process/data'

    df_access = extract_accessibility_df(fasta_path, N, mode, out_dir=data_folder)
    df_pre_processed = Filters(df_access).compute_confidence()

    df = perform_inference(df_pre_processed)

    print(df.head(10))