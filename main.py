import argparse
import pandas as pd
from src.Preprocess.Preprocessing import extract_accessibility_df
from src.Preprocess.Preprocessing import Filters
from src.Preprocess.Preprocessing import combine
from src.AttSioff.inference import perform_inference




if __name__ == '__main__':
    fasta_path = 'sirna_pre_process/data/TTR_mrna.fasta'


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--N', default=19, type=int)
    # parser.add_argument('--mode', default='joint', type=str, choices=['single', 'joint'])

    # args = parser.parse_args()

    # N = args.N
    # mode = args.mode
    # data_folder = '/home/somya/drugdiscoTTvery/siRNA/sirna_pre_process/data'

    # df_access = extract_accessibility_df(fasta_path, N, mode, out_dir=data_folder)
    # df_pre_processed = Filters(df_access).compute_confidence()

    df_pre_processed = combine()
    print(df_pre_processed.columns)
    df = perform_inference(df_pre_processed)


    df.to_csv('data/siRNA_inference_results.csv', index=False)
    print(f"✅ Inference results saved to sirna_pre_process/data/siRNA_inference_results.csv ({len(df)} rows)")