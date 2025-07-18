import argparse
import pandas as pd
from src.Preprocess.Preprocessing import extract_accessibility_df
from src.Preprocess.Preprocessing import Filters
from src.Preprocess.Preprocessing import combine
from src.AttSioff.inference import perform_inference
from src.offtarget.scripts.offtarget_main import offtarget
from src.AttSioff.utils import calculate_Tm, get_gc_percentage


if __name__ == '__main__':
    # fasta_path = 'sirna_pre_process/data/TTR_mrna.fasta'

    df_pre_processed , fasta_file = combine()
    fasta_name= fasta_file.split('.')[0]  # Extract the name without extension
    print(df_pre_processed.columns)
    df = perform_inference(df_pre_processed)

    df.to_csv(f'data/{fasta_name}_results.csv', index=False)


    # Run off-target analysis
    df_offtarget=offtarget(inference_csv_path=f'data/{fasta_name}_results.csv')

    df_offtarget['Tm_value'] = df_offtarget['Antisense'].apply(calculate_Tm)
    df_offtarget['GC_content'] = df_offtarget['Antisense'].apply(get_gc_percentage)

    df_offtarget.to_csv(f'output_csv/{fasta_name}_siRNA_Design.csv', index=False)

    print(f"✅ Inference results saved to sirna_pre_process/data/{fasta_name}_results.csv ({len(df)} rows)")