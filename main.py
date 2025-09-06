import argparse
import pandas as pd
from src.Preprocess.Preprocessing import Filters
from src.Preprocess.Preprocessing import combine
from src.AttSioff.inference import perform_inference
from src.AttSioff.utils import calculate_Tm, get_gc_percentage, free_energy_5_end, gc_ratio_5_3, free_energy_3_end


if __name__ == '__main__':
    # fasta_path = 'sirna_pre_process/data/TTR_mrna.fasta'

    df , fasta_file, length= combine()
    fasta_name= fasta_file.split('.')[0]  
    print(df.columns)

    if length==19:
        df = perform_inference(df)

    df['Tm_value'] = df['Antisense'].apply(calculate_Tm)
    df['GC_content'] = df['Antisense'].apply(
    lambda seq: get_gc_percentage(seq)[0][0]
    )
    df['seed_Tm'] = df['Antisense'].apply(calculate_Tm, start_pos=1, end_pos=9)
    df['free_energy_5_end'] = df.apply(
    lambda row: free_energy_5_end(row['Antisense'], row['Sense'], n=4),
    axis=1
    )
    df['free_energy_3_end'] = df.apply(
    lambda row: free_energy_3_end(row['Antisense'], row['Sense'], n=4),
    axis=1
    )
    df['gc_ratio_5_3'] = df.apply(
    lambda row: gc_ratio_5_3(row['Antisense'], n=4),
    axis=1
    )



    df.to_csv(f'output_csv/{fasta_name}_siRNA_{length}.csv', index=False)

    print(f"✅ Inference results saved to sirna_pre_process/data/{fasta_name}_results.csv ({len(df)} rows)")