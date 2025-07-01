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
    # print(df_pre.head())
    df_pre_process = Filters(df_pre).compute_confidence()
    # print('-'*20)
    # print(df_pre_process.columns)


    df = perform_inference(df_pre_process, mRNA_seq, MODELS_DIR, CACHE_PATH)

    merged_df = df.copy()
    merged_df['Start_Position'] = df_pre_process['Start_Position']
    merged_df['Accessibility_Prob'] = df_pre_process['Accessibility_Prob']
    merged_df['Ui_Tei_Norm'] = df_pre_process['Ui-Tei_Norm']
    merged_df['Reynolds_Norm'] = df_pre_process['Reynolds_Norm']
    merged_df['Amarzguioui_Norm'] = df_pre_process['Amarzguioui_Norm']
    merged_df['Confidence_Score'] = df_pre_process['Confidence_Score']

    merged_df = merged_df[['Antisense', 'Start_Position', 'Accessibility_Prob', 'Ui_Tei_Norm', 'Reynolds_Norm', 'Amarzguioui_Norm', 'Confidence_Score', 'Predicted_inhibition', 'GC Percent', 'Tm_value']]
    merged_df = merged_df.sort_values(by='Predicted_inhibition', ascending=False)

    # print(df.shape)
    print(merged_df.head(10))

    merged_df.to_csv(OUTPUT_DIR + "/" + "ALAS1_v4.csv")