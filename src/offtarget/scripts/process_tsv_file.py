# process_tsv_file.py

import pandas as pd
from Bio import SeqIO
import os

def process_blast_results(
    blast_tsv_path,
    sirna_fasta_path,
    inference_path,
    siRNA_OTE_Summary_path,
    siRNA_OTE_Summary_ntseq_output_path,
    design_output_path
):
    # Step 1: Load BLAST TSV
    df = pd.read_csv(blast_tsv_path, sep='\t', header=None,
                     names=["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                            "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qseq", "sseq"])

    # Step 2: Filter for reverse strand matches
    df["strand"] = df.apply(lambda row: "-" if row["sstart"] > row["send"] else "+", axis=1)
    df_filtered = df[df["strand"] == "-"].copy()

    # Step 3: Save filtered OT summary
    summary = df_filtered.groupby('qseqid').agg(
        total_hits=('sseqid', 'count'),
        avg_bitscore=('bitscore', 'mean'),
        max_bitscore=('bitscore', 'max'),
        avg_length=('length', 'mean'),
        avg_mismatch=('mismatch', 'mean')
    ).reset_index()
    summary = summary.sort_values(by='total_hits', ascending=True)
    # summary.to_csv(siRNA_OTE_Summary_path, index=False)

    # Step 4: Extract sequences from FASTA
    seq_dict = {}
    for record in SeqIO.parse(sirna_fasta_path, "fasta"):
        seq_dict[record.id] = str(record.seq).replace("T", "U")

    # Step 5: Map sequence to OT summary
    df_ot = df_filtered.groupby('qseqid').agg(
        total_off_target_hits=('sseqid', 'count'),
        avg_bitscore=('bitscore', 'mean'),
        max_bitscore=('bitscore', 'max'),
        avg_length=('length', 'mean'),
        avg_mismatch=('mismatch', 'mean')
    ).reset_index()
    df_ot['Antisense'] = df_ot['qseqid'].map(seq_dict)
    df_ot = df_ot.sort_values(by='total_off_target_hits', ascending=True)
    # df_ot.to_csv(siRNA_OTE_Summary_ntseq_output_path, index=False)

    # Step 6: Merge with inference scores
    df_infer = pd.read_csv(inference_path)
    df_final = pd.merge(df_ot, df_infer, on="Antisense", how="inner")

    # Reorder columns: move BLAST metrics to the end
    to_move = ['total_off_target_hits', 'avg_bitscore', 'max_bitscore', 'avg_mismatch', 'avg_length']
    cols = list(df_final.columns)
    new_order = [col for col in cols if col not in to_move] + to_move
    df_final = df_final[new_order]

    # Sort by start position and clean
    if 'Start_Position' in df_final.columns:
        df_final = df_final.sort_values(by='Start_Position')
    if 'SNo' in df_final.columns:
        df_final = df_final.drop(columns=['SNo'])

    df_final.to_csv(design_output_path, index=False)

    print("✅ siRNA_Design.csv created !!.")
    print("✅ All OT processing steps completed successfully.")
