import pandas as pd
import os

def csv_to_fasta(csv_path, fasta_output_path, column_name="Antisense", top_k=15):
    """
    Converts the top K siRNA sequences from a CSV to a FASTA file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        raise ValueError(f"Expected column '{column_name}' not found in CSV.")

    # Get top K siRNAs
    top_k_sirnas = df.head(top_k)[column_name].dropna()

    # Write to FASTA
    with open(fasta_output_path, "w") as f:
        for i, seq in enumerate(top_k_sirnas, 1):
            cleaned_seq = seq.strip().upper().replace("U", "T")  # Ensure RNA format
            f.write(f">siRNA_{i}\n{cleaned_seq}\n")

    print(f"✅ Wrote top {top_k} siRNAs to FASTA file: {fasta_output_path}")
