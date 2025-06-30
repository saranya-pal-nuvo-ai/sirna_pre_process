import pandas as pd
df = pd.read_csv("../results/sirna_blast_results.tsv", sep='\t', header=None,
                 names=["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                        "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qseq", "sseq"])

print(df["length"].describe())

print(df["pident"].describe())

print(df["mismatch"].describe())
print(df["bitscore"].describe())
"""
print(df["length"].describe())


col_names = ["SNo", "Antisense", "mrna", "21_mer_Sequence", "Start_Position",
             "Accessibility_Prob", "Ui-Tei_Norm", "Reynolds_Norm", "Amarzguioui_Norm",
             "Confidence_Score", "Predicted_inhibition", "total_hits", "avg_final_score"]

df2 = pd.read_csv("../results/Final_csv_with_all_scores.csv", names=col_names, header=0)  # use header=None if no header in file
print(df2["total_hits"].describe())


fasta_path = "../data/human_all.rna.fna"

count = 0
with open(fasta_path, "r") as f:
    for line in f:
        if line.startswith(">"):
            count += 1

print(f"Total number of sequences in the FASTA file: {count}")


import pandas as pd
"""
# Load the CSV
df = pd.read_csv("../results/siRNA_OTSummary.csv")

# Ensure total_hits is numeric (in case of any parsing issues)
df["total_hits"] = pd.to_numeric(df["total_hits"], errors="coerce")

# Basic statistics
print("Statistics for 'total_hits':")
print(df["total_hits"].describe())
print(df["avg_bitscore"].describe())
print(df["avg_length"].describe())
print(df["avg_mismatch"].describe())

"""

# Load the CSV
df = pd.read_csv("../result/siRNA_OTSummary.csv")

# Ensure total_hits is numeric (in case of any parsing issues)
df["total_hits"] = pd.to_numeric(df["total_hits"], errors="coerce")

# Basic statistics
print("Statistics for 'total_hits':")
print(df["total_hits"].describe())
"""

df = pd.read_csv("../results/siRNA_Design.csv")

# Ensure total_hits is numeric (in case of any parsing issues)
df["total_off_target_hits"] = pd.to_numeric(df["total_off_target_hits"], errors="coerce")

# Basic statistics
print("Statistics for 'total_off_target_hits':")
print(df["total_off_target_hits"].describe())