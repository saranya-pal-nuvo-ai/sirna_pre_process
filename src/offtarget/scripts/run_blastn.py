import os
import subprocess

def run_blast(fasta_path, db_prefix, output_path):
    """
    Runs BLASTN using the given FASTA file and BLAST DB.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("▶️ Running BLASTN alignment...")


    cmd = [
        "blastn",
        "-task", "blastn-short",
        "-query", fasta_path,
        "-db", db_prefix,
        "-evalue", "1000",
        "-word_size", "6",
        "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq",
        "-out", output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ BLASTN completed successfully. Output saved to: {output_path}")
    else:
        print("❌ Error running BLASTN:")
        print(result.stderr)
        raise RuntimeError("BLASTN failed.")
