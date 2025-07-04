import subprocess
import os
import yaml
from sirna_csv_to_FASTA import csv_to_fasta
from run_blastn import run_blast
from process_tsv_file import process_blast_results


def load_config(config_path = "config_OT.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    

def run_blast_db_script(blast_db_prefix):
    nsq_file = blast_db_prefix + ".nsq"
    if os.path.exists(nsq_file):
        print(f"✅ BLAST DB already exists at: {blast_db_prefix}.* — skipping database creation.")
    else:
        print("▶️ BLAST DB not found — running blast_db.sh to create it...")
        try:
            result = subprocess.run(["bash", "blast_db.sh"], check=True, capture_output=True, text=True)
            print(result.stdout)
            print("✅ blast_db.sh executed successfully.")
        except subprocess.CalledProcessError as e:
            print("❌ Error running blast_db.sh:")
            print(e.stderr)
            raise


def main():
    config = load_config()
    paths = config["paths"]
    params = config["parameters"]

    # Step 1: Run the blast_db.sh script only if needed
    run_blast_db_script(paths["BLAST_DB_PREFIX"])

    # Step 2: Ask user for Top-k siRNA
    try:
        top_k_siRNA = int(input("Enter number of siRNAs to use for off-target estimation (Top-K) [e.g., 15]: "))
    except ValueError:
        print("❌ Invalid number. Please enter an integer.")
        return
    
    # Step 3: Convert CSV to FASTA
    csv_to_fasta(
        csv_path=paths["INFERENCE_CSV_PATH"],
        fasta_output_path=paths["SIRNA_FASTA_PATH"],
        column_name=params.get("column_name", "Antisense"),
        top_k= top_k_siRNA
    )

    # step 4: Run Blastn
    run_blast(
        fasta_path= paths["SIRNA_FASTA_PATH"],
        db_prefix= paths["BLAST_DB_PREFIX"] ,
        output_path= paths["BLAST_OUTPUT_TSV"]
    )

    # Step 5: Process tsv
    process_blast_results(
        blast_tsv_path=paths["BLAST_OUTPUT_TSV"],
        sirna_fasta_path=paths["SIRNA_FASTA_PATH"],
        inference_path=paths["INFERENCE_CSV_PATH"],
        siRNA_OTE_Summary_path=paths["siRNA_OTE_summary_output_path"],
        siRNA_OTE_Summary_ntseq_output_path=paths["siRNA_OTE_ntseq_added_output_path"],
        design_output_path=paths["design_output_path"]
    )

    print("✅ Entire siRNA pipeline executed successfully.")



if __name__ == "__main__":
    main()