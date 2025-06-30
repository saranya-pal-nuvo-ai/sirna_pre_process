#!/bin/bash

# Define paths
INPUT_FASTA="../data/human_all.rna.fna"
BLAST_DB_DIR="../H_Sapien_blast_db_1"
BLAST_DB_PREFIX="$BLAST_DB_DIR/H_Sapien_mrna_db"

# Create output directory if it doesn't exist
mkdir -p "$BLAST_DB_DIR"

# Step 1: Make BLAST database (if not already created)
echo "Checking BLAST database..."

if [ -f "$BLAST_DB_PREFIX.nsq" ]; then
  echo "BLAST DB already exists at $BLAST_DB_PREFIX.* — skipping makeblastdb."
else
  echo "Creating new BLAST DB at $BLAST_DB_PREFIX.*"
  makeblastdb \
    -in "$INPUT_FASTA" \
    -dbtype nucl \
    -out "$BLAST_DB_PREFIX"
  echo "✅ BLAST database created."
fi
