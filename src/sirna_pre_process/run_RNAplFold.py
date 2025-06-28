import argparse
import pandas as pd


# Mapping dictionary for RNA complements (A<->U, C<->G)
complement_map = str.maketrans("AUGC", "UACG")


def load_fasta(filename):
    """Loads sequence from a FASTA file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    return ''.join([line.strip() for line in lines if not line.startswith('>')])


def get_complementary_rna(dna_seq):
    """Returns the reverse complementary antisense RNA strand."""
    rna_seq = dna_seq.replace("T", "U")
    complement = rna_seq.translate(complement_map)
    return complement[::-1]


def extract_all_accessible_regions_with_sirna(seq_file, lunp_file, sirna_length, mode):
    """
    Extracts siRNA-sized mers with accessibility probabilities and antisense siRNAs.
    Returns a pandas DataFrame.
    """

    if mode not in ['single', 'average']:
        raise ValueError("❌ Invalid mode. Choose either 'single' or 'average'.")

    column_index = sirna_length  # column 1 is position 1

    seq = load_fasta(seq_file)
    results = []

    with open(lunp_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue  # skip comments
            parts = line.strip().split()
            if len(parts) < sirna_length + 1:
                continue  # skip incomplete lines

            try:
                pos = int(parts[0]) - 1
                if len(seq) - pos < sirna_length:
                    continue  # insufficient sequence

                # Calculate accessibility
                if mode == 'single':
                    col_val = parts[column_index]
                    if col_val == 'NA':
                        continue
                    prob = float(col_val)
                elif mode == 'average':
                    nums = [float(p) for p in parts[1:sirna_length+1] if p != 'NA']
                    if not nums:
                        continue
                    prob = sum(nums) / len(nums)

                region = seq[pos:pos + sirna_length]
                antisense_sirna = get_complementary_rna(region)
                results.append((prob, pos + 1, region, antisense_sirna))

            except ValueError:
                continue

    df = pd.DataFrame(results, columns=['Accessibility_Prob', 'Start_Position', 'Sense', 'Antisense'])
    print(f"✅ Extracted {len(df)} siRNA regions | mode: {mode} | siRNA length: {sirna_length}")
    return df





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract accessible siRNA regions with antisense strands.")

    parser.add_argument('--fasta_path', default='data/TTR_mrna.fasta', type=str, help='Path to the FASTA file')
    parser.add_argument('--lunp_file', required=True, type=str, help='Path to the RNAplFold .lunp file')
    parser.add_argument('--sirna_length', default=19, type=int, help='Length of siRNA (e.g., 19, 21, 23)')
    parser.add_argument('--mode', default='average', choices=['single', 'average'], help="Mode: 'single' or 'average'")
    parser.add_argument('--output_csv', default='RNAplFold_score.csv', type=str, help='Output CSV filename')
    
    args = parser.parse_args()

    df = extract_all_accessible_regions_with_sirna(
        seq_file=args.fasta_path,
        lunp_file=args.lunp_file,
        sirna_length=args.sirna_length,
        mode=args.mode
    )

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"💾 Saved DataFrame to: {args.output_csv}")
    else:
        print(df.head())