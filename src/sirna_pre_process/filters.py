import argparse
import itertools
import pandas as pd
from sirna_pre_process.run_RNAplFold import extract_all_accessible_regions_with_sirna

class Filters:
    def __init__(self, df, weights=None):
        self.df = df.copy()
        self.weights = weights or {'ui_tei': 3, 'reynolds': 2, 'amarzguioui': 1}

    @staticmethod
    def reynolds_score(sense):
        score = 0
        gc_count = sense.count("G") + sense.count("C")
        if 6 <= gc_count <= 10:
            score += 1
        terminal_at_count = sense[14:19].count("A") + sense[14:19].count("U") + sense[14:19].count("T")
        score += terminal_at_count
        if any(stretch in sense for stretch in ["AAAA", "TTTT", "UUUU", "CCCC", "GGGG"]):
            score -= 1
        if sense[18] == "A":
            score += 1
        if sense[2] == "A":
            score += 1
        if sense[9] in ("T", "U"):
            score += 1
        if sense[18] in ("G", "C"):
            score -= 1
        if sense[12] == "G":
            score -= 1
        return score

    @staticmethod
    def ui_tei_score(antisense):
        score = 0
        if antisense[0] in ('A', 'U'):
            score += 1
        if antisense[18] in ('G', 'C'):
            score += 1
        if sum(1 for b in antisense[0:7] if b in ('A', 'U')) >= 4:
            score += 1
        gc_stretch = False
        for is_gc, group in itertools.groupby(antisense, lambda x: x in ('G','C')):
            if is_gc and sum(1 for _ in group) >= 10:
                gc_stretch = True
                break
        score += 1 if not gc_stretch else -1
        return score

    @staticmethod
    def amarzguioui_score(sense):
        score = 0
        gc = (sense.count('G') + sense.count('C')) / len(sense)
        if 0.32 <= gc <= 0.58:
            score += 1
        half = len(sense) // 2
        au_first = sum(1 for b in sense[:half] if b in ('A', 'U', 'T'))
        au_second = sum(1 for b in sense[half:] if b in ('A', 'U', 'T'))
        if au_second > au_first:
            score += 1
        if sense[0] in ('G', 'C'):
            score += 1
        if sense[5] == 'A':
            score += 1
        if sense[18] in ('A', 'U', 'T'):
            score += 1
        return score

    def compute_confidence(self):
        results = []
        total_weight = sum(self.weights.values())
        for _, row in self.df.iterrows():
            ui = self.ui_tei_score(row['Antisense'])
            re = self.reynolds_score(row['Sense'])
            am = self.amarzguioui_score(row['Sense'])
            # Normalize: (score - min) / (max - min)
            ui_norm = (ui - (-1)) / (4 - (-1))  # min -1, max 4
            re_norm = (re - (-3)) / (9 - (-3))  # min -3, max 9
            am_norm = (am - 0) / (5 - 0)        # min 0, max 5

            combined = (
                self.weights['ui_tei'] * ui_norm +
                self.weights['reynolds'] * re_norm +
                self.weights['amarzguioui'] * am_norm
            ) / total_weight

            results.append({
                'Accessibility_Prob': row.get('Accessibility_Prob'),
                'Start_Position': row.get('Start_Position'),
                '21_mer_Sequence': row['Sense'],
                'Antisense_siRNA': row['Antisense'],
                'Ui-Tei_Norm': ui_norm,
                'Reynolds_Norm': re_norm,
                'Amarzguioui_Norm': am_norm,
                'Confidence_Score': combined
            })
        return pd.DataFrame(results)
    
    

if __name__ == "__main__":
    input_file = "RNAplFold_score.csv"
    # df = pd.read_csv(input_file)

    parser = argparse.ArgumentParser(description="Extract accessible siRNA regions with antisense strands.")

    parser.add_argument('--fasta_path', default='data/TTR_mrna.fasta', type=str, help='Path to the FASTA file')
    parser.add_argument('--lunp_file', required=True, type=str, help='Path to the RNAplFold .lunp file')
    parser.add_argument('--sirna_length', default=19, type=int, help='Length of siRNA (e.g., 19, 21, 23)')
    parser.add_argument('--mode', default='average', choices=['single', 'average'], help="Mode: 'single' or 'average'")

    args = parser.parse_args()

    df = extract_all_accessible_regions_with_sirna(
        seq_file=args.fasta_path,
        lunp_file=args.lunp_file,
        sirna_length=args.sirna_length,
        mode=args.mode
    )

    filters = Filters(df)
    output_df = filters.compute_confidence()

    return output_df

    # output_file = "filter_score.csv"
    # output_df.to_csv(output_file, index=False)
    # print(f"Filter scores successfully saved to '{output_file}'")
