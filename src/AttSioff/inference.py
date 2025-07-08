import os
import fm
import torch
# import RNA
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F
from src.AttSioff.model import RNAFM_SIPRED_2
from torch.nn.utils.rnn import pad_sequence
from src.AttSioff.utils import get_gc_sterch, get_gc_percentage, get_single_comp_percent, get_di_comp_percent, get_tri_comp_percent
from src.AttSioff.utils import secondary_struct, score_seq_by_pssm, gibbs_energy, create_pssm, calculate_Tm, free_energy_5_end



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pad_and_stack(list_of_ndarrays):
    """
    Convert a list of [L_i, D] numpy arrays (variable‐length sequences)
    into a single tensor of shape [B, L_max, D] with zero-padding.
    """
    tensors = [torch.as_tensor(x, dtype=torch.float32) for x in list_of_ndarrays]
    return pad_sequence(tensors, batch_first=True)          # [B, L_max, D]




def build_model_inputs(
        idx: int,
        sirna_embed_tensor: torch.Tensor,   # expected shape [1, L_sirna, 640]
        mrna_embed_tensor : torch.Tensor,   # expected shape [1, L_mrna , 640]
        prior_feats       : dict,
        device            : torch.device,
    ):
    """
    Assemble ONE sample in the exact format used during training.

    All handcrafted features are converted to shape [1, N, 1] so that
    `torch.cat([...], dim=1).squeeze(2)` in the model works without
    dimension mismatches.
    """


    def to_column(vec):
        t = torch.as_tensor(vec, dtype=torch.float32, device=device).flatten()  # [N]
        t = t.unsqueeze(0).unsqueeze(-1)  # → [1, N, 1]
        return t


    sample = {
        "rnafm_encode"       : sirna_embed_tensor.to(device),     # [1, L_sirna, 640]
        "rnafm_encode_mrna"  : mrna_embed_tensor.to(device),      # [1, L_mrna , 640]

        "gc_sterch"          : to_column(prior_feats["gc_sterch"]),        # [1, 20, 1]
        "gc_content"         : to_column(prior_feats["gc_percent"]),       # [1,  1, 1]
        "single_nt_percent"  : to_column(prior_feats["single_com"]),       # [1,  4, 1]
        "di_nt_percent"      : to_column(prior_feats["di_com"]),           # [1, 16, 1]
        "tri_nt_percent"     : to_column(prior_feats["tri_com"]),          # [1, 64, 1]
        "pssm_score"         : to_column([prior_feats["pssm_score"]]),     # [1,  1, 1]
        "sirna_gibbs_energy" : to_column(prior_feats["gibbs"]),            # [1, 20, 1]

        "sirna_second_percent": to_column(
            prior_feats["second_struct"][0]                                  # length-2 vector
            if isinstance(prior_feats["second_struct"], tuple)
            else prior_feats["second_struct"]),
        "sirna_second_energy" : to_column(
            prior_feats["second_struct"][1]
            if isinstance(prior_feats["second_struct"], tuple)
            else 0.0),    # fallback if not provided
    }

    return sample




#   Code for Embeding generation and Integration
def check_model_cache(CACHE_PATH) -> str:
    """
    Check if RNA-FM model is available in cache.

    Returns:
     Path of the model stored in cache (so that we dont need to download everytime)
    """

    cache_path = Path.home() / CACHE_PATH
    
    if cache_path.exists():
        return str(cache_path)
    else:
        return None
   



def load_rna_fm_fast(model_path=None):
    torch.serialization.add_safe_globals([argparse.Namespace])

    if model_path and os.path.exists(model_path):
        model, alphabet = fm.pretrained.load_model_and_alphabet_local(
            model_path,
            theme="rna"
        )
    else:
        print("Model Downloading....")
        model, alphabet = fm.pretrained.rna_fm_t12()

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter




def generate_embeddings(seq_list, model, alphabet, batch_converter):
    modified_seq = []
    for i, seq in enumerate(seq_list, 1):
        modified_seq.append( (str(i), seq ))

    model.eval()
    batch_labels, batch_strs, batch_tokens = batch_converter(modified_seq)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])

    token_embeddings = results["representations"][12]
    token_embeddings_np = token_embeddings.cpu().numpy()
    num_seq = token_embeddings.shape[0]

    final_embeddings = []
    for i in range(num_seq):
        final_embeddings.append(token_embeddings_np[i])

    return final_embeddings





def input_to_inference(inp_df, mRNA_seq):
    mrna_seq = ""

    def extract_segment(start_pos: int) -> str:
        """Return 20 nt upstream + 19 nt site + 20 nt downstream (59 nt total)."""
        i = start_pos - 1                      # convert to 0-based index
        return mrna_seq[i - UP_LEN : i + SI_LEN + DOWN_LEN]
    

    for nt in mRNA_seq:
        if nt == 'T':
            mrna_seq += 'U'
        else:
            mrna_seq += nt

    UP_LEN   = 19          # upstream context
    SI_LEN   = 19          # antisense-binding length
    DOWN_LEN = 19          # downstream context
    SEG_LEN  = UP_LEN + SI_LEN + DOWN_LEN   # 59
    MRNA_LEN = len(mrna_seq)


    valid_starts = (
    inp_df["Start_Position"]
      .between(UP_LEN + 1, MRNA_LEN - SI_LEN - DOWN_LEN)  # 1-based!
    )
    df = inp_df.loc[valid_starts].copy()    

    df["mrna"] = df["Start_Position"].apply(extract_segment)
    df = df[['Sense', 'Antisense', 'mrna', 'Start_Position', 'Accessibility_Prob', 'Ui-Tei_Norm', 'Reynolds_Norm', 'Amarzguioui_Norm', 'Confidence_Score']]
    # df = df.rename(columns={'Antisense_siRNA': 'Antisense'})

    return df, UP_LEN, SI_LEN, DOWN_LEN, MRNA_LEN






def load_RNAFM_and_data(data_pre, mRNA_seq, CACHE_PATH):    

    data, UP_LEN, SI_LEN, DOWN_LEN, MRNA_LEN = input_to_inference(data_pre, mRNA_seq)

    cache_model_path = check_model_cache(CACHE_PATH)
    model, alphabet, batch_converter = load_rna_fm_fast(cache_model_path)
    
    model.eval()
    sense_seq = data['Sense'].to_list()
    siRNA_seq, mRNA_seq = data['Antisense'].to_list(), data['mrna'].to_list()
    # mRNA_seq = ['ACAGAAGTCCACTCATTCTTGGCAGGATGGCTTCTCATCGTCTGCTCCTCCTCTGCCTTGCTGGACTGGTATTTGTGTCTGAGGCTGGCCCTACGGGCACCGGTGAATCCAAGTGTCCTCTGATGGTCAAAGTTCTAGATGCTGTCCGAGGCAGTCCTGCCATCAATGTGGCCGTGCATGTGTTCAGAAAGGCTGCTGATGACACCTGGGAGCCATTTGCCTCTGGGAAAACCAGTGAGTCTGGAGAGCTGCATGGGCTCACAACTGAGGAGGAATTTGTAGAAGGGATATACAAAGTGGAAATAGACACCAAATCTTACTGGAAGGCACTTGGCATCTCCCCATTCCATGAGCATGCAGAGGTGGTATTCACAGCCAACGACTCCGGCCCCCGCCGCTACACCATTGCCGCCCTGCTGAGCCCCTACTCCTATTCCACCACGGCTGTCGTCACCAATCCCAAGGAATGAGGGACTTCTCCTCCAGTGGACCTGAAGGACGAGGGATGGGATTTCATGTAACCAAGAGTATTCCATTTTTACTAAAGCAGTGTTTTCACCTCATATGCTATGTTAGAAGTCCAGGCAGAGACAATAAAACATTCCTGTGAAAGGCA']

    siRNA_embeddings = generate_embeddings(siRNA_seq, model, alphabet, batch_converter)
    mRNA_embeddings = generate_embeddings(mRNA_seq, model, alphabet, batch_converter)

    return sense_seq, siRNA_seq, siRNA_embeddings, mRNA_embeddings, UP_LEN, SI_LEN, DOWN_LEN, MRNA_LEN






def prepare_prior_knowledge_features(sirna_seq_list):

    gc_sterch, gc_precent = [], []
    single_com, di_com, tri_com = [], [], []
    second_struct = []
    pssm_score = []
    gibbs = []

    pssm = create_pssm(sirna_seq_list)

    for sirna in sirna_seq_list:

        gc_sterch.append(get_gc_sterch(sirna))
        gc_precent.append(get_gc_percentage(sirna))

        single_com.append(get_single_comp_percent(sirna))
        di_com.append(get_di_comp_percent(sirna))
        tri_com.append(get_tri_comp_percent(sirna))

        second_struct.append(secondary_struct(sirna))
        pssm_score.append(float(score_seq_by_pssm(pssm, sirna)))
        gibbs.append(gibbs_energy(sirna))


    return gc_sterch, gc_precent, single_com, di_com, tri_com, second_struct, pssm_score, gibbs


def process_siRNA_embeds(siRNA_embeddings):
    return pad_and_stack(siRNA_embeddings)


def process_mRNA_embeds(mRNA_embeddings):
    return pad_and_stack(mRNA_embeddings)




#   MAIN INFERENCE SECTION

def perform_inference(df_pre, mRNA_seq, MODEL_PATH, CACHE_PATH):

    # NUM_EXAMPLES = 1000

    # CAN SWITCH TO ANY OF THE SAVED 8 WEIGHTS OF THE MODEL
    model = RNAFM_SIPRED_2(dp=0.1, device=device).to(torch.float32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
    model.eval()

    print(f"Loaded model weights from {MODEL_PATH}")
    
    # siRNA_seq => AS_seq
    sense_seq, siRNA_seq, siRNA_embeds, mRNA_embeds, UP_LEN, SI_LEN, DOWN_LEN, MRNA_LEN = load_RNAFM_and_data(df_pre, mRNA_seq, CACHE_PATH)
    sirna_embed_tensor = process_siRNA_embeds(siRNA_embeds)   # [B, L_max1, D]
    mrna_embed_tensor  = process_mRNA_embeds(mRNA_embeds)    # [B, L_max2, D]
    siRNA_seq = siRNA_seq


    gc_sterch, gc_precent, single_com, di_com, tri_com, second_struct, pssm_score, gibbs = prepare_prior_knowledge_features(siRNA_seq)

    print("Embeddings converted and all prior knowledge features calculated")

    print("Inference starting......")

    preds = []
    with torch.no_grad():
        for i in tqdm(range(len(siRNA_seq)), desc="Inferencing"):
            prior_dict = {
                "gc_sterch"   : gc_sterch[i],
                "gc_percent"  : gc_precent[i],
                "single_com"  : single_com[i],
                "di_com"      : di_com[i],
                "tri_com"     : tri_com[i],
                "second_struct": second_struct[i],
                "pssm_score"  : pssm_score[i],
                "gibbs"       : gibbs[i],
            }

            inp = build_model_inputs(
                idx=i,
                sirna_embed_tensor=sirna_embed_tensor[i].unsqueeze(0),
                mrna_embed_tensor=mrna_embed_tensor[i].unsqueeze(0),
                prior_feats=prior_dict,
                device=device,
            )

            score = model(inp)          
            preds.append(score.item())


    valid_idx = df_pre["Start_Position"].between(
        UP_LEN + 1,                                  # 20  (1-based)
        MRNA_LEN - SI_LEN - DOWN_LEN                 # MRNA_LEN-38
    )

    meta_df = (
        df_pre.loc[valid_idx]        # drop rows outside valid window
            .reset_index(drop=True)
            .iloc[:len(siRNA_seq)] # guard against any extra trimming
    )

    out_df = pd.DataFrame({
        "Sense": sense_seq,
        "Antisense": siRNA_seq,
        "Start_Position": meta_df['Start_Position'],
        "Accessibility_Prob": meta_df['Accessibility_Prob'],
        "Ui_Tei_Norm": meta_df['Ui-Tei_Norm'],
        "Reynolds_Norm": meta_df['Reynolds_Norm'],
        "Amarzguioui_Norm": meta_df['Amarzguioui_Norm'],
        "Confidence_Score": meta_df['Confidence_Score'],
        "Predicted_inhibition": preds,
        "GC Percent": gc_precent
    })


    #   Compute Biological Factors....
    out_df['GC Percent'] = (
        out_df['GC Percent']
        .apply(lambda v: float(np.squeeze(v)))   # np.squeeze removes all singleton dims/lists
    ) * 100

    out_df['Tm_value'] = out_df['Antisense'].apply(calculate_Tm)
    out_df["Tm_seed_2_8"] = out_df["Antisense"].apply(
        lambda seq: calculate_Tm(seq, start_pos=1, end_pos=9)
    )


    out_df['Free_energy_5_end'] = out_df.apply(
        lambda row: free_energy_5_end(row['Antisense'], row['Sense'], n=4),
        axis=1
    )

    out_df = out_df.drop(columns='Sense', axis=1)
    out_df = out_df.sort_values(by='Predicted_inhibition', ascending=False)

    print("Inference complete")
    

    return out_df





if __name__ == '__main__':

    load_dotenv()
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")

    df = perform_inference()      ##### THERE IS AN ERROR IN THE LINE (function params not passed....)
    df.to_csv(OUTPUT_DIR / "inference_results.csv", index=False)
    print(f"Saved inference results to {OUTPUT_DIR / 'inference_results.csv'}")
