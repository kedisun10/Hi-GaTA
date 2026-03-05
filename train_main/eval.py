"""
eval.py
"""
import os
import csv
import re
import argparse
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from peft import PeftModel

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
from pycocoevalcap.cider.cider import Cider

from HPTA import HierarchicalAggregator, TemporalPyramidPooling

hf_logging.set_verbosity_error()

# ============== defaults ==============
DEFAULT_MODEL     = " " 
DEFAULT_FEATURES  = " "  
DEFAULT_REPORTCSV = " "
HPTA_CKPT         = " "
DEFAULT_LORA_DIR  = " "
DEFAULT_OUTCSV    = " "
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = (
)

BANNED_PREFIXES = [
    r"^(this\s+(strongly\s+)?(may\s+)?(likely\s+)?(indicates|suggests|demonstrates|shows)\b.*?)([.!?])\s*",
    r"^(a\s+competent\s+approach\b.*?)([.!?])\s*",
    r"^(overall[:,]?\s*good\b.*?)([.!?])\s*",
    r"^(otherwise\s+good\s+technique\b.*?)([.!?])\s*",
]

def read_refs(csv_path: str):
    ref_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.reader(f, dialect)
        rows = list(reader)
        
        has_header = False
        if len(rows) > 0:
            first = rows[0]
            if len(first) >= 2 and (first[0].lower().startswith("video") or first[0].lower() == "id"):
                has_header = True
                
        for i, r in enumerate(rows):
            if i == 0 and has_header:
                continue
            if len(r) < 2:
                continue
            ref_map[r[0].strip()] = r[1].strip()
    return ref_map

def clean_output(text: str, max_sentences: int = 4) -> str:
    t = text.strip()
    if PROMPT_TEMPLATE in t:
        t = t.split(PROMPT_TEMPLATE)[-1].strip()
        
    for pat in BANNED_PREFIXES:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)

    sents = re.split(r'(?<=[.!?])\s+', t)
    if sents:
        s0 = re.sub(r'^[\s;:,\.\-–—]+', '', sents[0])  
        s0 = re.sub(r';{2,}', ';', s0)                
        sents[0] = s0

    out = " ".join(sents[:max_sentences]).strip()
    return out

def load_hpta_module(args, model_config, device):
    vis_dim = None
    for p in Path(args.feature_dir).glob("*.npy"):
        arr = np.load(p)
        if arr.ndim == 2:
            vis_dim = int(arr.shape[1])  
            break
        elif arr.ndim == 1:
            vis_dim = int(arr.shape[0])
            break
            
    if vis_dim is None:
        raise RuntimeError(f"Could not infer feature dimension from {args.feature_dir}")
    print(f"[Init] Inferred visual dimension D={vis_dim}")

    win_sizes = tuple(int(x) for x in args.hpta_scales.split(","))
    n_scales = len(win_sizes)
    hpta_prefix_tokens = n_scales * args.hpta_q_per_scale

    hpta_pooler = TemporalPyramidPooling(window_sizes=win_sizes, stride_factor=0.5)
    hpta = HierarchicalAggregator(
        vis_dim=vis_dim,
        hidden=model_config.hidden_size,
        n_levels=n_scales,
        queries_per_level=args.hpta_q_per_scale,
        n_heads=args.hpta_heads,
        dropout=args.hpta_dropout
    ).to(device)

    if args.hpta_ckpt and os.path.exists(args.hpta_ckpt):
        print(f"[HPTA] Loading weights from {args.hpta_ckpt} ...")
        ckpt = torch.load(args.hpta_ckpt, map_location="cpu")
        state_dict = ckpt.get("hpta", ckpt)
        
        missing, unexpected = hpta.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print(f"[WARN] HPTA missing keys: {missing}")
    else:
        print(f"[WARN] HPTA checkpoint not found at {args.hpta_ckpt}. Using random weights (Garbage Output Expected).")

    hpta.eval()
    return hpta_pooler, hpta, hpta_prefix_tokens

@torch.no_grad()
def build_inputs_hpta(model, tokenizer, hpta_pooler, hpta, feature_path, prompt, device, prefix_tokens):
    vis = np.load(feature_path).astype(np.float32)   
    if vis.ndim == 1:
        vis = vis.reshape(1, -1)
    
    vis_t = torch.from_numpy(vis).unsqueeze(0).to(device) 
    mask  = torch.ones(1, vis_t.size(1), dtype=torch.bool, device=device) 

    token_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  
    tok_emb   = model.get_input_embeddings()(token_ids)  

    vis_scales, vis_masks = hpta_pooler(vis_t, mask) 
    prefix = hpta(vis_scales, tok_emb, vis_attn_masks=vis_masks)  
    
    if prefix.size(1) != prefix_tokens:
        raise ValueError(f"HPTA output {prefix.size(1)} tokens, expected {prefix_tokens}")

    inputs_embeds = torch.cat([prefix.to(tok_emb.dtype), tok_emb], dim=1)   
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    
    return inputs_embeds, attn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--feature_dir", type=str, default=DEFAULT_FEATURES)
    parser.add_argument("--report_csv", type=str, default=DEFAULT_REPORTCSV)
    
    parser.add_argument("--hpta_ckpt", type=str, default=HPTA_CKPT, help="Path to best_hpta.pt")
    parser.add_argument("--lora_dir", type=str, default=DEFAULT_LORA_DIR, help="Path to LoRA folder")
    parser.add_argument("--out_csv", type=str, default=DEFAULT_OUTCSV)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N samples")
    
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=4)
    
    parser.add_argument("--bertscore_model", type=str, default="roberta-base")
    parser.add_argument("--bertscore_baseline", action="store_true")
    parser.add_argument("--rouge_type", type=str, default="rougeL")
    parser.add_argument("--medbert_model", type=str, default="Shushant/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ContaminationQAmodel_PubmedBERT")
    
    parser.add_argument("--hpta_scales", type=str, default="2,4,6,8")
    parser.add_argument("--hpta_q_per_scale", type=int, default=4)
    parser.add_argument("--hpta_heads", type=int, default=8)
    parser.add_argument("--hpta_dropout", type=float, default=0.0) 

    args = parser.parse_args()

    # 1. Load Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # 2. Load LoRA Adapter
    print(f"Loading LoRA adapter from: {args.lora_dir}")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model.eval()

    # 3. Load HPTA Module
    hpta_pooler, hpta, hpta_prefix_tokens = load_hpta_module(args, base_model.config, DEVICE)
    print(f"[Info] HPTA Loaded. Prefix tokens: {hpta_prefix_tokens}")

    # 4. Setup Precision
    amp_dtype = None
    if args.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        amp_dtype = torch.float16

    # 5. Prepare Data
    ref_map = read_refs(args.report_csv)
    video_ids = sorted(ref_map.keys())
    
    valid_ids = []
    for vid in video_ids:
        if (Path(args.feature_dir) / f"{vid}.npy").exists():
            valid_ids.append(vid)
            
    if args.limit and args.limit > 0:
        valid_ids = valid_ids[:args.limit]
        
    print(f"Evaluating {len(valid_ids)} samples ...")

    # 6. Setup Metrics initializers (Lightweight metrics only)
    bleu_metric = BLEU(effective_order=True, max_ngram_order=1) 
    rouge = rouge_scorer.RougeScorer([args.rouge_type], use_stemmer=True)

    ban_phrases = [
        "this suggests", "competent approach", "otherwise good technique",
        "overall good", "very competent", "this strongly suggests", 
        "this indicates", "this demonstrates", "this shows",
        "a competent approach", "overall, good", "overall:", "generally good"
    ]
    bad_words = []
    for ph in ban_phrases:
        toks = tokenizer(ph, add_special_tokens=False).input_ids
        if len(toks) > 0:
            bad_words.append(toks)

    # 7. Generation Loop (Fast per-sample metrics only)
    records = []
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bad_words_ids=bad_words,
    )
    
    if args.do_sample:
        gen_kwargs.update(dict(do_sample=True, top_p=args.top_p, temperature=args.temperature))
    else:
        gen_kwargs.update(dict(do_sample=False, num_beams=args.num_beams, repetition_penalty=1.2))

    for vid in valid_ids:
        ref = ref_map[vid]
        feature_path = str(Path(args.feature_dir) / f"{vid}.npy")

        ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else nullcontext()

        with ctx:
            inputs_embeds, attn_mask = build_inputs_hpta(
                model, tokenizer, hpta_pooler, hpta, feature_path, PROMPT_TEMPLATE, DEVICE, hpta_prefix_tokens
            )
            output = model.generate(inputs_embeds=inputs_embeds, attention_mask=attn_mask, **gen_kwargs)

        pred_full = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = clean_output(pred_full)

        bleu = bleu_metric.sentence_score(pred, [ref]).score / 100.0
        rouge_val = rouge.score(ref, pred)[args.rouge_type].fmeasure 
        meteor = single_meteor_score(word_tokenize(ref), word_tokenize(pred)) 
        records.append([vid, ref, pred, bleu, rouge_val, meteor])
        
        print(f"[{vid}] BLEU={bleu:.4f} {args.rouge_type}={rouge_val:.4f} MT={meteor:.4f}")

    # 8. Batched Metrics Calculation (CIDEr & BERTScore)

    print("\n>>> Computing Heavy Metrics (CIDEr, Standard BERTScore, Med-BERTScore) in Batches...")
    
    refs_list = [r[1] for r in records]
    preds_list = [r[2] for r in records]
    
    # --- CIDEr ---
    gts = {i: [ref] for i, ref in enumerate(refs_list)}
    res = {i: [pred] for i, pred in enumerate(preds_list)}
    cider_scorer = Cider()
    _, cider_scores = cider_scorer.compute_score(gts, res)
    print("    [Done] CIDEr calculated.")

    # --- Standard BERTScore (roberta-base) ---
    print(f"    [Running] Standard BERTScore ({args.bertscore_model}) with Batch Size 32...")
    _, _, std_f1_tensor = bert_score(
        preds_list, refs_list,
        lang="en",
        model_type=args.bertscore_model,
        rescale_with_baseline=args.bertscore_baseline,
        verbose=True, 
        device=DEVICE,
        batch_size=32 
    )
    std_f1_scores = std_f1_tensor.cpu().numpy()

    # --- Med-BERTScore ---
    print(f"    [Running] Med-BERTScore ({args.medbert_model}) with Batch Size 32...")
    _, _, med_f1_tensor = bert_score(
        preds_list, refs_list,
        lang="en",
        model_type=args.medbert_model,
        rescale_with_baseline=False,
        verbose=True, 
        device=DEVICE,
        batch_size=32,
        num_layers=12
    )
    med_f1_scores = med_f1_tensor.cpu().numpy()

    # 9. DataFrame Merge and Save Results
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    
    base_cols = ["video_id", "reference", "prediction", "BLEU", args.rouge_type.upper(), "METEOR"]
    df = pd.DataFrame(records, columns=base_cols)
    
    df['BERTScore_F1'] = std_f1_scores
    df['MedBERTScore_F1'] = med_f1_scores
    df['CIDEr'] = cider_scores
    
    final_cols = ["video_id", "reference", "prediction", "BLEU", args.rouge_type.upper(), "METEOR", "BERTScore_F1", "MedBERTScore_F1", "CIDEr"]
    df = df[final_cols]
    
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*50)
    print(f"FINAL METRICS SUMMARY (N={len(df)})")
    print("="*50)
    print(f"BLEU:               {df['BLEU'].mean():.4f}")
    print(f"{args.rouge_type.upper():18s}: {df[args.rouge_type.upper()].mean():.4f}")
    print(f"METEOR:             {df['METEOR'].mean():.4f}")
    print(f"BERTScore F1:       {df['BERTScore_F1'].mean():.4f}")
    print(f"Med-BERTScore F1:   {df['MedBERTScore_F1'].mean():.4f}")
    print(f"CIDEr:              {df['CIDEr'].mean():.4f}")
    print("="*50)
    print(f"Saved to {args.out_csv}")

if __name__ == "__main__":
    main()