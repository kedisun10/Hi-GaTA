"""
train_lora.py
Stage 2: LoRA Fine-tuning for LLM (Freezing HPTA)
Based on clean train_adapter.py logic.
"""
import os
import argparse
import csv
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging as hf_logging
)
hf_logging.set_verbosity_error()

from train_adapter import (
    FeatureReportDataset,
    BatchCollator,
    read_reports,
    build_inputs_hpta_from_seq,
    compute_loss_with_label_smoothing,
)
from HPTA import HierarchicalAggregator, TemporalPyramidPooling

# ---------------- util ----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def count_trainable_params(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return tr, tot

# ---------------- defaults ----------------
# Qwen/Qwen2.5-3B-Instruct
# Qwen/Qwen2.5-1.5B-Instruct
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# google/gemma-2-2b-it

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FEATURE_DIR = "../Dataset/train_feature/sur40k"
REPORT_CSV = "../Dataset/text_data.csv"
HPTA_CKPT     = "models/stage1/sur40_Qwen1.5b/best_hpta.pt"
OUT_DIR       = "models/stage2/sur40_Qwen1.5b_lora"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

PROMPT_TEMPLATE = (
    "You are an objective surgical skills assessor. Write a short technical note based only on what is clearly visible in the video. \n"
    "Use purely descriptive language. Do not use any evaluative adjectives (e.g. excellent, good, competent, poor, adequate, confident).\n"
    "Mandatory items to report when visible (state 'not visible' if unclear):\n"
    "- Incision orientation over the stone or enterotomy closure direction: longitudinal or transverse\n"
    "- Suture technique used for enterotomy closure (e.g. simple interrupted, continuous)\n"
    "- Knot-tying method (hand-tied or instrument-tied)\n"
    "- Whether stay sutures were placed at the apices\n"
    "- Whether an assistant was used to maintain tension on stay sutures\n"
    "- Approximate number of throws per knot and whether sufficient\n"
    "- Tension when knots were laid (correct, too tight, too loose)\n"
    "- Any obvious technical errors (e.g. dropped suture, multiple attempts to extend enterotomy, dangerous scalpel handling, frayed bowel edges, catching adjacent suture)\n"
    "Write 2–4 short sentences only. Begin directly with the observed actions; do not start with 'The operator...', 'Overall...', or 'This demonstrates...'\n"
    "Example of desired style:\n"
    "'A transverse incision was made over the gallstone. Simple interrupted sutures were placed using hand ties with correct tension and sufficient throws. An assistant was used to maintain tension on stay sutures."
)

def list_valid_ids(feature_dir: str, report_map):
    vids = []
    p = Path(feature_dir)
    for vid in report_map:
        if (p / f"{vid}.npy").exists():
            vids.append(vid)
    vids.sort()
    return vids

def make_kfold_indices(n_samples: int, k: int, seed: int = 42, shuffle: bool = True):
    assert k >= 2 and k <= n_samples, "k must be in [2, n_samples]"
    idx = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1
    splits = []
    current = 0
    for fold_size in fold_sizes:
        val_idx = idx[current: current + fold_size]
        train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)
        splits.append((train_idx, val_idx))
        current += fold_size
    return splits

def build_datasets(tokenizer, args):
    report_map = read_reports(args.report_csv)
    full_ds = FeatureReportDataset(
        feature_dir=args.feature_dir,
        report_map=report_map,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        max_tokens=args.max_tokens,
        augment=True,
        aug_noise_std=args.aug_noise_std,
    )
    # Disable augmentation for validation
    val_ds = FeatureReportDataset(
        feature_dir=args.feature_dir,
        report_map=read_reports(args.report_csv),
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        max_tokens=args.max_tokens,
        augment=False, aug_noise_std=0.0
    )
    return report_map, full_ds, val_ds

def load_hpta_module(args, model):
    """
    Builds HPTA and loads weights from Stage 1 checkpoint.
    This module will be FROZEN during Stage 2.
    """
    # 1. Determine feature dimension D from a sample file
    sample_id = None
    for name in os.listdir(args.feature_dir):
        if name.endswith(".npy"):
            sample_id = os.path.splitext(name)[0]
            break
    if sample_id is None:
        raise RuntimeError(f"No .npy found under {args.feature_dir}")
    
    sample_path = Path(args.feature_dir) / f"{sample_id}.npy"
    sample = np.load(str(sample_path))
    
    # Validation for sequence features
    if sample.ndim == 2:
        D = int(sample.shape[1])
    else:
        # Fallback (though Stage 1 enforces sequence)
        D = int(sample.reshape(-1).shape[0])
        print(f"[WARN] Feature shape {sample.shape} is not strictly [S, D]. Using dim={D}")

    # 2. Build TPP
    win_sizes = tuple(int(x) for x in args.hpta_scales.split(","))
    n_scales = len(win_sizes)
    hpta_prefix_tokens = n_scales * args.hpta_q_per_scale
    hpta_pooler = TemporalPyramidPooling(window_sizes=win_sizes, stride_factor=0.5)

    # 3. Build HPTA
    hpta = HierarchicalAggregator(
        vis_dim=D, hidden=model.config.hidden_size,
        n_levels=n_scales,
        queries_per_level=args.hpta_q_per_scale,
        n_heads=args.hpta_heads, dropout=args.hpta_dropout
    ).to(args.device)

    # 4. Load Weights from Stage 1
    if args.hpta_ckpt and os.path.exists(args.hpta_ckpt):
        print(f"[HPTA] Loading Stage 1 weights from {args.hpta_ckpt} ...")
        ckpt = torch.load(args.hpta_ckpt, map_location="cpu")
        
        # train_adapter.py saves as {"hpta": state_dict}
        state_dict = ckpt.get("hpta", ckpt) 
        
        missing, unexpected = hpta.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print(f"[WARN] Missing keys in HPTA: {missing}")
    else:
        raise ValueError(f"Stage 1 checkpoint not found at {args.hpta_ckpt}. Cannot perform Stage 2 training.")

    # 5. Freeze HPTA
    for p in hpta.parameters():
        p.requires_grad = False
    hpta.eval()
    
    return hpta_pooler, hpta, hpta_prefix_tokens

# ---------------- one fold ----------------
def run_one_fold(args, train_indices=None, val_indices=None, fold_tag="fold1"):
    print(f"Loading model {args.model} for {fold_tag}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    # --- LoRA Setup ---
    from peft import LoraConfig, TaskType, get_peft_model
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", target_modules=target_modules
    )
    model = get_peft_model(model, lcfg)

    # Precision
    amp_dtype = None
    if args.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        amp_dtype = torch.float16

    # Dataset
    report_map, full_ds, val_ds = build_datasets(tokenizer, args)
    print(f"Matched {len(full_ds)} feature-report pairs")
    
    if train_indices is None or val_indices is None:
        n = len(full_ds)
        split_idx = int(0.8 * n)
        train_subset = torch.utils.data.Subset(full_ds, range(split_idx))
        val_subset   = torch.utils.data.Subset(val_ds, range(split_idx, n))
    else:
        train_subset = torch.utils.data.Subset(full_ds, train_indices.tolist())
        val_subset   = torch.utils.data.Subset(val_ds,   val_indices.tolist())

    collator = BatchCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, collate_fn=collator)
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, shuffle=False,
                              drop_last=False, collate_fn=collator)

    # Load Stage 1 HPTA (Frozen)
    hpta_pooler, hpta, hpta_prefix_tokens = load_hpta_module(args, model)

    tr, _ = count_trainable_params(model)
    print(f"LoRA Trainable Params (LLM): {tr/1e6:.2f}M")
    print(f"HPTA (Frozen) Prefix Tokens: {hpta_prefix_tokens}")

    # Optimizer (Only optimize LLM LoRA parameters)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Output setup
    best_val = float("inf")
    patience_left = args.early_stop_patience
    fold_dir = os.path.join(args.out_dir, fold_tag)
    best_dir = os.path.join(fold_dir, "lora_best")
    last_dir = os.path.join(fold_dir, "lora_last")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(last_dir, exist_ok=True)
    
    loss_csv_path = os.path.join(fold_dir, "log.csv")
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    for epoch in range(1, args.epochs + 1):
        # Model train mode (LoRA enabled), HPTA eval mode (Frozen)
        model.train()
        hpta.eval()
        
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for batch in pbar:
            # Context manager for mixed precision
            with torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else elseable():
                # Use the clean input builder from train_adapter
                inputs_embeds, attn_mask, labels, _, _ = build_inputs_hpta_from_seq(
                    hpta_pooler, hpta, model, tokenizer, batch, args.device, hpta_prefix_tokens
                )
                
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                use_cache=False, return_dict=True)
                loss = compute_loss_with_label_smoothing(outputs.logits, labels, args.label_smoothing)

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train = running / max(1, len(train_loader))

        # Validation
        model.eval()
        hpta.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate", leave=False):
                with torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else elseable():
                    inputs_embeds, attn_mask, labels, _, _ = build_inputs_hpta_from_seq(
                        hpta_pooler, hpta, model, tokenizer, batch, args.device, hpta_prefix_tokens
                    )
                    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                    use_cache=False, return_dict=True)
                    vloss = compute_loss_with_label_smoothing(outputs.logits, labels, args.label_smoothing)
                    val_total += vloss.item()

        avg_val = val_total / max(1, len(val_loader))
        print(f"[Epoch {epoch}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        with open(loss_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}"])

        model.save_pretrained(last_dir)
        if avg_val < best_val:
            best_val = avg_val
            model.save_pretrained(best_dir)
            print(f"[BEST] Updated best.pt (val={best_val:.4f})")
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] no improvement for {args.early_stop_patience} checks.")
                break

    return best_dir

# ---------------- dummy context for autocast if cpu ----------------
class elseable:
    def __enter__(self): return
    def __exit__(self, exc_type, exc_val, exc_tb): return

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--feature_dir", type=str, default=FEATURE_DIR)
    parser.add_argument("--report_csv", type=str, default=REPORT_CSV)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--device", type=str, default=DEVICE)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5) 
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=1024)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.2)
    parser.add_argument("--target_modules", type=str, default=DEFAULT_TARGET_MODULES)

    parser.add_argument("--aug_noise_std", type=float, default=0.02)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")

    parser.add_argument("--kfold", type=int, default=5)

    parser.add_argument("--hpta_ckpt", type=str, default=HPTA_CKPT, help="Path to Stage 1 best_hpta.pt")
    parser.add_argument("--hpta_scales", type=str, default="2,4,6,8")
    parser.add_argument("--hpta_q_per_scale", type=int, default=4)
    parser.add_argument("--hpta_heads", type=int, default=8)
    parser.add_argument("--hpta_dropout", type=float, default=0.1)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.kfold <= 1:
        best_dir = run_one_fold(args, fold_tag="fold0")
        print(f"[DONE] best adapter at: {best_dir}")
    else:
        report_map = read_reports(args.report_csv)
        valid_ids = list_valid_ids(args.feature_dir, report_map)
        n = len(valid_ids)
        if n == 0:
            raise RuntimeError("No valid features+reports found")
        if args.kfold > n:
            raise ValueError(f"kfold={args.kfold} > number of samples={n}")
        
        splits = make_kfold_indices(n, args.kfold, seed=args.seed, shuffle=True)
        saved = []
        for k, (train_idx, val_idx) in enumerate(splits):
            set_seed(args.seed + k)
            tag = f"fold{k}"
            best_dir = run_one_fold(
                args,
                train_indices=train_idx,
                val_indices=val_idx,
                fold_tag=tag
            )
            saved.append(best_dir)
        print(f"[DONE] K={args.kfold} folds best adapters: {saved}")

if __name__ == "__main__":
    main()