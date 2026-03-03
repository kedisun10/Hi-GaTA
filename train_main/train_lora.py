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
    VisualPrefixProjector,
    FeatureReportDataset,
    BatchCollator,
    read_reports,
    build_inputs_hpta_from_seq,
    compute_loss_with_label_smoothing,
)

from HPTA import HierarchicalTTAAggregator as HPTAAggregator, TemporalPyramidPooling
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
DEFAULT_MODEL = ""
FEATURE_DIR   = ""
REPORT_CSV    = ""
OUT_DIR       = ""
PRO_CKPT      = ""
HPTA_CKPT     = ""
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

PROMPT_TEMPLATE = (
    )

# ---------------- helpers ----------------
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

# ---------------- data builders ----------------
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
        aug_feat_drop=args.aug_feat_drop,
    )
    # Disable augmentation for validation
    val_ds = FeatureReportDataset(
        feature_dir=args.feature_dir,
        report_map=read_reports(args.report_csv),
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        max_tokens=args.max_tokens,
        augment=False, aug_noise_std=0.0, aug_feat_drop=0.0
    )
    return report_map, full_ds, val_ds

def load_projector(args, model, report_map):

    ckpt = torch.load(args.projector_ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    w0 = sd["net.0.weight"]  # [mid_dim, feat_dim]
    w3 = sd["net.3.weight"]  # [prefix_tokens*H, mid_dim]
    mid_dim, feat_dim = w0.shape[0], w0.shape[1]
    total_out = w3.shape[0]
    llm_hidden = model.config.hidden_size
    assert total_out % llm_hidden == 0, "projector ckpt hidden dim mismatch with LLM"
    prefix_tokens = total_out // llm_hidden

    adapter = VisualPrefixProjector(
        feat_dim=feat_dim,
        prefix_tokens=prefix_tokens,
        llm_hidden=llm_hidden,
        mid_dim=mid_dim,
        dropout=0.0,
        activation="gelu",
        prefix_dropout=args.prefix_dropout
    ).to(args.device)

    adapter.load_state_dict(sd, strict=True)
    for p in adapter.parameters():
        p.requires_grad = False

    return adapter, prefix_tokens


def build_hpta_modules(args, model):
    """
    Build TPP + HPTA for the sequence path (frozen, used to construct prefix embeddings).
    vis_dim must equal the channel dimension D of the sequence features (i.e. the second dim of .npy).
    """
    # Pick a sample from feature_dir to read [S, D] and get D
    sample_id = None
    for name in os.listdir(args.feature_dir):
        if name.endswith(".npy"):
            sample_id = os.path.splitext(name)[0]
            break
    if sample_id is None:
        raise RuntimeError(f"No .npy found under {args.feature_dir}")
    sample = np.load(str(Path(args.feature_dir) / f"{sample_id}.npy"))
    if sample.ndim != 2:
        raise ValueError(f"HPTA expects sequence features [S,D], but sample shape is {sample.shape}")
    D = int(sample.shape[1])

    # TPP
    win_sizes = tuple(int(x) for x in args.hpta_scales.split(","))
    n_scales = len(win_sizes)
    hpta_prefix_tokens = n_scales * args.hpta_q_per_scale
    hpta_pooler = TemporalPyramidPooling(window_sizes=win_sizes, stride_factor=0.5)

    # HPTA aggregator (vis_dim=D)
    hpta = HPTAAggregator(
        vis_dim=D, hidden=model.config.hidden_size,
        n_levels=4,
        queries_per_level=args.hpta_q_per_scale,
        n_heads=args.hpta_heads, dropout=args.hpta_dropout
    ).to(args.device)

    # Load ckpt (from train_prefix_v1's best_hpta.pt)
    if args.hpta_ckpt and os.path.exists(args.hpta_ckpt):
        ck = torch.load(args.hpta_ckpt, map_location="cpu")
        if ck.get("hpta", None) is not None:
            try:
                hpta.load_state_dict(ck["hpta"], strict=False)  # use strict=False for safety
                print(f"[HPTA] loaded from {args.hpta_ckpt}")
            except Exception as e:
                print(f"[WARN] load HPTA state_dict failed: {e}")
    else:
        print("[WARN] --hpta_ckpt not provided or path does not exist; HPTA using random initialization (not recommended).")

    for p in hpta.parameters():
        p.requires_grad = False
    return hpta_pooler, hpta, hpta_prefix_tokens

# ---------------- one fold ----------------
def run_one_fold(args, train_indices=None, val_indices=None, fold_tag="fold1"):
    print(f"Loading model {args.model} ")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    # LoRA setup
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

    report_map, full_ds, val_ds = build_datasets(tokenizer, args)
    print(f"Matched {len(full_ds)} feature-report pairs")
    if len(full_ds) == 0:
        raise RuntimeError("No valid features+reports found")

    n = len(full_ds)
    if train_indices is None or val_indices is None:
        split_idx = int(0.8 * n)
        train_subset = torch.utils.data.Subset(full_ds, range(split_idx))
        val_subset   = torch.utils.data.Subset(val_ds, range(split_idx, n))
        print(f"Train/Val split: {len(train_subset)}/{len(val_subset)} (80/20)")
    else:
        train_subset = torch.utils.data.Subset(full_ds, train_indices.tolist())
        val_subset   = torch.utils.data.Subset(val_ds,   val_indices.tolist())
        print(f"Train/Val split: {len(train_subset)}/{len(val_subset)} (K-fold)")

    collator = BatchCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, collate_fn=collator)
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, shuffle=False,
                              drop_last=False, collate_fn=collator)

    # Freeze projector (flat path) & build HPTA (sequence path)
    adapter, adapter_prefix_tokens = load_projector(args, model, report_map)
    hpta_pooler, hpta, hpta_prefix_tokens = build_hpta_modules(args, model)

    tr, tot = count_trainable_params(model)
    print(f"LoRA trainable parameters (LLM): {tr/1e6:.2f}M")
    print(f"[Info] adapter_prefix_tokens={adapter_prefix_tokens} ; hpta_prefix_tokens={hpta_prefix_tokens}")

    # Optimizer / Scheduler
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Training main loop
    best_val = float("inf")
    patience_left = args.early_stop_patience
    best_dir = os.path.join(args.out_dir, fold_tag, "lora_best")
    last_dir = os.path.join(args.out_dir, fold_tag, "lora_last")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(last_dir, exist_ok=True)
    fold_dir = os.path.join(args.out_dir, fold_tag)
    os.makedirs(fold_dir, exist_ok=True)
    loss_csv_path = os.path.join(fold_dir, "log.csv")
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    for epoch in range(1, args.epochs + 1):
        model.train(); adapter.train()  # LoRA training; adapter is frozen but .train() is harmless
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for batch in pbar:
            if amp_dtype is None:
                inputs_embeds, attn_mask, labels, _, _ = build_inputs_hpta_from_seq(
                    hpta_pooler, hpta, model, tokenizer, batch, args.device, hpta_prefix_tokens
                )
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                use_cache=False, return_dict=True)
                loss = compute_loss_with_label_smoothing(outputs.logits, labels, args.label_smoothing)
                loss.backward()
            else:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    inputs_embeds, attn_mask, labels, _, _ = build_inputs_hpta_from_seq(
                        hpta_pooler, hpta, model, tokenizer, batch, args.device, hpta_prefix_tokens
                    )
                    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                    use_cache=False, return_dict=True)
                    loss = compute_loss_with_label_smoothing(outputs.logits, labels, args.label_smoothing)
                scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if amp_dtype is None:
                optimizer.step()
            else:
                scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train = running / max(1, len(train_loader))

        # Validation
        model.eval(); adapter.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate", leave=False):
                inputs_embeds, attn_mask, labels, _, _ = build_inputs_hpta_from_seq(
                    hpta_pooler, hpta, model, tokenizer, batch, args.device, hpta_prefix_tokens
                )
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                use_cache=False, return_dict=True)
                vloss = compute_loss_with_label_smoothing(outputs.logits, labels, args.label_smoothing)
                val_total += vloss.item()

        avg_val = val_total / max(1, len(val_loader))
        print(f"[Epoch {epoch}] Train Loss: {avg_train:.4f} \n Val Loss: {avg_val:.4f}")

        with open(loss_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}"])

        model.save_pretrained(last_dir)  # Save as LoRA directory
        if avg_val < best_val:
            best_val = avg_val
            model.save_pretrained(best_dir)
            print(f"[BEST] Updated best.pt (val={best_val:.4f})")
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] no improvement for {args.early_stop_patience} val checks. Stop.")
                break

    return best_dir

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

    parser.add_argument("--projector_ckpt", type=str, default=PRO_CKPT)
    parser.add_argument("--prefix_dropout", type=float, default=0.1)
    parser.add_argument("--aug_noise_std", type=float, default=0.02)
    parser.add_argument("--aug_feat_drop", type=float, default=0.05)

    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--kfold", type=int, default=5)

    parser.add_argument("--hpta_ckpt", type=str, default=HPTA_CKPT)
    parser.add_argument("--hpta_scales", type=str, default="2,4,6,8")
    parser.add_argument("--hpta_q_per_scale", type=int, default=4)
    parser.add_argument("--hpta_layers", type=int, default=2)
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
        valid_ids = list_valid_ids(args.feature_dir, report_map)  # both .npy and report must exist
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