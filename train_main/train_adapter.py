"""
train for HIGATA
"""
import os
import csv
import argparse
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from HPTA import HierarchicalAggregator, TemporalPyramidPooling
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging as hf_logging
)
hf_logging.set_verbosity_error()

# -------------------- 默认配置 --------------------
# Qwen/Qwen2.5-3B-Instruct
# Qwen/Qwen2.5-1.5B-Instruct
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# google/gemma-2-2b-it
DEFAULT_MODEL = ""
FEATURE_DIR = ""
REPORT_CSV = ""
OUT_DIR = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = (
   )

# -------------------- 工具函数 --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_reports(csv_path: str) -> Dict[str, str]:
    """
    :param csv_path: 第一列表头应为'video' 或'id'，第二列表头应改由'free'开始
    :return:{视频id：文本}的映射
    """
    import csv
    res = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.reader(f, dialect)
        rows = list(reader)
        first = rows[0]
        has_header = len(first) >= 2 and (first[0].lower() in ("video", "id") or first[1].lower().startswith("free"))
        for i, r in enumerate(rows):
            if i == 0 and has_header:
                continue
            if len(r) < 2:
                continue
            vid = str(r[0]).strip()
            txt = r[1].strip()
            res[vid] = txt
    return res

# -------------------- 模块定义 --------------------
class FeatureReportDataset(Dataset):

    def __init__(self, feature_dir: str, report_map: Dict[str, str], tokenizer, prompt_template: str,
                 max_tokens: int, augment: bool = False, aug_noise_std: float = 0.0):
        self.feature_dir = Path(feature_dir)
        self.report_map = report_map
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.augment = augment
        self.aug_noise_std = float(aug_noise_std)

        self.video_ids: List[str] = []
        for vid in report_map:
            if (self.feature_dir / f"{vid}.npy").exists():
                self.video_ids.append(vid)
            else:
                # print(f"[WARN] feature not found for video id {vid} -> skipped")
                continue
        self.video_ids.sort()

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        feat_np = np.load(str(self.feature_dir / f"{vid}.npy")).astype(np.float32)

        sample = {
            "video_id": vid,
        }

        feat_t = torch.from_numpy(feat_np)  # [S, D]
        if self.augment and self.aug_noise_std > 0:
            feat_t = feat_t + torch.randn_like(feat_t) * self.aug_noise_std
        sample["feat_seq"] = feat_t

        # 文本部分
        report_text = self.report_map[vid]
        prompt_text = self.prompt_template.format(video_id=vid)
        full_text = prompt_text + report_text

        tokenized_full = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.max_tokens)
        input_ids = tokenized_full["input_ids"].squeeze(0)  # [L]

        tokenized_prompt = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.max_tokens)
        prompt_len = tokenized_prompt["input_ids"].size(1)

        sample["input_ids"] = input_ids
        sample["prompt_len"] = prompt_len
        return sample

class BatchCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # ---- 文本对齐 ----
        B = len(batch)
        input_ids_list = [b["input_ids"] for b in batch]
        lengths = [x.size(0) for x in input_ids_list]
        max_len = max(lengths)
        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        for i, x in enumerate(input_ids_list):
            input_ids[i, :x.size(0)] = x
        prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)
        ids = [b["video_id"] for b in batch]

        out = {
            "video_id": ids,
            "input_ids": input_ids,                         # [B, Lmax]
            "input_lens": torch.tensor(lengths, dtype=torch.long),
            "prompt_lens": prompt_lens
        }

        # ---- 视觉特征对齐（序列特征） ----
        # 时序：pad 到同一时间长度 -> [B, S_max, D] + mask [B, S_max]
        seqs = [b["feat_seq"] for b in batch]          # list of [S_i, D]
        S_max = max(x.size(0) for x in seqs)
        D = seqs[0].size(1)
        feat_seq = torch.zeros(B, S_max, D, dtype=seqs[0].dtype)
        feat_seq_mask = torch.zeros(B, S_max, dtype=torch.bool)
        for i, x in enumerate(seqs):
            S = x.size(0)
            feat_seq[i, :S, :] = x
            feat_seq_mask[i, :S] = True
        out["feat_seq"] = feat_seq
        out["feat_seq_mask"] = feat_seq_mask

        return out


# -------------------- 训练/验证 --------------------
def compute_loss_with_label_smoothing(logits, labels, label_smoothing: float, ignore_index: int = -100):
    """
    logits: [B, T, V]
    labels: [B, T]  （与 logits 对齐，但会在内部 shift）
    """
    vocab_size = logits.size(-1)
    # causal LM: 预测位置 t 的目标是 t+1 的 token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
    loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    return loss

def build_inputs_hpta_from_seq(hpta_pooler, hpta, llm, tokenizer,
                              batch, device, prefix_tokens: int, ignore_index: int = -100):
    """
    使用真实时序特征:
      输入：
        - batch["feat_seq"]      : [B, S_max, D]
        - batch["feat_seq_mask"] : [B, S_max] (True=有效)
      输出：
        - inputs_embeds, attn_mask, labels, prefix_emb, token_embeds
    """
    vis_seq = batch["feat_seq"].to(device).float()           # [B, S_max, D]
    vis_mask = batch["feat_seq_mask"].to(device).bool()      # [B, S_max]

    input_ids = batch["input_ids"].to(device)                # [B, Lmax]
    lens = batch["input_lens"].to(device)                    # [B]
    prompt_lens = batch["prompt_lens"].to(device)            # [B]
    B, Lmax = input_ids.size()

    # 1) 文本嵌入（仅用 prompt 段作为条件）
    with torch.no_grad():
        token_embeds = llm.get_input_embeddings()(input_ids)  # [B, Lmax, H]
        max_p = prompt_lens.max().item()
        txt_tokens = token_embeds.new_zeros(B, max_p, token_embeds.size(-1))
        for i in range(B):
            p = prompt_lens[i].item()
            txt_tokens[i, :p, :] = token_embeds[i, :p, :]

    # 2) 多尺度池化（带 mask）
    vis_scales, vis_masks = hpta_pooler(vis_seq, vis_mask)     # lists: [ [B,Ss,D] , ... ], [ [B,Ss], ... ]

    # 3) HPTA 汇聚（带 mask）
    prefix_emb = hpta(vis_scales, txt_tokens, vis_attn_masks=vis_masks)  # [B, P, H]
    assert prefix_emb.size(1) == prefix_tokens, \
        f"prefix_tokens={prefix_tokens} 必须等于 n_scales * queries_per_scale"

    # 4) 与文本拼接、构造 mask/labels
    dtype = token_embeds.dtype
    inputs_embeds = torch.cat([prefix_emb.to(dtype), token_embeds], dim=1)  # [B, P+Lmax, H]

    attn_mask = torch.zeros(B, prefix_tokens + Lmax, dtype=torch.long, device=device)
    labels = torch.full((B, prefix_tokens + Lmax), ignore_index, dtype=torch.long, device=device)
    for i in range(B):
        eff_len = prefix_tokens + lens[i].item()
        attn_mask[i, :eff_len] = 1
        P_i = prompt_lens[i].item()
        L_i = lens[i].item()
        if P_i < L_i:
            labels[i, prefix_tokens + P_i: prefix_tokens + L_i] = input_ids[i, P_i:L_i]

    return inputs_embeds, attn_mask, labels, prefix_emb, token_embeds

def train_loop(hpta_pooler, hpta, llm, tokenizer, train_loader, val_loader, args):
    """
    训练/验证主循环：
      使用时序特征 + HPTA 多尺度池化 + 聚合
    """
    llm.eval()  # 始终冻结 LLM
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 优化器设置 ----
    hpta_params = list(hpta.parameters())
    trainable_params = hpta_params
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # ---- 日志文件 ----
    loss_csv_path = os.path.join(args.out_dir, "log.csv")
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    patience_left = args.early_stop_patience
    best_path_hpta = os.path.join(args.out_dir, "best_hpta.pt")
    last_path_hpta = os.path.join(args.out_dir, "last_hpta.pt")

    for epoch in range(1, args.epochs + 1):
        # ---- 训练阶段 ----
        hpta.train()

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        running = 0.0

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 时序特征 + mask
                inputs_embeds, attn_mask, labels, prefix_emb, token_emb = build_inputs_hpta_from_seq(
                    hpta_pooler, hpta, llm, tokenizer, batch, args.device, args.prefix_tokens
                )

                outputs = llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    use_cache=False,
                    return_dict=True
                )
                loss = compute_loss_with_label_smoothing(
                    outputs.logits, labels, args.label_smoothing
                )
                # 前缀范数正则
                norm_loss = (prefix_emb ** 2).mean()
                loss = loss + args.prefix_norm_lambda * norm_loss

            loss = loss.float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                avg_prefix_norm = prefix_emb.norm(dim=-1).mean().item()
                avg_token_norm = token_emb.norm(dim=-1).mean().item()

            running += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                pnorm=f"{avg_prefix_norm:.2f}",
                tnorm=f"{avg_token_norm:.2f}"
            )

        avg_train = running / max(1, len(train_loader))

        # ---- 验证阶段 ----
        hpta.eval()

        val_total = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate", leave=False):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    inputs_embeds, attn_mask, labels, prefix_emb, token_emb = build_inputs_hpta_from_seq(
                        hpta_pooler, hpta, llm, tokenizer, batch, args.device, args.prefix_tokens
                    )

                    outputs = llm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        use_cache=False,
                        return_dict=True
                    )
                    val_loss = compute_loss_with_label_smoothing(
                        outputs.logits, labels, args.label_smoothing
                    )
                    val_total += val_loss.item()

        avg_val = val_total / max(1, len(val_loader))
        print(f"[Epoch {epoch}] Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

        # ---- 记录日志 ----
        with open(loss_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}"])

        # ---- 保存权重 ----
        torch.save({"hpta": hpta.state_dict()}, last_path_hpta)

        if avg_val < best_val:
            best_val = avg_val
            torch.save({"hpta": hpta.state_dict()}, best_path_hpta)
            print(f"[BEST] Updated best ckpt (val={best_val:.4f})")
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] no improvement for {args.early_stop_patience} val checks. Stop.")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--feature_dir", type=str, default=FEATURE_DIR)
    parser.add_argument("--report_csv", type=str, default=REPORT_CSV)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--device", type=str, default=DEVICE)

    parser.add_argument("--prefix_tokens", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--prefix_norm_lambda", type=float, default=2e-2, help='启用prefix_token范数正则时的λ')

    # 数据增强
    parser.add_argument("--aug_noise_std", type=float, default=0.01)

    # HPTA参数
    parser.add_argument("--hpta_q_per_scale", type=int, default=4)  # 每尺度的查询数
    parser.add_argument("--hpta_scales", type=str, default="2,4,6,8")  # 多尺度窗口
    parser.add_argument("--hpta_heads", type=int, default=8)
    parser.add_argument("--hpta_dropout", type=float, default=0.1)


    args = parser.parse_args()
    set_seed(args.seed)

    print(f"Loading model {args.model} ")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    model.eval() # eval模式，冻结LLM
    # Llama的架构中是不带pad token的，为了对齐batch长度，使用eos token来代替pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for p in model.parameters():
        p.requires_grad = False

    report_map = read_reports(args.report_csv)
    print(f"Loaded {len(report_map)} reports")

    # 数据集构建
    full_dataset = FeatureReportDataset(
        args.feature_dir, report_map, tokenizer, PROMPT_TEMPLATE, args.max_tokens,
        augment=True, aug_noise_std=args.aug_noise_std
    )
    print(f"Matched {len(full_dataset)} feature-report pairs")
    if len(full_dataset) == 0:
        raise RuntimeError("No valid features+reports found")

    split_idx = int(0.8 * len(full_dataset))
    train_subset = torch.utils.data.Subset(full_dataset, range(split_idx))
    # 验证集禁用增强：用相同Dataset类但augment=False
    val_dataset = FeatureReportDataset(
        args.feature_dir, read_reports(args.report_csv), tokenizer, PROMPT_TEMPLATE, args.max_tokens,
        augment=False, aug_noise_std=0.0
    )
    val_subset = torch.utils.data.Subset(val_dataset, range(split_idx, len(full_dataset)))

    print(f"Train/Val split: {len(train_subset)}/{len(val_subset)}")

    # 探测 feat_dim（时序特征的特征维度 D）
    sample_path = Path(args.feature_dir) / f"{full_dataset.video_ids[0]}.npy"
    sample_feat = np.load(str(sample_path))
    assert sample_feat.ndim == 2, f"Expected 2D features [S, D], got {sample_feat.ndim}D"
    feat_dim = int(sample_feat.shape[1])
    llm_hidden = model.config.hidden_size

    # === HPTA 模块（仅支持序列特征） ===
    # 解析窗口
    win_sizes = tuple(int(x) for x in args.hpta_scales.split(","))
    n_scales = len(win_sizes)
    # P = n_scales * queries_per_scale 必须等于 prefix_tokens
    assert n_scales * args.hpta_q_per_scale == args.prefix_tokens, \
        "prefix_tokens 必须等于 n_scales × queries_per_scale，例如 3×4=12"

    # 构建 TPP（多尺度时间池化）
    hpta_pooler = TemporalPyramidPooling(window_sizes=win_sizes, stride_factor=0.5)

    # 构建 HPTA 聚合器
    hpta = HierarchicalAggregator(
        vis_dim=feat_dim, hidden=llm_hidden,
        n_levels=n_scales,
        queries_per_level=args.hpta_q_per_scale,
        n_heads=args.hpta_heads, dropout=args.hpta_dropout
    ).to(args.device)
    print(f"[HPTA] Using SEQUENCE features: vis_dim={feat_dim}, n_scales={n_scales}, "
          f"q_per_scale={args.hpta_q_per_scale} -> prefix_tokens={args.prefix_tokens}")

    total_params = sum(p.numel() for p in hpta.parameters() if p.requires_grad)
    print(f"HPTA trainable parameters: {total_params/1e6:.2f}M")

    # DataLoader
    collator = BatchCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collator)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collator)

    train_loop(hpta_pooler, hpta, model, tokenizer, train_loader, val_loader, args)


if __name__ == "__main__":
    main()