"""
Train for Sur40k Encoder
"""
import argparse
import math
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import timm
from tqdm import tqdm

import csv
import json
from datetime import datetime

torch.backends.cudnn.benchmark = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_paths(csv_path: str) -> List[str]:
    """
    Should be the local path of Bulebear file
    """
    import csv as _csv, gzip, re

    def _normalize_path(p: str) -> str:
        if p is None:
            return ""
        p = p.strip().strip('"').strip("'")
        if not p:
            return ""
        p2 = p.replace('\\', '/')
        if p2.startswith('//'):
            p2 = re.sub(r'^//its-rds\.bham\.ac\.uk/rdsprojects/', '/rds/projects/', p2, flags=re.IGNORECASE)
        return p2

    open_fn = gzip.open if csv_path.endswith('.gz') else open
    with open_fn(csv_path, 'rt', newline='', encoding='utf-8-sig') as f:
        reader = _csv.DictReader(f)
        raw_fieldnames = reader.fieldnames
        fieldnames = [c.strip() for c in raw_fieldnames] if raw_fieldnames else []
        rename = dict(zip(raw_fieldnames, fieldnames))
        rows = []
        for row in reader:
            row = {rename.get(k, k): v for k, v in row.items()}
            col = 'video_path' if 'video_path' in row else (fieldnames[0] if fieldnames else None)
            if col is None:
                continue
            p = _normalize_path(row[col])
            if p:
                rows.append(p)
    paths_exist = [p for p in rows if os.path.exists(p)]
    if len(paths_exist) == 0:
        print(f"[DEBUG] CSV PATH: {csv_path}")
        print(f"[DEBUG] NUM OF VIDEOS: {len(rows)}")
        print(f"[DEBUG] EXAMPLE：")
        for s in rows[:3]:
            print(" ", s, "\n exists:", os.path.exists(s))
        raise FileNotFoundError(
            "No valid video paths found in {}. Probably：\n"
            "1) CSV file is Windows/UNC Path；\n"
            "2) UNC->POSIX Prefix mapping mismatch；\n"
            "3) The path does not exist or you do not have sufficient permissions.".format(csv_path)
        )
    return paths_exist


def uniform_sample_indices(num_frames: int, target_len: int) -> np.ndarray:
    if num_frames <= 0:
        return np.array([], dtype=np.int64)
    if num_frames >= target_len:
        idx = np.linspace(0, num_frames - 1, target_len).astype(np.int64)
    else:
        base = np.linspace(0, num_frames - 1, num_frames).astype(np.int64)
        pad = np.random.choice(base, target_len - num_frames)
        idx = np.sort(np.concatenate([base, pad]))
    return idx


class VideoDecoder:
    """if "auto", decord -> PyAV -> torchvision.io.read_video"""
    def __init__(self, backend: str = "auto"):
        try:
            import decord  # noqa: F401
            self.have_decord = True
        except Exception:
            self.have_decord = False
        try:
            import av  # noqa: F401
            self.have_pyav = True
        except Exception:
            self.have_pyav = False
        if backend == "auto":
            if self.have_decord:
                self.backend = "decord"
            elif self.have_pyav:
                self.backend = "pyav"
            else:
                self.backend = "torchvision"
        else:
            self.backend = backend

    def __call__(self, path: str, clip_secs: Optional[float] = None, random_window: bool = True) -> List[Image.Image]:
        if self.backend == "decord":
            return self._read_decord(path, clip_secs, random_window)
        elif self.backend == "pyav":
            return self._read_pyav(path, clip_secs, random_window)
        else:
            return self._read_torchvision(path, clip_secs, random_window)

    def _read_decord(self, path: str, clip_secs: Optional[float], random_window: bool) -> List[Image.Image]:
        from decord import VideoReader, gpu, cpu
        vr = VideoReader(path, ctx=cpu(0))
        num = len(vr)
        if num == 0:
            return []
        fps = float(vr.get_avg_fps()) if vr.get_avg_fps() is not None else 25.0
        if clip_secs is None:
            indices = np.arange(num)
        else:
            window = int(round(clip_secs * fps))
            if window <= 0:
                window = min(num, 32)
            if random_window and num > window:
                start = np.random.randint(0, num - window)
            else:
                start = 0
            stop = min(start + window, num)
            indices = np.arange(start, stop)
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
        imgs = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        return imgs

    def _read_pyav(self, path: str, clip_secs: Optional[float], random_window: bool) -> List[Image.Image]:
        import av
        container = av.open(path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        dur = float(stream.duration * stream.time_base) if stream.duration else None
        start_sec = 0.0
        window = clip_secs if clip_secs is not None else dur
        if clip_secs is not None and random_window and dur and dur > clip_secs:
            start_sec = np.random.uniform(0, dur - clip_secs)
        imgs = []
        seek_ts = int(start_sec / stream.time_base) if stream.time_base else None
        if seek_ts is not None:
            container.seek(seek_ts, any_frame=False, backward=True, stream=stream)
        cur_sec = start_sec
        end_sec = start_sec + (window if window is not None else 1e9)
        for frame in container.decode(stream):
            cur_sec = float(frame.pts * stream.time_base) if frame.pts is not None else cur_sec
            if cur_sec < start_sec:
                continue
            if cur_sec > end_sec:
                break
            imgs.append(frame.to_image())
        container.close()
        return imgs

    def _read_torchvision(self, path: str, clip_secs: Optional[float], random_window: bool) -> List[Image.Image]:
        vframes, _, info = torchvision.io.read_video(path, pts_unit='sec')
        fps = float(info['video_fps']) if 'video_fps' in info and info['video_fps'] else 25.0
        frames = vframes.numpy()  # (T, H, W, C)
        num = len(frames)
        if clip_secs is not None:
            window = int(round(clip_secs * fps))
            if window <= 0:
                window = min(num, 32)
            if random_window and num > window:
                start = np.random.randint(0, num - window)
            else:
                start = 0
            stop = min(start + window, num)
            frames = frames[start:stop]
        imgs = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        return imgs
class VideoTransform:
    def __init__(self, size=224, strong_aug=True, hflip=False):
        aug_list = []
        if hflip:
            aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if strong_aug:
            aug_list.extend([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.RandomGrayscale(p=0.05),
            ])
        self.resize_crop = transforms.RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=Image.BICUBIC)
        self.aug = transforms.Compose(aug_list)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        imgs = []
        for img in frames:
            img = self.resize_crop(img)
            img = self.aug(img)
            imgs.append(self.to_tensor(img))
        clip = torch.stack(imgs, dim=0)  # (T, 3, H, W)
        return clip


class VideoTransformTwin:
    def __init__(self, size=224, strong_aug=True, hflip=False):
        self.t1 = VideoTransform(size, strong_aug, hflip)
        self.t2 = VideoTransform(size, strong_aug, hflip)

    def __call__(self, frames: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.t1(frames), self.t2(frames)


class VideoTransformEval:
    def __init__(self, size=224):
        self.proc = transforms.Compose([
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        imgs = [self.proc(im) for im in frames]
        return torch.stack(imgs, dim=0)  # (T, 3, H, W)


class VideoTransformEvalTwin:
    
    def __init__(self, size=224):
        self.eval = VideoTransformEval(size)

    def __call__(self, frames):
        clip = self.eval(frames)
        return clip, clip


class VideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        clip_len: int = 16,
        clip_secs: Optional[float] = 8.0,  
        transform=None,
        backend: str = "auto",
        random_window: bool = True,
    ):
        self.paths = read_csv_paths(csv_path)
        self.clip_len = clip_len
        self.transform = transform
        self.decoder = VideoDecoder(backend=backend)
        self.clip_secs = clip_secs
        self.random_window = random_window

    def __len__(self):
        return len(self.paths)

    def _sample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        n = len(frames)
        if n == 0:
            raise ValueError("Decoded 0 frames.")
        idx = uniform_sample_indices(n, self.clip_len)
        return [frames[i] for i in idx]

    def __getitem__(self, idx):
        path = self.paths[idx]
        frames = self.decoder(path, clip_secs=self.clip_secs, random_window=self.random_window)
        frames = self._sample_frames(frames)
        if self.transform is None:
            to_tensor = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
            clip = torch.stack([to_tensor(im) for im in frames], 0)
            return clip
        else:
            return self.transform(frames)

class VideoDatasetFromList(Dataset):
    def __init__(self, paths: List[str], clip_len: int, clip_secs: float, transform=None, backend="auto"):
        self.paths = paths
        self.clip_len = clip_len
        self.clip_secs = clip_secs
        self.transform = transform
        self.decoder = VideoDecoder(backend=backend)

    def __len__(self):
        return len(self.paths)

    def _sample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        idx = uniform_sample_indices(len(frames), self.clip_len)
        return [frames[i] for i in idx]

    def __getitem__(self, idx):
        path = self.paths[idx]
        frames = self.decoder(path, clip_secs=self.clip_secs, random_window=True)
        frames = self._sample_frames(frames)
        return self.transform(frames) if self.transform else frames

class LearnableTemporalPE(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)  

    def forward(self, x):  # x:(B, T, C)
        return x + self.pe[:, :x.size(1), :]


def _resize_pos_embed(pos_embed: torch.Tensor, new_grid_h: int, new_grid_w: int) -> torch.Tensor:
    """
    将图像 ViT 的 pos_embed（1, 1+N, C）插值到新网格 (new_grid_h, new_grid_w)。
    """
    cls_pos = pos_embed[:, :1, :]
    tok_pos = pos_embed[:, 1:, :]
    C = tok_pos.shape[-1]
    old_n = tok_pos.shape[1]
    old_g = int(math.sqrt(old_n))
    tok_pos = tok_pos.reshape(1, old_g, old_g, C).permute(0, 3, 1, 2)  # (1, C, Gh, Gw)
    tok_pos = F.interpolate(tok_pos, size=(new_grid_h, new_grid_w), mode='bicubic', align_corners=False)
    tok_pos = tok_pos.permute(0, 2, 3, 1).reshape(1, new_grid_h * new_grid_w, C)
    return torch.cat([cls_pos, tok_pos], dim=1)


class TubeletEmbedding3D(nn.Module):
    """
    3D Tubelet Embedding
    """
    def __init__(self, in_chans: int, embed_dim: int, t: int, patch: int,
                 init: str = "central", from_2d_weight: Optional[torch.Tensor] = None, bias: bool = True):
        super().__init__()
        assert t >= 1 and patch >= 1
        self.t = t
        self.patch = patch
        self.embed = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim,
            kernel_size=(t, patch, patch), stride=(t, patch, patch),
            padding=(0, 0, 0), bias=bias
        )
        self._init_weights(init, from_2d_weight)

    def _init_weights(self, init: str, from_2d_weight: Optional[torch.Tensor]):
        # from_2d_weight: (embed_dim, in_chans, patch, patch)
        if from_2d_weight is None:
            nn.init.kaiming_normal_(self.embed.weight, mode='fan_out', nonlinearity='linear')
            if self.embed.bias is not None:
                nn.init.zeros_(self.embed.bias)
            return
        with torch.no_grad():
            w2d = from_2d_weight  # (D, C, P, P)
            D, C, P, P2 = w2d.shape
            assert P == self.patch and P2 == self.patch, \
                f"2D weigeht patch={P} does not match the expect patch={self.patch}"
            w3d = torch.zeros(D, C, self.t, P, P, device=w2d.device, dtype=w2d.dtype)
            if init == "avg":
                for i in range(self.t):
                    w3d[:, :, i, :, :] = w2d / self.t  # Eq.(8)
            else:  # 'central'
                center = self.t // 2
                w3d[:, :, center, :, :] = w2d        # Eq.(9)
            self.embed.weight.copy_(w3d)
            if self.embed.bias is not None:
                self.embed.bias.zero_()

    def forward(self, clip: torch.Tensor):
        # clip: (B, T, 3, H, W) -> (B, 3, T, H, W)
        x = clip.permute(0, 2, 1, 3, 4).contiguous()
        x = self.embed(x)                    # (B, D, T', H', W')
        B, D, T_, H_, W_ = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T', H', W', D)
        x = x.view(B, T_, H_ * W_, D)        # (B, T', Np, D)
        return x, (T_, H_, W_)               # tokens + grid


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, proj_dim=512, hidden_dim=3072):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return x


class VideoEncoder(nn.Module):
    def __init__(
        self,
        backbone_name='vit_base_patch16_224',
        pretrained_path=None,
        freeze_patch_embed=True,      
        freeze_blocks=8,
        temporal='transformer',
        proj_dim=256,
        temporal_layers=2,
        temporal_heads=6,             
        temporal_ffn=1536,            
        tubelet_t: int = 2,
        tubelet_init: str = "central",
        drop_path_rate: float = 0.1
    ):
        super().__init__()

        # 1) create ViT backbone 
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            drop_path_rate = drop_path_rate
        )
        self.embed_dim = self.backbone.num_features

        # 2)  (chooseable) load pretrained weights
        if pretrained_path is not None and os.path.isfile(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            if pretrained_path is not None:
                print(f"[DEBUG] Pre-trained weights not found in：{pretrained_path}， proceed with random initialisation")

        # 3) (CHOOSEABLE) get weight from 2D patch_embed
        patch_kernel_2d = None
        patch_size = None
        if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "proj"):

            patch_kernel_2d = self.backbone.patch_embed.proj.weight.detach().clone()
            ps = self.backbone.patch_embed.proj.kernel_size[0]
            patch_size = ps
        else:
            patch_size = 16

        # 4) CREATE 3D Tubelet Embedding
        self.tubelet = TubeletEmbedding3D(
            in_chans=3, embed_dim=self.embed_dim,
            t=tubelet_t, patch=patch_size,
            init=tubelet_init, from_2d_weight=patch_kernel_2d, bias=True
        )

        # 5) FREEZE
        self.frozen_until = freeze_blocks
        if freeze_patch_embed and hasattr(self.backbone, "patch_embed"):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
        for i, blk in enumerate(self.backbone.blocks):
            requires = i >= self.frozen_until
            for p in blk.parameters():
                p.requires_grad = requires

        # 6) CLS 
        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.spatial_cls_token, std=0.02)
        self.register_parameter('spatial_pos_embed', None)  

        # 7) temporal encoder
        self.temporal = temporal
        if temporal == 'transformer':
            self.cls_token_t = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.temporal_pos = LearnableTemporalPE(d_model=self.embed_dim, max_len=256)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=temporal_heads,
                dim_feedforward=temporal_ffn,
                batch_first=True,
                activation='gelu',
                norm_first=True  
            )
            self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=temporal_layers)
            nn.init.trunc_normal_(self.cls_token_t, std=0.02)
        elif temporal == 'mean':
            self.temporal_encoder = None
        else:
            raise ValueError("temporal must be 'transformer' or 'mean'.")

        self.proj = ProjectionHead(in_dim=self.embed_dim, proj_dim=proj_dim, hidden_dim=max(512, temporal_ffn))

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        new_ckpt = {}
        for k, v in ckpt.items():
            nk = k
            for prefix in ['module.', 'backbone.', 'model.']:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            if nk.startswith('head') or nk.startswith('fc') or 'classifier' in nk:
                continue
            new_ckpt[nk] = v
        missing, unexpected = self.backbone.load_state_dict(new_ckpt, strict=False)
        print(f"[Load pretrained] missing: {len(missing)} unexpected: {len(unexpected)}")

    @torch.no_grad()
    def _init_spatial_pos(self, grid_hw: Tuple[int, int], device):

        gh, gw = grid_hw
        if hasattr(self.backbone, "pos_embed"):
            pe = self.backbone.pos_embed
            pe_resized = _resize_pos_embed(pe, gh, gw)  # (1, 1+Np, D)
        else:
            pe_resized = torch.zeros(1, 1 + gh * gw, self.embed_dim, device=device)
            nn.init.trunc_normal_(pe_resized, std=0.02)
        self.spatial_pos_embed = nn.Parameter(pe_resized.to(device))

    def forward(self, clip: torch.Tensor):   # clip: (B, T, 3, H, W)
        # --- 3D tubelet embedding ---
        tokens, (Tprime, Hgrid, Wgrid) = self.tubelet(clip)  # tokens: (B, T', Np, D); Np=H'*W'
        B, T_, Np, D = tokens.shape
        device = tokens.device

        if self.spatial_pos_embed is None or self.spatial_pos_embed.shape[1] != (1 + Np):
            self._init_spatial_pos((Hgrid, Wgrid), device)
        #  (B*T', 1+Np, D)
        cls_spatial = self.spatial_cls_token.expand(B * T_, -1, -1)           # (B*T', 1, D)
        x = tokens.reshape(B * T_, Np, D)                                     # (B*T', Np, D)
        x = torch.cat([cls_spatial, x], dim=1)                                # (B*T', 1+Np, D)
        x = x + self.spatial_pos_embed                                        # spatial pos embed 

        # --- spatial encode ---
        for blk in self.backbone.blocks:
            x = blk(x)
        if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
            x = self.backbone.norm(x)
        h = x[:, 0, :].reshape(B, T_, D)                                      # (B, T', D)

        # --- temporal encode ---
        if self.temporal == 'mean':
            vid_feat = h.mean(dim=1)                                          # (B, D)
        else:
            cls_t = self.cls_token_t.expand(B, -1, -1)                        # (B, 1, D)
            ht = self.temporal_pos(h)                                         # (B, T', D)
            seq = torch.cat([cls_t, ht], dim=1)                               # (B, 1+T', D)
            seq = self.temporal_encoder(seq)                                  # (B, 1+T', D)
            vid_feat = seq[:, 0, :]                                           # (B, D)

        z = self.proj(vid_feat)                                               # (B, proj_dim)
        return z, vid_feat

    def unfreeze_all(self):
        for blk in self.backbone.blocks:
            for p in blk.parameters():
                p.requires_grad = True


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(B, device=z1.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc.item()

def cosine_scheduler(base_lr, final_lr, epochs, iters_per_epoch, warmup_epochs=5):
    total = epochs * iters_per_epoch
    warmup = warmup_epochs * iters_per_epoch
    lrs = []
    for t in range(total):
        if t < warmup:
            lr = base_lr * (t + 1) / max(1, warmup)
        else:
            tt = (t - warmup) / max(1, (total - warmup))
            lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * tt))
        lrs.append(lr)
    return lrs


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    last_epoch = ckpt.get('epoch', 0)
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and 'scaler' in ckpt and ckpt['scaler'] is not None:
        scaler.load_state_dict(ckpt['scaler'])
    return last_epoch

def train_one_epoch(model, loader, optimizer, scaler, device, scheduler_lrs,
                    start_iter, temperature=0.07, accum_steps=1, epoch_idx=None, total_epochs=None):
    model.train()
    loss_m, acc_m = 0.0, 0.0
    it = start_iter
    optimizer.zero_grad(set_to_none=True)

    epoch_desc = f"Epoch {epoch_idx:03d}/{total_epochs}" if (epoch_idx is not None and total_epochs is not None) else "Training"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=epoch_desc, leave=False)

    for step, (x1, x2) in pbar:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        it_lr = scheduler_lrs[it] if scheduler_lrs is not None else None
        if it_lr is not None:
            for pg in optimizer.param_groups:
                pg['lr'] = it_lr

        with torch.cuda.amp.autocast(enabled=scaler is not None, dtype=getattr(torch, args.amp_dtype)):
            z1, _ = model(x1)
            z2, _ = model(x2)
            loss, acc = info_nce_loss(z1, z2, temperature=temperature)
        loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        loss_m += loss.item() * accum_steps
        acc_m += acc
        it += 1

        cur_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{(loss.item()*accum_steps):.4f}',
            'acc': f'{acc:.4f}',
        })

    n = len(loader)
    return loss_m / n, acc_m / n, it

@torch.no_grad()
def evaluate(model, loader, device, temperature=0.07, epoch_idx=None, total_epochs=None):
    model.eval()
    loss_m, acc_m = 0.0, 0.0
    eval_desc = f"Eval {epoch_idx:03d}/{total_epochs}" if (epoch_idx is not None and total_epochs is not None) else "Eval"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=eval_desc, leave=False)

    for _, (x1, x2) in pbar:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        z1, _ = model(x1)
        z2, _ = model(x2)
        loss, acc = info_nce_loss(z1, z2, temperature=temperature)
        loss_m += loss.item()
        acc_m += acc
        pbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'val_acc': f'{acc:.4f}'})

    n = len(loader)
    return loss_m / n, acc_m / n

# ===== config =====
def write_config_json(out_dir: str, args: argparse.Namespace):
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "config.json")
    data = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in vars(args).items()}
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] config saved to {cfg_path}")


def log_metrics_csv(out_dir: str, row: dict):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        filtered = {k: row.get(k, "") for k in fields}
        writer.writerow(filtered)


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    write_config_json(args.out_dir, args)

    all_paths = read_csv_paths(args.video_csv)
    print(f"Loaded {len(all_paths)} video paths")

    split_idx = int(len(all_paths) * args.split_ratio)
    train_paths = all_paths[:split_idx]
    val_paths = all_paths[split_idx:]
    print(f"Train/Val split: {len(train_paths)}/{len(val_paths)}")

    transform_train = VideoTransformTwin(size=args.size, strong_aug=True, hflip=False)
    transform_val = VideoTransformTwin(size=args.size, strong_aug=False, hflip=False)

    train_set = VideoDatasetFromList(train_paths, clip_len=args.clip_len, clip_secs=args.clip_secs,
                                     transform=transform_train, backend=args.decoder)
    sample = train_set[0]  
    if isinstance(sample, tuple):
        a, b = sample
        print('[SANITY] one sample shapes:', a.shape, b.shape)
    else:
        print('[SANITY] one sample shape:', sample.shape)

    val_set = VideoDatasetFromList(val_paths, clip_len=args.clip_len, clip_secs=args.clip_secs,
                                   transform=transform_val, backend=args.decoder)
    # DataLoader
    def collate_fn(batch):
        x1 = torch.stack([b[0] for b in batch], dim=0)
        x2 = torch.stack([b[1] for b in batch], dim=0)
        return x1, x2

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    model = VideoEncoder(
        backbone_name='vit_base_patch16_224',
        pretrained_path=args.pretrained,
        freeze_patch_embed=not args.unfreeze_patch_embed,
        freeze_blocks=args.freeze_blocks,
        temporal=args.temporal,
        proj_dim=args.proj_dim,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        temporal_ffn=args.temporal_ffn,
        tubelet_t=args.tubelet_t,
        tubelet_init=args.tubelet_init
    ).to(device)

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_trainable/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    iters_per_epoch = len(train_loader)
    lrs = cosine_scheduler(
        base_lr=args.lr,
        final_lr=args.min_lr,
        epochs=args.epochs,
        iters_per_epoch=iters_per_epoch,
        warmup_epochs=args.warmup_epochs
    )

    start_epoch = 1
    global_iter = 0
    if args.resume and os.path.isfile(args.resume):
        last_epoch = load_checkpoint(args.resume, model, optimizer, scaler)
        start_epoch = last_epoch + 1
        global_iter = (start_epoch - 1) * iters_per_epoch
        print(f"Resumed from {args.resume}, epoch={last_epoch}")

    best_val = float('inf')
    for epoch in range(start_epoch, args.epochs + 1):
        if args.unfreeze_epoch is not None and epoch == args.unfreeze_epoch:
            print(f"Unfreezing all ViT blocks at epoch {epoch}")
            model.unfreeze_all()

        train_loss, train_acc, global_iter = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            scheduler_lrs=lrs, start_iter=global_iter,
            temperature=args.temperature, accum_steps=args.accum_steps,
            epoch_idx=epoch, total_epochs=args.epochs
        )

        np.random.seed(args.seed + epoch)
        random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        torch.cuda.manual_seed_all(args.seed + epoch)

        val_loss, val_acc = evaluate(model, val_loader, device, temperature=args.temperature,
                                     epoch_idx=epoch, total_epochs=args.epochs)

        cur_lr = optimizer.param_groups[0]['lr']
        log_metrics_csv(args.out_dir, {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "lr": f"{cur_lr:.8f}",
            "time": datetime.now().isoformat(timespec="seconds")
        })
        print(f"Epoch {epoch:03d}/{args.epochs} "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} "
              f"lr {cur_lr:.6e}")

        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'args': vars(args)
        }, os.path.join(args.out_dir, 'last.pt'))

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'args': vars(args),
                'best_val': best_val
            }, os.path.join(args.out_dir, 'best.pt'))

    print(f"Training done. Best val loss: {best_val:.4f} checkpoints at {args.out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-csv', type=str, default='')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Train/Val split ratio')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='./outputs/sur40k')
    parser.add_argument('--resume', type=str, default='')  # checkpoint path 
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--accum-steps', type=int, default=2)  # Gradient accumulation
    parser.add_argument('--num-workers', type=int, default=8)

    parser.add_argument('--clip-len', type=int, default=16)
    parser.add_argument('--clip-secs', type=float, default=4.0) 
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--decoder', type=str, default='auto', choices=['auto', 'decord', 'pyav', 'torchvision'])

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--clip-grad', type=float, default=1.0, help='Global gradient norm clipping')
    parser.add_argument('--amp-dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'])
    parser.add_argument('--drop-path', type=float, default=0.1)

    parser.add_argument('--unfreeze-patch-embed', action='store_true')
    parser.add_argument('--freeze-blocks', type=int, default=6)
    parser.add_argument('--unfreeze-epoch', type=int, default=3)  

    parser.add_argument('--temporal', type=str, default='transformer', choices=['transformer', 'mean'])
    parser.add_argument('--proj-dim', type=int, default=512)
    parser.add_argument('--temporal-layers', type=int, default=2)
    parser.add_argument('--temporal-heads', type=int, default=12)
    parser.add_argument('--temporal-ffn', type=int, default=3072)

    parser.add_argument('--tubelet-t', type=int, default=2, choices=range(1, 65),
                        help='Temporal tubelet length t')
    parser.add_argument('--tubelet-init', type=str, default='central', choices=['central', 'avg'],
                        help='Tubelet init strategy: central or avg')

    parser.add_argument('--seed', type=int, default=33)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    main(args)