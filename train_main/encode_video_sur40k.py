"""
video --> .npy
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image

from decord import VideoReader, cpu
from train_sur40k import VideoEncoder, VideoTransformEval, read_csv_paths

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_args = ckpt.get('args', {})  
    model = VideoEncoder(
        backbone_name='vit_base_patch16_224',
        pretrained_path=None,
        freeze_patch_embed=True,  
        freeze_blocks=ckpt_args.get('freeze_blocks', 6),
        temporal=ckpt_args.get('temporal', 'transformer'),
        proj_dim=ckpt_args.get('proj_dim', 512),
        temporal_layers=ckpt_args.get('temporal_layers', 2),
        temporal_heads=ckpt_args.get('temporal_heads', 12),
        temporal_ffn=ckpt_args.get('temporal_ffn', 3072),
        tubelet_t=ckpt_args.get('tubelet_t', 2),
        tubelet_init=ckpt_args.get('tubelet_init', 'central'),
        drop_path_rate=ckpt_args.get('drop_path', 0.1)
    )
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model.to(device).eval()


def build_transform(size: int = 224):
    """
    video normalization: Resize -> CenterCrop -> ToTensor + Normalize
    """
    return VideoTransformEval(size=size)  


def compute_windows(total_secs: float, clip_secs: float, overlap: float) -> np.ndarray:

    if clip_secs <= 0:
        raise ValueError("--clip-secs must > 0")
    stride = clip_secs * (1.0 - overlap)
    if stride <= 0:
        stride = clip_secs
    if total_secs <= 0:
        return np.array([[0.0, clip_secs]], dtype=np.float32)

    starts = []
    cur = 0.0

    while cur + 1e-6 < total_secs:
        start = cur
        end = min(cur + clip_secs, total_secs)
        starts.append([start, end])
        if end >= total_secs:
            break
        cur += stride
    return np.array(starts, dtype=np.float32)


@torch.no_grad()
def extract_video_feature(
    video_path: str,
    model: torch.nn.Module,
    transform: VideoTransformEval,
    clip_len: int = 32,
    clip_secs: float = 8.0,
    overlap: float = 0.0,
    device: str = DEVICE,
    batch_size: int = 16,
    use_amp: bool = True,
):
    """
    embedding the whole video
    ：np.ndarray, shape=[num_windows, D]
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = float(vr.get_avg_fps()) if vr.get_avg_fps() is not None else 25.0
    total_secs = (total_frames / fps) if fps > 0 else total_frames / 25.0

    win_secs = compute_windows(total_secs, clip_secs, overlap)  # [N,2]

    indices_list = []
    for start_sec, end_sec in win_secs:
        start_idx = int(round(start_sec * fps))
        end_idx = int(round(end_sec * fps))
        end_idx = max(end_idx, start_idx + 1)  # 至少包含1帧
        idx = np.linspace(start_idx, end_idx - 1, clip_len).astype(np.int64)
        idx = np.clip(idx, 0, total_frames - 1)
        indices_list.append(idx)

    if len(indices_list) == 0:

        dummy = torch.zeros(model.embed_dim, dtype=torch.float32).cpu().numpy()
        return dummy[None, :]

    feats = []
    scaler = torch.cuda.amp.autocast(enabled=(use_amp and device.startswith('cuda')))
    with scaler:
        for b in range(0, len(indices_list), batch_size):
            batch_idx = indices_list[b:b + batch_size]
            clips = []
            for ind in batch_idx:
                frames = vr.get_batch(ind).asnumpy()  # (T, H, W, C) uint8
                imgs = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
                clip = transform(imgs)  # (T, 3, H, W) 
                clips.append(clip)
            clip_tensor = torch.stack(clips, dim=0).to(device, non_blocking=True)  # (B, T, 3, H, W)
            _, vid_feat = model(clip_tensor)
            feats.append(vid_feat.cpu())

    feats = torch.cat(feats, dim=0).numpy()  # (N, D)
    return feats.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-csv', type=str, default='', help='CSV')
    parser.add_argument('--ckpt', type=str, default='', help='weight path')
    parser.add_argument('--save-dir', type=str, default='video_feature/vitb16_sur40k_v1', help='save path')
    parser.add_argument('--clip-len', type=int, default=16, help='The number of frames uniformly sampled per window')
    parser.add_argument('--clip-secs', type=float, default=4.0, help='Sliding window length（秒）')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap ratio of sliding windows')
    parser.add_argument('--size', type=int, default=224, help='resize size')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--no-amp', action='store_true', help='close Mixed-precision inference')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = DEVICE
    model = load_model(args.ckpt, device)
    transform = build_transform(size=args.size)

    paths = read_csv_paths(args.video_csv)
    print(f"total: {len(paths)}, save to: {args.save_dir}")

    for i, p in enumerate(paths, 1):
        vid = os.path.splitext(os.path.basename(p))[0]
        save_path = os.path.join(args.save_dir, f"{vid}.npy")
        feats = extract_video_feature(
            video_path=p,
            model=model,
            transform=transform,
            clip_len=args.clip_len,
            clip_secs=args.clip_secs,
            overlap=args.overlap,
            device=device,
            batch_size=args.batch_size,
            use_amp=(not args.no_amp),
        )
        np.save(save_path, feats)
        print(f"[{i}/{len(paths)}] Saved {save_path}, shape={feats.shape}")

    print("\n🎉 Finished!")


if __name__ == "__main__":
    main()