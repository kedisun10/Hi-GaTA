"""
HPTA moudle
"""
import torch
import torch.nn as nn

class TemporalPyramidPooling(nn.Module):
    """
    Multi-scale temporal pooling with sliding window averaging and mask support.

    Input:
      x:    [B, T, D] - temporal sequence features
            B: batch size, T: sequence length, D: feature dimension
      mask: [B, T] optional; True=valid, False=PAD

    Output:
      - If mask is None:
          Returns vis_list: List[[B, S_s, D]]
          (len(vis_list) = len(window_sizes), S_s = num_segments per window_size)

      - If mask is provided:
          Returns tuple (vis_list, msk_list)
            vis_list: List[[B, S_s, D]] - pooled features at multiple scales
            msk_list: List[[B, S_s]] - corresponding masks (True=valid, False=all_PAD)
    """

    def __init__(self, window_sizes=(4, 8, 16), stride_factor=0.5):
        super().__init__()
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.stride_factor = float(stride_factor)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        if mask is None:
            # w/o mask
            outs = []
            for w in self.window_sizes:
                if w > T:
                    pooled = x.mean(dim=1, keepdim=True)  # [B,1,D]
                    outs.append(pooled)
                    continue
                stride = max(1, int(w * self.stride_factor))
                segs = []
                for start in range(0, T - w + 1, stride):
                    segs.append(x[:, start:start + w, :].mean(dim=1, keepdim=True))  # [B,1,D]
                outs.append(torch.cat(segs, dim=1))  # [B, S_s, D]
            return outs

        # with mask
        mask = mask.to(dtype=torch.bool)     # [B, T]
        vis_list, msk_list = [], []
        for w in self.window_sizes:
            if w > T:

                valid = mask.float()                              # [B, T]
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1e-6)  # [B,1]
                pooled = (x * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)  # [B,1,D]
                vis_list.append(pooled)
                msk_list.append((denom.squeeze(1) > 0).unsqueeze(1))  # [B,1]
                continue

            stride = max(1, int(w * self.stride_factor))
            segs, segm = [], []
            for start in range(0, T - w + 1, stride):
                end = start + w
                m = mask[:, start:end]                      # [B,w]
                v = x[:, start:end, :]                      # [B,w,D]
                valid = m.float().sum(dim=1, keepdim=True)  # [B,1]
                num = (v * m.unsqueeze(-1).float()).sum(dim=1)   # [B,D]
                denom = valid.clamp_min(1e-6)                    # [B,1]
                avg = num / denom                                  # [B,D]
                segs.append(avg.unsqueeze(1))                      # [B,1,D]
                segm.append((valid.squeeze(1) > 0).unsqueeze(1))   # [B,1]
            if len(segs) == 0:

                segs = [x.mean(dim=1, keepdim=True)]
                segm = [mask.any(dim=1, keepdim=True)]
            vis_list.append(torch.cat(segs, dim=1))   # [B, S_s, D]
            msk_list.append(torch.cat(segm, dim=1))   # [B, S_s]

        return vis_list, msk_list


class DCABlock(nn.Module):
    """
    DCA：Self-Attn(queries) -> Cross-Attn(visual) -> Cross-Attn(text)
    """
    def __init__(self, hidden: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.vis_attn  = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.txt_attn  = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.ln_q1 = nn.LayerNorm(hidden)
        self.ln_q2 = nn.LayerNorm(hidden)
        self.ln_q3 = nn.LayerNorm(hidden)

    def forward(self, queries, vis_tokens, txt_tokens,
                vis_attn_mask=None, txt_attn_mask=None):
        # 1) queries self-attn
        q = self.ln_q1(queries)
        q2, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + q2

        # 2) cross-attn-visual
        q = self.ln_q2(queries)
        v2, _ = self.vis_attn(
            q, vis_tokens, vis_tokens,
            key_padding_mask=vis_attn_mask,  
            need_weights=False
        )
        queries = queries + v2

        # 3) cross-attn-text
        q = self.ln_q3(queries)
        t2, _ = self.txt_attn(
            q, txt_tokens, txt_tokens,
            key_padding_mask=txt_attn_mask,  
        )
        queries = queries + t2
        return queries

class HierarchicalAggregator(nn.Module):

    def __init__(self,
                 vis_dim: int,
                 hidden: int,
                 n_levels: int = 4,            #num of scales
                 queries_per_level: int = 4,   #num of queries per scale
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.n_levels = n_levels
        self.queries_per_level = queries_per_level
        self.hidden = hidden

        self.proj_v = nn.ModuleList([
            nn.Linear(vis_dim, hidden, bias=False) for _ in range(n_levels)
        ])

        self.query_tables = nn.ParameterList([
            nn.Parameter(torch.randn(queries_per_level, hidden) / (hidden ** 0.5))
            for _ in range(n_levels)
        ])

        max_blocks = n_levels * 2  
        self.blocks = nn.ModuleList([
            DCABlock(hidden, n_heads, dropout) for _ in range(max_blocks)
        ])

        self.gate_proj = nn.Linear(hidden, hidden)
        self.gate_sigmoid = nn.Sigmoid()

        self.final_norm = nn.LayerNorm(hidden)

    def forward(self, vis_tokens_scales: list, txt_tokens: torch.Tensor,
                vis_attn_masks: list = None):
        """
        vis_tokens_scales: List[[B, S_s, D]]  
        txt_tokens:        [B, Lp, H]
        vis_attn_masks:    (可选) List[[B, S_s]]
        """
        B = vis_tokens_scales[0].size(0)
        device = vis_tokens_scales[0].device
        prefix_tokens = []
        prev_level_queries = None  

        for lvl in range(self.n_levels):
            vis = vis_tokens_scales[lvl]                          # [B, S_l, D]
            V = self.proj_v[lvl](vis)                             # [B, S_l, H]
            Q = self.query_tables[lvl].unsqueeze(0).expand(B, -1, -1)  # [B, Q_l, H]

            # ============Gate fusion ============
            if prev_level_queries is not None:

                coarse_context = prev_level_queries.mean(dim=1, keepdim=True)  # [B,1,H]
                gate = self.gate_sigmoid(self.gate_proj(coarse_context))      # [B,1,H]
                Q = Q + gate * coarse_context

            # ============ Increasing depth ============
            num_blocks = (lvl + 1) * 2
            selected_blocks = self.blocks[:num_blocks]

            start_block = lvl * 2
            end_block = (lvl + 1) * 2 + self.n_levels
            selected_blocks = self.blocks[start_block:end_block]

            fixed_num_blocks = 4
            selected_blocks = self.blocks[:fixed_num_blocks]

            pad_mask_for_mha = None
            if vis_attn_masks is not None and len(vis_attn_masks) > lvl:
                pad_mask_for_mha = ~vis_attn_masks[lvl].to(dtype=torch.bool)

            for blk in selected_blocks:
                Q = blk(Q, V, txt_tokens, vis_attn_mask=pad_mask_for_mha)

            prefix_tokens.append(Q)
            prev_level_queries = Q  

        final_prefix = torch.cat(prefix_tokens, dim=1)        # [B, total_Q, H]
        return self.final_norm(final_prefix)
