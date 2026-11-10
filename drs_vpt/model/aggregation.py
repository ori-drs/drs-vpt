# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

import torch


@torch.no_grad()
def sample_evenly_with_inverse(
    batch: torch.Tensor,            # (N,) long
    serialized_order: torch.Tensor, # (N,) long permutation of [0..N-1]
    K: int = 5000,
):
    """
    For each batch id:
      - take points in that batch in serialized order
      - choose K evenly spaced positions (linspace + round), wrapping if nb < K
    Returns both:
      - sample_idx (B,K): indices into original arrays (coord/feat)
      - inv_gather (N,): indices into flattened sampled array (B*K,)
        so you can do: dense = sampled_flat[inv_gather]

    Also returns:
      - batch_ids (B,)
      - sample_pos_in_batch (B,K): position in that batch's serialized list (0..nb-1), modulo if repeated
      - orig_pos_in_batch (N,): each original point's position j in its batch's serialized list (0..nb-1)
        (useful for debugging / sanity)
    """
    device = batch.device
    assert batch.ndim == 1 and serialized_order.ndim == 1
    assert batch.shape[0] == serialized_order.shape[0]
    assert batch.dtype == torch.long and serialized_order.dtype == torch.long

    N = batch.numel()
    so = serialized_order
    batch_in_serial = batch[so]  # (N,)

    batch_ids = torch.unique(batch, sorted=True)
    B = batch_ids.numel()

    sample_idx = torch.empty((B, K), device=device, dtype=torch.long)
    sample_pos_in_batch = torch.empty((B, K), device=device, dtype=torch.long)

    inv_gather = torch.empty((N,), device=device, dtype=torch.long)
    orig_pos_in_batch = torch.full((N,), -1, device=device, dtype=torch.long)

    for bi, b in enumerate(batch_ids):
        # positions in global serialized list that belong to this batch
        idx_in_serial = torch.nonzero(batch_in_serial == b, as_tuple=False).squeeze(1)  # (nb,)
        nb = idx_in_serial.numel()
        if nb == 0:
            raise RuntimeError(f"Batch id {int(b)} has 0 points.")

        # original indices for this batch, in serialized order
        orig_idx_in_order = so[idx_in_serial]  # (nb,)

        # record each original point's position j in its batch list
        orig_pos_in_batch[orig_idx_in_order] = torch.arange(nb, device=device, dtype=torch.long)

        # choose K positions (exactly as your original)
        if nb >= K:
            pos = torch.linspace(0, nb - 1, steps=K, device=device).round().to(torch.long)  # (K,)
        else:
            pos = torch.linspace(0, nb - 1, steps=K, device=device).round().to(torch.long) % nb

        # gather original indices
        orig_idx = orig_idx_in_order[pos]  # (K,)

        sample_idx[bi] = orig_idx
        sample_pos_in_batch[bi] = pos

        # --- inverse mapping: for every original point at position j, choose nearest sampled slot k ---
        # This maps each j in [0..nb-1] to an index in [0..K-1] consistent with "even spacing".
        if nb == 1:
            k_for_j = torch.zeros((1,), device=device, dtype=torch.long)
        else:
            j = torch.arange(nb, device=device, dtype=torch.float32)              # (nb,)
            k_for_j = torch.round(j * (K - 1) / (nb - 1)).to(torch.long)          # (nb,)

        inv_gather[orig_idx_in_order] = bi * K + k_for_j

    return sample_idx, inv_gather, batch_ids, sample_pos_in_batch, orig_pos_in_batch


import torch.nn as nn

class ZeroInitResidualFuse(nn.Module):
    """
    fused = s + delta(d, s)
    where delta is an MLP ending in a zero-initialized Linear => starts as exact identity on s.
    """
    def __init__(self, Cd: int, Cs: int, hidden: int = 512, use_s_in_delta: bool = True):
        super().__init__()
        in_dim = Cd + (Cs if use_s_in_delta else 0)

        self.pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, Cs)

        # critical: start as no-op
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.use_s_in_delta = use_s_in_delta

    def forward(self, s: torch.Tensor, d: torch.Tensor, chunk_size: int = 16384) -> torch.Tensor:
        # s: (N, Cs), d: (N, Cd)
        if torch.is_grad_enabled() or s.shape[0] <= chunk_size:
            return self._fuse(s, d)
        # Inference: every op is per point, so fuse in chunks to bound the (N, Cd + Cs) activations,
        # the peak memory of a full-resolution forward pass.
        out = torch.empty_like(s)
        for i in range(0, s.shape[0], chunk_size):
            out[i:i + chunk_size] = self._fuse(s[i:i + chunk_size], d[i:i + chunk_size])
        return out

    def _fuse(self, s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        x = torch.cat([d, s], dim=-1) if self.use_s_in_delta else d
        delta = self.out(self.pre(x))          # (N, Cs), initially all zeros
        return s + delta


def dense_pack_simple(x, batch_idx):
    N, C = x.shape
    B = int(batch_idx.max()) + 1

    counts = torch.bincount(batch_idx, minlength=B)
    Nmax = int(counts.max())

    x_pad = x.new_zeros((B, Nmax, C))
    mask = torch.zeros((B, Nmax), dtype=torch.bool, device=x.device)

    for b in range(B):
        sel = (batch_idx == b).nonzero(as_tuple=True)[0]
        nb = sel.numel()
        x_pad[b, :nb] = x[sel]
        mask[b, :nb] = True

    return x_pad, mask



def gather_valid_indices(
    validity_mask: torch.Tensor,   # (B, N) bool
    max_n: int | None = None,
) -> torch.Tensor:
    """
    Return idx: (B, max_n) long, such that validity_mask[b, idx[b, :]] is True
    whenever there is at least one valid in that batch.
    If a batch has > max_n valids, they are subsampled evenly (in index order).
    If a batch has < max_n valids, indices are repeated (wrap).
    If a batch has 0 valids, indices are all 0 (fallback).
    """
    assert validity_mask.ndim == 2
    B, N = validity_mask.shape
    device = validity_mask.device

    counts = validity_mask.sum(dim=1)  # (B,)
    if max_n is None:
        max_n = int(counts.max().item())

    idx = torch.empty((B, max_n), dtype=torch.long, device=device)

    for b in range(B):
        valid = torch.nonzero(validity_mask[b], as_tuple=False).squeeze(1)  # (k,)
        k = valid.numel()
        if k == 0:
            idx[b].fill_(0)
        elif k > max_n:
            idx[b] = valid[torch.linspace(0, k - 1, max_n, dtype=torch.float64, device=device).round().long()]
        else:
            # repeat / wrap
            reps = (max_n + k - 1) // k
            idx[b] = valid.repeat(reps)[:max_n]

    return idx
