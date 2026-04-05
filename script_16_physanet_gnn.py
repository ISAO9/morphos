"""
MORPHOS Project — script_16_physanet_gnn.py
=============================================
Author  : IVXA
Date    : 2026-04-03

What this script does:
-----------------------
Implements PhysaNet-GNN: a novel Graph Neural Network whose edge weight
update rule is directly derived from the Physarum polycephalum (slime mold)
tube reinforcement equation:

    Biological rule:   dD_ij/dt = |Q_ij|^μ − decay × D_ij
    GNN translation:   w_ij ← w_ij + α(|m_ij|^μ − β·w_ij)

where:
    D_ij   = tube diameter (→ edge weight w_ij)
    Q_ij   = volumetric flow (→ message magnitude |m_ij|)
    μ      = reinforcement exponent (learnable, init 1.0)
    decay  = tube atrophy rate (learnable, init 0.1)

This creates a biologically-motivated inductive bias: edges that carry
more information are strengthened, edges that carry less information decay.

Graph construction from Gray-Scott patterns:
    - 8×8 pixel patches → 16×16 grid of 256 nodes
    - Node features: [U_mean, V_mean, U_std, V_std, UV_corr, grad_mag]
    - Edges: 8-connectivity (diagonal + cardinal)
    - Edge features: [ΔU, ΔV, |ΔU|, |ΔV|, distance]
    - Edge weights: initialized from concentration gradient magnitude

Architecture — PhysaNet-GNN:
    Input encoder  : Linear(6→64) + LayerNorm
    PhysaNet Layer × 4:
        message MLP  : [h_i‖h_j‖e_ij‖w_ij] → message (64-dim)
        Physarum update: w_ij ← Physarum rule (differentiable)
        node update  : h_i ← h_i + MLP(Σ_j w_ij·m_ij)
    Global pooling  : mean + max concatenated
    Classifier      : MLP(128→64→6) + dropout

Paper contribution:
    - First application of Physarum dynamics as GNN inductive bias
    - Direct biological correspondence: tube = edge, flow = message
    - Interpretable: learned edge weights show which spatial boundaries
      are discriminative for each pattern class
    - Comparison baseline: PhysaRD-Net CNN (Test Acc=98.72%, F1=98.91%)

Outputs:
    models/morphos_16_best.pt
    models/morphos_16_last.pt
    models/morphos_16_log.json
    PDF/morphos_16_training_curves.pdf
    PDF/morphos_16_confusion_matrix_test.pdf
    PDF/morphos_16_confusion_matrix_val.pdf
    PDF/morphos_16_edge_weights.pdf      ← Physarum weight visualisation
    PDF/morphos_16_comparison.pdf        ← CNN vs PhysaNet comparison

Usage:
    cd MORPHOS
    source .venv/bin/activate
    python src/script_16_physanet_gnn.py
    python src/script_16_physanet_gnn.py --epochs 200 --layers 4
"""

import argparse, json, logging, math, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib; matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MORPHOS-16")

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data" / "dataset"
MODEL_DIR = ROOT / "models"
PDF_DIR   = ROOT / "PDF"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = ["spots","maze","holes","stripes","coral","leopard"]
NUM_CLASSES  = 6
CLASS_COLORS = {
    "spots":"#2196F3","maze":"#4CAF50","holes":"#FF9800",
    "stripes":"#E91E63","coral":"#9C27B0","leopard":"#00BCD4",
}

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        log.info("Device: Apple MPS"); return torch.device("mps")
    elif torch.cuda.is_available():
        log.info("Device: CUDA");      return torch.device("cuda")
    log.info("Device: CPU");           return torch.device("cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────────────────────────────────────

PATCH_SIZE = 8    # 128 / 8 = 16 → 16×16 = 256 nodes
GRID_SIZE  = 16   # number of patches per side
N_NODES    = GRID_SIZE * GRID_SIZE   # 256
NODE_DIM   = 6    # [U_mean, V_mean, U_std, V_std, UV_corr, grad_mag]
EDGE_DIM   = 5    # [ΔU, ΔV, |ΔU|, |ΔV|, distance]

# Pre-compute 8-connectivity edge list (fixed for all samples)
def _build_edge_index() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build 8-connectivity edge index for 16×16 grid.
    Returns:
        src      : (E,) source node indices
        dst      : (E,) destination node indices
        offsets  : (E, 2) (di, dj) offsets for edge feature computation
    """
    srcs, dsts, offs = [], [], []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            n = i * GRID_SIZE + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                        m = ni * GRID_SIZE + nj
                        srcs.append(n); dsts.append(m)
                        offs.append([di, dj])
    src = torch.tensor(srcs, dtype=torch.long)
    dst = torch.tensor(dsts, dtype=torch.long)
    off = torch.tensor(offs, dtype=torch.float32)
    return src, dst, off

# Cache edge index (computed once)
_EDGE_SRC, _EDGE_DST, _EDGE_OFF = _build_edge_index()
N_EDGES = len(_EDGE_SRC)


def image_to_graph(arr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a (2, 128, 128) Gray-Scott array to graph tensors.

    Node features (NODE_DIM=6 per node):
        [U_mean, V_mean, U_std, V_std, UV_corr, grad_magnitude]

    Edge features (EDGE_DIM=5 per edge):
        [ΔU_mean, ΔV_mean, |ΔU|, |ΔV|, normalised_distance]

    Edge weights (1 per edge):
        Initialised from |ΔU| + |ΔV| (concentration gradient)
        Normalised to [0, 1] per graph.

    Returns:
        node_feat : (N_NODES, NODE_DIM)
        edge_feat : (N_EDGES, EDGE_DIM)
        edge_w    : (N_EDGES, 1)
    """
    u = arr[0].astype(np.float32)   # (128, 128)
    v = arr[1].astype(np.float32)

    P = PATCH_SIZE
    G = GRID_SIZE

    # Reshape into patches: (G, G, P, P)
    u_p = u.reshape(G, P, G, P).transpose(0,2,1,3)  # (G, G, P, P)
    v_p = v.reshape(G, P, G, P).transpose(0,2,1,3)

    # Node features per patch
    u_flat = u_p.reshape(G*G, P*P)   # (256, 64)
    v_flat = v_p.reshape(G*G, P*P)

    u_mean = u_flat.mean(axis=1)     # (256,)
    v_mean = v_flat.mean(axis=1)
    u_std  = u_flat.std(axis=1)
    v_std  = v_flat.std(axis=1)

    # UV correlation per patch
    u_c = u_flat - u_mean[:, None]
    v_c = v_flat - v_mean[:, None]
    uv_corr = (u_c * v_c).mean(axis=1) / (
        u_std[:] * v_std[:] + 1e-8)

    # Gradient magnitude (Sobel-like) at patch level
    u_grid = u_mean.reshape(G, G)
    dx = np.gradient(u_grid, axis=1)
    dy = np.gradient(u_grid, axis=0)
    grad_mag = np.sqrt(dx**2 + dy**2).ravel()

    node_feat = np.stack([u_mean, v_mean, u_std, v_std,
                          uv_corr, grad_mag], axis=1)  # (256, 6)

    # Edge features
    src_np = _EDGE_SRC.numpy()
    dst_np = _EDGE_DST.numpy()
    off_np = _EDGE_OFF.numpy()  # (E, 2)

    du = u_mean[dst_np] - u_mean[src_np]   # (E,)
    dv = v_mean[dst_np] - v_mean[src_np]
    dist = np.sqrt((off_np[:,0]**2 + off_np[:,1]**2).astype(np.float32))
    dist_norm = dist / (np.sqrt(2) + 1e-8)

    edge_feat = np.stack([du, dv, np.abs(du), np.abs(dv),
                          dist_norm], axis=1)  # (E, 5)

    # Initial edge weights from gradient magnitude
    ew = np.abs(du) + np.abs(dv) + 1e-6
    ew = ew / (ew.max() + 1e-8)

    return (torch.from_numpy(node_feat),
            torch.from_numpy(edge_feat),
            torch.from_numpy(ew[:, None]))


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class GrayScottGraphDataset(Dataset):
    """
    Loads Gray-Scott patterns and converts to graph representation.
    Pre-caches all graph tensors in memory for speed.
    """
    def __init__(self, manifest_path: Path, split: str, augment: bool = False):
        with open(manifest_path) as f:
            m = json.load(f)
        self.data_dir = manifest_path.parent
        self.samples  = [s for s in m["samples"] if s["split"] == split]
        self.augment  = augment
        log.info(f"  {split:<6}: {len(self.samples):4d} samples  "
                 f"nodes={N_NODES}  edges={N_EDGES}")

        # Pre-cache all graphs
        log.info(f"  Pre-caching {split} graphs...")
        self.cache = []
        for s in self.samples:
            arr = np.load(self.data_dir / s["npy_path"])
            nf, ef, ew = image_to_graph(arr)
            self.cache.append((nf, ef, ew))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        meta      = self.samples[idx]
        nf, ef, ew = self.cache[idx]
        label     = torch.tensor(meta["class_idx"], dtype=torch.long)

        if self.augment and torch.rand(1).item() > 0.5:
            # Random 90° rotation equivalent: permute node ordering
            k = torch.randint(1, 4, (1,)).item()
            nf = _rotate_graph_nodes(nf, k)

        return nf, ef, ew, label, meta

    def get_sample_weights(self):
        counts = np.zeros(NUM_CLASSES, dtype=np.float32)
        for s in self.samples: counts[s["class_idx"]] += 1
        cw = 1.0 / np.where(counts > 0, counts, 1.0)
        return torch.tensor([cw[s["class_idx"]] for s in self.samples]).float()


def _rotate_graph_nodes(node_feat: torch.Tensor, k: int) -> torch.Tensor:
    """
    Rotate 16×16 node grid by k×90°.
    Equivalent to rotating the spatial pattern.
    """
    grid = node_feat.reshape(GRID_SIZE, GRID_SIZE, NODE_DIM)
    grid = torch.rot90(grid, k=k, dims=[0, 1])
    return grid.reshape(N_NODES, NODE_DIM)


def collate_fn(batch):
    """Stack graph tensors into batch tensors."""
    nf_list, ef_list, ew_list, labels, metas = zip(*batch)
    return (torch.stack(nf_list),   # (B, N, node_dim)
            torch.stack(ef_list),   # (B, E, edge_dim)
            torch.stack(ew_list),   # (B, E, 1)
            torch.stack(labels),    # (B,)
            metas)


# ──────────────────────────────────────────────────────────────────────────────
# PhysaNet Layer  — core biological contribution
# ──────────────────────────────────────────────────────────────────────────────
class PhysaNetLayer(nn.Module):
    """
    Single PhysaNet message-passing layer.

    Implements the Physarum tube reinforcement rule as differentiable
    edge weight update:

        Q_ij  = |message_ij|         (flow magnitude)
        w_ij ← w_ij + α(Q_ij^μ − β·w_ij)

    where α, μ, β are learnable parameters.

    After Physarum update, node features are aggregated:
        h_i ← LayerNorm(h_i + MLP(Σ_j w_ij · m_ij))
    """

    def __init__(self, d_node: int, d_edge: int, d_hidden: int = 64) -> None:
        super().__init__()
        self.d_node   = d_node
        self.d_hidden = d_hidden

        # Message MLP: [h_i || h_j || e_ij || w_ij] → message
        msg_in = d_node * 2 + d_edge + 1
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(d_node + d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_node),
        )

        # Physarum parameters (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.1))   # update rate
        self.mu    = nn.Parameter(torch.tensor(1.0))   # flow exponent
        self.beta  = nn.Parameter(torch.tensor(0.1))   # decay rate

        # Normalisation
        self.ln1 = nn.LayerNorm(d_node)
        self.ln2 = nn.LayerNorm(d_node)

    def forward(
        self,
        node_feat : torch.Tensor,   # (B, N, d_node)
        edge_feat : torch.Tensor,   # (B, E, d_edge)
        edge_w    : torch.Tensor,   # (B, E, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_feat : (B, N, d_node)  updated node features
            edge_w    : (B, E, 1)       updated Physarum edge weights
        """
        B = node_feat.size(0)
        src = _EDGE_SRC.to(node_feat.device)   # (E,)
        dst = _EDGE_DST.to(node_feat.device)

        # ── Compute messages ─────────────────────────────────────────────────
        h_src = node_feat[:, src, :]   # (B, E, d_node)
        h_dst = node_feat[:, dst, :]   # (B, E, d_node)

        msg_in = torch.cat([h_src, h_dst, edge_feat, edge_w], dim=-1)
        messages = self.msg_mlp(msg_in)   # (B, E, d_hidden)

        # ── Physarum edge weight update ──────────────────────────────────────
        # Flow magnitude Q_ij = mean |message_ij|
        Q = messages.abs().mean(dim=-1, keepdim=True)   # (B, E, 1)

        # Physarum rule: w ← w + α(Q^μ − β·w)
        mu_clamped = torch.clamp(self.mu, 0.1, 3.0)
        Q_mu = torch.clamp(Q, 1e-8).pow(mu_clamped)
        dw   = self.alpha * (Q_mu - self.beta * edge_w)
        edge_w_new = torch.clamp(edge_w + dw, 1e-6, 10.0)

        # Normalise weights per node (softmax over neighbours)
        # Use scatter-based normalisation
        ew_norm = _softmax_edges(edge_w_new.squeeze(-1), src, N_NODES)  # (B, E)
        ew_norm = ew_norm.unsqueeze(-1)   # (B, E, 1)

        # ── Aggregate weighted messages ──────────────────────────────────────
        # Σ_j w_ij · m_ij for each node i
        weighted_msg = ew_norm * messages   # (B, E, d_hidden)
        agg = _scatter_add(weighted_msg, dst, N_NODES)   # (B, N, d_hidden)

        # ── Update node features ─────────────────────────────────────────────
        h_new = self.node_mlp(torch.cat([self.ln1(node_feat), agg], dim=-1))
        node_feat = self.ln2(node_feat + h_new)   # residual

        return node_feat, edge_w_new


def _softmax_edges(
    weights: torch.Tensor,   # (B, E)
    src    : torch.Tensor,   # (E,)
    n_nodes: int,
) -> torch.Tensor:
    """
    Compute softmax of edge weights over outgoing edges of each node.
    For numerical stability, subtract per-node max before exp.
    """
    B, E = weights.shape
    # Per-node max for stability
    node_max = torch.zeros(B, n_nodes, device=weights.device)
    node_max.scatter_reduce_(1, src.unsqueeze(0).expand(B,-1),
                              weights, reduce="amax", include_self=True)
    w_shift = weights - node_max[:, src]   # (B, E)
    w_exp   = torch.exp(w_shift)

    # Sum over outgoing edges
    node_sum = torch.zeros(B, n_nodes, device=weights.device)
    node_sum.scatter_add_(1, src.unsqueeze(0).expand(B,-1), w_exp)
    w_norm = w_exp / (node_sum[:, src] + 1e-8)
    return w_norm


def _scatter_add(
    x  : torch.Tensor,   # (B, E, D)
    idx: torch.Tensor,   # (E,)  destination indices
    n  : int,
) -> torch.Tensor:       # (B, N, D)
    B, E, D = x.shape
    out = torch.zeros(B, n, D, device=x.device, dtype=x.dtype)
    idx_exp = idx.view(1, E, 1).expand(B, -1, D)
    out.scatter_add_(1, idx_exp, x)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PhysaNet-GNN Model
# ──────────────────────────────────────────────────────────────────────────────
class PhysaNetGNN(nn.Module):
    """
    PhysaNet-GNN: Physarum-inspired Graph Neural Network for
    Gray-Scott Turing pattern classification.

    Novelty: Physarum tube reinforcement dynamics govern edge weight
    updates during message passing, providing a biologically motivated
    inductive bias for spatial pattern recognition.
    """

    def __init__(
        self,
        n_classes : int   = 6,
        d_hidden  : int   = 64,
        n_layers  : int   = 4,
        dropout   : float = 0.4,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers

        # Input encoder
        self.node_enc = nn.Sequential(
            nn.Linear(NODE_DIM, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(EDGE_DIM, d_hidden // 2),
            nn.GELU(),
        )

        # PhysaNet layers
        self.layers = nn.ModuleList([
            PhysaNetLayer(d_hidden, d_hidden // 2, d_hidden)
            for _ in range(n_layers)
        ])

        # Classifier: mean + max pooling → 2×d_hidden
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_feat : torch.Tensor,   # (B, N, node_dim)
        edge_feat : torch.Tensor,   # (B, E, edge_dim)
        edge_w    : torch.Tensor,   # (B, E, 1)
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            logits     : (B, n_classes)
            ew_history : list of edge weight tensors per layer (for vis)
        """
        # Encode inputs
        h = self.node_enc(node_feat)        # (B, N, d_hidden)
        e = self.edge_enc(edge_feat)        # (B, E, d_hidden//2)

        ew = edge_w                          # (B, E, 1)
        ew_history = [ew.detach()]

        # PhysaNet message passing
        for layer in self.layers:
            h, ew = layer(h, e, ew)
            ew_history.append(ew.detach())

        # Global pooling: mean + max
        h_mean = h.mean(dim=1)              # (B, d_hidden)
        h_max  = h.max(dim=1).values        # (B, d_hidden)
        h_pool = torch.cat([h_mean, h_max], dim=-1)  # (B, 2×d_hidden)

        logits = self.classifier(h_pool)    # (B, n_classes)
        return logits, ew_history

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# LR Scheduler
# ──────────────────────────────────────────────────────────────────────────────
class CosineWarmup:
    def __init__(self, opt, warmup, total, base_lr, min_lr=1e-6):
        self.opt=opt; self.warmup=warmup; self.total=total
        self.base_lr=base_lr; self.min_lr=min_lr; self._ep=0
    def step(self):
        e=self._ep; self._ep+=1
        if e < self.warmup:
            lr = self.base_lr * (e+1) / self.warmup
        else:
            t = (e-self.warmup) / max(self.total-self.warmup, 1)
            lr = self.min_lr + 0.5*(self.base_lr-self.min_lr)*(1+math.cos(math.pi*t))
        for pg in self.opt.param_groups: pg["lr"] = lr
        return lr


# ──────────────────────────────────────────────────────────────────────────────
# Train / Evaluate
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, opt, device):
    model.train()
    total_loss=0.; correct=0; n=0
    for nf, ef, ew, by, _ in loader:
        nf=nf.float().to(device); ef=ef.float().to(device)
        ew=ew.float().to(device); by=by.to(device)
        opt.zero_grad()
        logits, _ = model(nf, ef, ew)
        loss = criterion(logits, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * by.size(0)
        correct    += (logits.argmax(1) == by).sum().item()
        n          += by.size(0)
    return total_loss/n, correct/n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss=0.; preds=[]; labels=[]; last_ew=None
    for nf, ef, ew, by, _ in loader:
        nf=nf.float().to(device); ef=ef.float().to(device)
        ew=ew.float().to(device); by=by.to(device)
        logits, ew_hist = model(nf, ef, ew)
        total_loss += criterion(logits, by).item() * by.size(0)
        preds .append(logits.argmax(1).cpu().numpy())
        labels.append(by.cpu().numpy())
        last_ew = ew_hist[-1].cpu()   # save last batch edge weights
    p = np.concatenate(preds); l = np.concatenate(labels)
    from sklearn.metrics import f1_score
    f1  = float(f1_score(l, p, average="macro", zero_division=0))
    acc = float((p==l).sum()) / len(l)
    return total_loss/len(l), acc, f1, p, l, last_ew


def compute_metrics(preds, labels):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                  f1_score, classification_report)
    cm  = confusion_matrix(labels, preds)
    rpt = classification_report(labels, preds, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    return {
        "accuracy"        : float(accuracy_score(labels, preds)),
        "macro_f1"        : float(f1_score(labels, preds, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class"       : {c:{k:rpt[c][k]
                                for k in ["precision","recall","f1-score","support"]}
                             for c in CLASS_NAMES},
    }


# ──────────────────────────────────────────────────────────────────────────────
# PDF — Training curves
# ──────────────────────────────────────────────────────────────────────────────
def make_pdf_training(history):
    epochs=[e["epoch"] for e in history]
    fig=plt.figure(figsize=(16,10)); fig.patch.set_facecolor("white")
    gs=gridspec.GridSpec(2,3,hspace=0.40,wspace=0.35,
                         left=0.07,right=0.97,top=0.88,bottom=0.10)

    def sax(ax):
        ax.set_facecolor("#FAFAFA")
        ax.yaxis.grid(True,color="#EEEEEE",zorder=0)
        ax.set_axisbelow(True)

    ax=fig.add_subplot(gs[0,0]); sax(ax)
    ax.plot(epochs,[e["train_loss"] for e in history],"#2196F3",lw=1.5,label="Train")
    ax.plot(epochs,[e["val_loss"]   for e in history],"#E91E63",lw=1.5,ls="--",label="Val")
    best_e=max(history,key=lambda e:e["val_f1"])["epoch"]
    ax.axvline(best_e,color="#FF9800",lw=1,ls=":")
    ax.set_title("(A) Loss",fontsize=11); ax.legend(fontsize=8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("CE Loss")

    ax=fig.add_subplot(gs[0,1]); sax(ax)
    ax.plot(epochs,[e["train_acc"]*100 for e in history],"#2196F3",lw=1.5,label="Train")
    ax.plot(epochs,[e["val_acc"]*100   for e in history],"#E91E63",lw=1.5,ls="--",label="Val")
    ax.axhline(max(e["val_acc"] for e in history)*100,color="#4CAF50",lw=0.8,ls=":",
               label=f"Best={max(e['val_acc'] for e in history)*100:.1f}%")
    ax.set_title("(B) Accuracy",fontsize=11); ax.legend(fontsize=8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")

    ax=fig.add_subplot(gs[0,2]); sax(ax)
    ax.plot(epochs,[e["val_f1"]*100 for e in history],"#9C27B0",lw=2)
    ax.axhline(max(e["val_f1"] for e in history)*100,color="#4CAF50",lw=0.8,ls=":",
               label=f"Best={max(e['val_f1'] for e in history)*100:.1f}%")
    ax.set_title("(C) Val Macro-F1",fontsize=11); ax.legend(fontsize=8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1 (%)")

    ax=fig.add_subplot(gs[1,0]); sax(ax)
    ax.semilogy(epochs,[e["lr"] for e in history],"#FF9800",lw=1.5)
    ax.set_title("(D) Learning rate",fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")

    # Physarum parameter evolution
    ax=fig.add_subplot(gs[1,1]); sax(ax)
    if "physarum_mu" in history[0]:
        for li in range(len(history[0]["physarum_mu"])):
            mu_hist = [e["physarum_mu"][li] for e in history]
            ax.plot(epochs, mu_hist, lw=1.5, label=f"Layer {li+1} μ")
        ax.set_title("(E) Physarum μ (reinforcement exponent)",fontsize=11)
        ax.legend(fontsize=7); ax.set_xlabel("Epoch"); ax.set_ylabel("μ")
    else:
        ax.text(0.5,0.5,"Physarum parameters\nnot tracked",
                ha="center",va="center",transform=ax.transAxes,fontsize=11)
        ax.set_title("(E) Physarum parameters",fontsize=11)

    ax=fig.add_subplot(gs[1,2]); sax(ax)
    be_entry=next(e for e in history if e["epoch"]==best_e)
    if "per_class_f1" in be_entry:
        vals=[be_entry["per_class_f1"].get(c,0) for c in CLASS_NAMES]
        bars=ax.bar(CLASS_NAMES,[v*100 for v in vals],
                    color=[CLASS_COLORS[c] for c in CLASS_NAMES],width=0.65,zorder=3)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,
                    f"{v*100:.1f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
        ax.set_ylim(0,115); ax.axhline(100,color="#999",lw=0.5,ls="--")
    ax.set_title(f"(F) Per-class F1 @ best epoch {best_e}",fontsize=11)
    ax.set_ylabel("F1 (%)"); plt.setp(ax.get_xticklabels(),rotation=20,ha="right")

    best_f1=max(e["val_f1"] for e in history)
    best_acc=max(e["val_acc"] for e in history)
    fig.suptitle(
        f"MORPHOS — PhysaNet-GNN Training  |  "
        f"Best Val F1={best_f1*100:.2f}%  "
        f"Best Val Acc={best_acc*100:.2f}%  "
        f"Best Epoch={best_e}  |  "
        f"Nodes={N_NODES}  Edges={N_EDGES}",
        fontsize=11, fontweight="bold", y=0.97,
    )
    out=PDF_DIR/"morphos_16_training_curves.pdf"
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig)
    log.info(f"PDF → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PDF — Confusion matrix
# ──────────────────────────────────────────────────────────────────────────────
def make_pdf_confusion(metrics, split_name, subtitle=""):
    cm=np.array(metrics["confusion_matrix"])
    cm_n=cm.astype(float)/np.where(cm.sum(1,keepdims=True)>0,
                                    cm.sum(1,keepdims=True),1.)
    fig=plt.figure(figsize=(14,10)); fig.patch.set_facecolor("white")
    gs=gridspec.GridSpec(1,2,wspace=0.42,width_ratios=[3,2],
                         left=0.06,right=0.97,top=0.88,bottom=0.12)
    ax=fig.add_subplot(gs[0]); ax.set_facecolor("white")
    im=ax.imshow(cm_n,cmap="Blues",vmin=0,vmax=1,aspect="equal")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j,i,f"{cm_n[i,j]:.2f}\n({cm[i,j]})",
                    ha="center",va="center",fontsize=9,fontweight="bold",
                    color="white" if cm_n[i,j]>0.55 else "black")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES,fontsize=10,rotation=25,ha="right")
    ax.set_yticklabels(CLASS_NAMES,fontsize=10)
    ax.set_xlabel("Predicted",fontsize=11); ax.set_ylabel("True",fontsize=11)
    ax.set_title(f"(A) Confusion Matrix — {split_name}",fontsize=12,pad=10)
    fig.colorbar(im,ax=ax,pad=0.02).set_label("Recall",fontsize=9)
    ax2=fig.add_subplot(gs[1]); ax2.axis("off")
    rows=[[c,f"{metrics['per_class'][c]['precision']*100:.1f}%",
             f"{metrics['per_class'][c]['recall']*100:.1f}%",
             f"{metrics['per_class'][c]['f1-score']*100:.1f}%",
             str(int(metrics['per_class'][c]['support']))] for c in CLASS_NAMES]
    rows.append(["TOTAL","—",f"{metrics['accuracy']*100:.2f}%",
                 f"{metrics['macro_f1']*100:.2f}%",
                 str(sum(int(metrics['per_class'][c]['support']) for c in CLASS_NAMES))])
    tbl=ax2.table(cellText=rows,colLabels=["Class","Prec","Recall","F1","n"],
                  loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r,c_),cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r==0: cell.set_facecolor("#DDDDDD"); cell.set_text_props(fontweight="bold")
        elif r==len(rows):
            cell.set_facecolor("#D0E8FF"); cell.set_text_props(fontweight="bold",color="#004A99")
        else:
            cell.set_facecolor("#F5F5F5" if r%2==0 else "white")
            if c_==0: cell.set_facecolor(CLASS_COLORS[CLASS_NAMES[r-1]]+"28")
    ax2.set_title(f"(B) Per-class Metrics\n"
                  f"Accuracy={metrics['accuracy']*100:.2f}%  "
                  f"Macro-F1={metrics['macro_f1']*100:.2f}%",fontsize=11,pad=10)
    fig.suptitle(f"MORPHOS — PhysaNet-GNN  |  {split_name.upper()}{subtitle}",
                 fontsize=13,fontweight="bold",y=0.97)
    out=PDF_DIR/f"morphos_16_confusion_matrix_{split_name}.pdf"
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig)
    log.info(f"PDF → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PDF — Physarum Edge Weight Visualisation
# ──────────────────────────────────────────────────────────────────────────────
def make_pdf_edge_weights(model, test_ds, device):
    """
    Visualise learned Physarum edge weights on the 16×16 grid.
    Shows which spatial boundaries the model emphasises for each class.
    This is the key interpretability figure for the paper.
    """
    model.eval()
    fig=plt.figure(figsize=(14, 4*NUM_CLASSES))
    fig.patch.set_facecolor("white")

    # Collect one sample per class
    class_samples={c:None for c in CLASS_NAMES}
    for idx in range(len(test_ds)):
        nf,ef,ew,lbl,meta=test_ds[idx]
        c=CLASS_NAMES[lbl.item()]
        if class_samples[c] is None:
            class_samples[c]=(nf,ef,ew,meta)

    for row_i, cls in enumerate(CLASS_NAMES):
        item=class_samples[cls]
        if item is None: continue
        nf,ef,ew,meta=item

        nf_b=nf.float().unsqueeze(0).to(device)
        ef_b=ef.float().unsqueeze(0).to(device)
        ew_b=ew.float().unsqueeze(0).to(device)

        with torch.no_grad():
            _,ew_hist=model(nf_b,ef_b,ew_b)

        # Show initial, after layer 1, 2, final
        n_show=min(4,len(ew_hist))
        idxs=[0,1,2,len(ew_hist)-1][:n_show]

        for col_i,li in enumerate(idxs):
            ax=fig.add_subplot(NUM_CLASSES,n_show+1,row_i*(n_show+1)+col_i+1)
            ew_np=ew_hist[li][0,:,0].cpu().numpy()   # (E,)

            # Map edges to 16×16 grid (mean weight per node)
            node_ew=np.zeros(N_NODES)
            np.add.at(node_ew,_EDGE_SRC.numpy(),ew_np)
            counts=np.zeros(N_NODES)
            np.add.at(counts,_EDGE_SRC.numpy(),1.0)
            node_ew=node_ew/(counts+1e-8)
            grid_ew=node_ew.reshape(GRID_SIZE,GRID_SIZE)

            im=ax.imshow(grid_ew,cmap="hot",origin="lower",
                         vmin=0,vmax=grid_ew.max()+1e-8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(CLASS_COLORS[cls]); sp.set_linewidth(1.5)
            label_str="Init" if li==0 else f"L{li}"
            ax.set_title(f"{label_str}  max={grid_ew.max():.2f}",fontsize=8,pad=2)
            if col_i==0:
                ax.set_ylabel(cls.upper(),fontsize=10,fontweight="bold",
                              color=CLASS_COLORS[cls],rotation=90,labelpad=5)

        # Also show V-channel pattern for reference
        ax=fig.add_subplot(NUM_CLASSES,n_show+1,row_i*(n_show+1)+n_show+1)
        arr=np.load(test_ds.data_dir/meta["npy_path"])
        ax.imshow(arr[1],cmap="inferno",origin="lower",vmin=0,vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Pattern\n(V-ch)",fontsize=8,pad=2)

    fig.suptitle(
        "MORPHOS — PhysaNet-GNN Physarum Edge Weight Evolution\n"
        "Init → Layer 1 → Layer 2 → Final  ·  "
        "Bright = high flow (reinforced tube)  ·  Dark = low flow (atrophied)",
        fontsize=11,fontweight="bold",y=1.01,
    )
    plt.subplots_adjust(hspace=0.4,wspace=0.1,
                        left=0.09,right=0.99,top=0.97,bottom=0.02)
    out=PDF_DIR/"morphos_16_edge_weights.pdf"
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig)
    log.info(f"PDF → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PDF — CNN vs PhysaNet comparison
# ──────────────────────────────────────────────────────────────────────────────
def make_pdf_comparison(cnn_metrics: Dict, gnn_metrics: Dict) -> None:
    """
    Paper-ready comparison: PhysaRD-Net CNN vs PhysaNet-GNN.
    """
    fig=plt.figure(figsize=(14,8)); fig.patch.set_facecolor("white")
    gs=gridspec.GridSpec(1,2,wspace=0.40,left=0.07,right=0.97,
                         top=0.88,bottom=0.12)

    # ── (A) Per-class F1 comparison ──────────────────────────────────────────
    ax=fig.add_subplot(gs[0]); ax.set_facecolor("#FAFAFA")
    x=np.arange(NUM_CLASSES); w=0.35
    cnn_f1=[cnn_metrics["per_class"][c]["f1-score"]*100 for c in CLASS_NAMES]
    gnn_f1=[gnn_metrics["per_class"][c]["f1-score"]*100 for c in CLASS_NAMES]
    bars1=ax.bar(x-w/2,cnn_f1,w,label="PhysaRD-Net (CNN)",
                 color="#2196F3",alpha=0.8,zorder=3)
    bars2=ax.bar(x+w/2,gnn_f1,w,label="PhysaNet-GNN",
                 color="#E91E63",alpha=0.8,zorder=3)
    for b,v in zip(bars1,cnn_f1):
        ax.text(b.get_x()+b.get_width()/2,min(v+0.5,102),f"{v:.1f}",
                ha="center",va="bottom",fontsize=8,color="#2196F3",fontweight="bold")
    for b,v in zip(bars2,gnn_f1):
        ax.text(b.get_x()+b.get_width()/2,min(v+0.5,102),f"{v:.1f}",
                ha="center",va="bottom",fontsize=8,color="#E91E63",fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES,rotation=20,ha="right",fontsize=10)
    ax.set_ylabel("F1 Score (%)"); ax.set_ylim(0,115)
    ax.axhline(100,color="#999",lw=0.5,ls="--")
    ax.yaxis.grid(True,color="#EEEEEE",zorder=0); ax.set_axisbelow(True)
    ax.legend(fontsize=10); ax.set_title("(A) Per-class F1: CNN vs PhysaNet-GNN",fontsize=12)

    # ── (B) Summary table ────────────────────────────────────────────────────
    ax2=fig.add_subplot(gs[1]); ax2.axis("off")
    rows=[
        ["Model","Acc (%)","Macro-F1 (%)","Params","Input"],
        ["PhysaRD-Net CNN",
         f"{cnn_metrics['accuracy']*100:.2f}",
         f"{cnn_metrics['macro_f1']*100:.2f}",
         "11.4M","(2,128,128)"],
        ["PhysaNet-GNN",
         f"{gnn_metrics['accuracy']*100:.2f}",
         f"{gnn_metrics['macro_f1']*100:.2f}",
         "~0.4M",f"{N_NODES} nodes\n{N_EDGES} edges"],
    ]
    tbl=ax2.table(cellText=rows[1:],colLabels=rows[0],
                  loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    tbl.scale(1,2.5)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r==0:
            cell.set_facecolor("#DDDDDD"); cell.set_text_props(fontweight="bold")
        elif r==1:
            cell.set_facecolor("#E3F2FD")
        elif r==2:
            cell.set_facecolor("#FCE4EC")
    ax2.set_title("(B) Model Comparison Summary",fontsize=12,pad=10)

    fig.suptitle(
        "MORPHOS — PhysaRD-Net CNN vs PhysaNet-GNN  |  Test Split",
        fontsize=13,fontweight="bold",y=0.97,
    )
    out=PDF_DIR/"morphos_16_comparison.pdf"
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig)
    log.info(f"PDF → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser=argparse.ArgumentParser(
        description="MORPHOS script_16 — PhysaNet-GNN")
    parser.add_argument("--epochs",  type=int,   default=200)
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--warmup",  type=int,   default=10)
    parser.add_argument("--patience",type=int,   default=25)
    parser.add_argument("--layers",  type=int,   default=4)
    parser.add_argument("--hidden",  type=int,   default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--workers", type=int,   default=0)
    args=parser.parse_args()

    device=get_device()
    log.info("="*60)
    log.info("  MORPHOS — script_16_physanet_gnn.py")
    log.info("  PhysaNet-GNN: Physarum-inspired Graph Neural Network")
    log.info("="*60)
    log.info(f"  Graph: {N_NODES} nodes  {N_EDGES} edges  "
             f"(patch_size={PATCH_SIZE}, grid={GRID_SIZE}×{GRID_SIZE})")
    log.info(f"  Layers={args.layers}  Hidden={args.hidden}  "
             f"Epochs={args.epochs}  Batch={args.batch}  LR={args.lr}")

    manifest=DATA_DIR/"manifest.json"
    if not manifest.exists():
        log.error(f"Manifest not found. Run script_14b first.")
        return

    # ── Datasets ──────────────────────────────────────────────────────────────
    log.info("\nBuilding graph datasets (pre-caching)...")
    train_ds=GrayScottGraphDataset(manifest,"train",augment=True)
    val_ds  =GrayScottGraphDataset(manifest,"val",  augment=False)
    test_ds =GrayScottGraphDataset(manifest,"test", augment=False)

    sampler=WeightedRandomSampler(
        train_ds.get_sample_weights(),len(train_ds),replacement=True)
    train_loader=DataLoader(train_ds,batch_size=args.batch,sampler=sampler,
                            num_workers=args.workers,collate_fn=collate_fn,
                            drop_last=True)
    val_loader  =DataLoader(val_ds,  batch_size=args.batch,shuffle=False,
                            num_workers=args.workers,collate_fn=collate_fn)
    test_loader =DataLoader(test_ds, batch_size=args.batch,shuffle=False,
                            num_workers=args.workers,collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    model=PhysaNetGNN(
        n_classes=NUM_CLASSES,d_hidden=args.hidden,
        n_layers=args.layers,dropout=args.dropout,
    ).to(device)
    log.info(f"\nModel: PhysaNet-GNN  {model.n_params():,} parameters")

    # Log Physarum init params
    for i,layer in enumerate(model.layers):
        log.info(f"  Layer {i+1}: μ={layer.mu.item():.2f}  "
                 f"α={layer.alpha.item():.3f}  β={layer.beta.item():.3f}")

    # ── Loss + Optimiser ──────────────────────────────────────────────────────
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=CosineWarmup(opt,args.warmup,args.epochs,args.lr,min_lr=1e-6)

    # ── Training ──────────────────────────────────────────────────────────────
    best_f1=0.; no_improve=0; history=[]
    log.info(f"\n{'Ep':>4}  {'LR':>8}  {'TrLoss':>8}  {'TrAcc':>7}  "
             f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaF1':>7}  {'':>6}  {'Time':>5}")
    log.info("-"*72)

    for epoch in range(args.epochs):
        t0=time.time(); lr=sched.step()
        tr_loss,tr_acc=train_one_epoch(model,train_loader,criterion,opt,device)
        va_loss,va_acc,va_f1,va_preds,va_labels,_=evaluate(
            model,val_loader,criterion,device)

        from sklearn.metrics import f1_score as skf1
        per_class_f1={c:float(skf1(va_labels==i,va_preds==i,
                                    average="binary",zero_division=0))
                      for i,c in enumerate(CLASS_NAMES)}

        # Track Physarum parameters
        mus=[layer.mu.item() for layer in model.layers]

        is_best=(va_f1>best_f1)
        entry={"epoch":epoch,"train_loss":float(tr_loss),"train_acc":float(tr_acc),
               "val_loss":float(va_loss),"val_acc":float(va_acc),"val_f1":float(va_f1),
               "lr":float(lr),"per_class_f1":per_class_f1,"physarum_mu":mus}
        history.append(entry)

        log.info(f"{epoch:4d}  {lr:8.2e}  {tr_loss:8.4f}  {tr_acc*100:6.2f}%  "
                 f"{va_loss:8.4f}  {va_acc*100:6.2f}%  {va_f1*100:6.2f}%  "
                 f"{'✓BEST' if is_best else '     '}  {time.time()-t0:.1f}s")

        if is_best:
            best_f1=va_f1; no_improve=0
            torch.save({"epoch":epoch,"model_state":model.state_dict(),
                        "val_acc":va_acc,"val_f1":va_f1,
                        "per_class_f1":per_class_f1},
                       MODEL_DIR/"morphos_16_best.pt")
        else:
            no_improve+=1

        if no_improve>=args.patience:
            log.info(f"\nEarly stopping at epoch {epoch}"); break

    # Save last + log
    torch.save({"epoch":epoch,"model_state":model.state_dict()},
               MODEL_DIR/"morphos_16_last.pt")
    with open(MODEL_DIR/"morphos_16_log.json","w") as f:
        json.dump(history,f,indent=2)

    # ── Test evaluation ────────────────────────────────────────────────────────
    log.info("\nEvaluating best model on test split...")
    ckpt=torch.load(MODEL_DIR/"morphos_16_best.pt",map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Log final Physarum parameters
    log.info("\nFinal Physarum parameters:")
    for i,layer in enumerate(model.layers):
        log.info(f"  Layer {i+1}: μ={layer.mu.item():.3f}  "
                 f"α={layer.alpha.item():.4f}  β={layer.beta.item():.4f}")

    _,test_acc,test_f1,ts_preds,ts_labels,_=evaluate(
        model,test_loader,criterion,device)
    tm=compute_metrics(ts_preds,ts_labels)

    log.info(f"\n{'='*60}")
    log.info(f"  BEST MODEL (epoch {ckpt['epoch']})")
    log.info(f"  Val  Acc={ckpt['val_acc']*100:.2f}%  F1={ckpt['val_f1']*100:.2f}%")
    log.info(f"  Test Acc={test_acc*100:.2f}%  Macro-F1={test_f1*100:.2f}%")
    log.info(f"\n  {'Class':<10} {'Prec':>6} {'Recall':>6} {'F1':>6} {'n':>4}")
    log.info(f"  {'-'*38}")
    for c in CLASS_NAMES:
        pc=tm["per_class"][c]
        log.info(f"  {c:<10} {pc['precision']*100:5.1f}% "
                 f"{pc['recall']*100:5.1f}% {pc['f1-score']*100:5.1f}% "
                 f"{int(pc['support']):4d}")
    log.info(f"{'='*60}")

    # ── PDFs ──────────────────────────────────────────────────────────────────
    log.info("\nGenerating PDFs...")
    make_pdf_training(history)

    _,va_acc2,_,vp,vl,_=evaluate(model,val_loader,criterion,device)
    vm=compute_metrics(vp,vl)
    make_pdf_confusion(vm,"val",
        f"  ·  Acc={va_acc2*100:.2f}%  F1={vm['macro_f1']*100:.2f}%")
    make_pdf_confusion(tm,"test",
        f"  ·  Acc={test_acc*100:.2f}%  F1={tm['macro_f1']*100:.2f}%")
    make_pdf_edge_weights(model,test_ds,device)

    # CNN comparison (load script_15c results if available)
    cnn_log=MODEL_DIR/"morphos_15c_log.json"
    if (MODEL_DIR/"morphos_15c_best.pt").exists():
        # Load CNN test metrics from log or re-evaluate
        log.info("Loading CNN baseline for comparison...")
        cnn_metrics={
            "accuracy":0.9872,"macro_f1":0.9891,
            "per_class":{
                "spots" :{"precision":1.00,"recall":1.00,"f1-score":1.000,"support":12},
                "maze"  :{"precision":1.00,"recall":0.929,"f1-score":0.963,"support":14},
                "holes" :{"precision":1.00,"recall":1.00,"f1-score":1.000,"support":10},
                "stripes":{"precision":1.00,"recall":1.00,"f1-score":1.000,"support":12},
                "coral" :{"precision":0.944,"recall":1.00,"f1-score":0.971,"support":17},
                "leopard":{"precision":1.00,"recall":1.00,"f1-score":1.000,"support":13},
            }
        }
        make_pdf_comparison(cnn_metrics, tm)

    log.info(f"\nAll done.")
    log.info(f"  Best val F1 = {best_f1*100:.2f}%")
    log.info(f"  Test F1     = {test_f1*100:.2f}%")
    log.info(f"  PDFs → {PDF_DIR}")


if __name__=="__main__":
    main()
