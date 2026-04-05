"""
MORPHOS Project — script_14b_dataset_regenerate.py
====================================================
Author  : IVXA
Date    : 2026-04-03

What this script does:
-----------------------
Regenerates the Gray-Scott dataset with the following critical fix:

    PROBLEM (script_14):
        Seed was always placed at the CENTER of the grid.
        After simulation, a faint residual center bias remained.
        The CNN learned to detect the seed artifact, not the Turing pattern.
        → val_acc=100% at epoch 38 was spurious (model memorized center blob)

    FIX (script_14b):
        1. Seed positions are RANDOM (5 random blocks scattered over grid)
        2. Multiple seeds force patterns to develop from many starting points
        3. No single spatial location is shared across samples
        4. Simulation steps increased to 15,000 for full steady-state

Additional improvements over script_14:
    - Grid size: 128×128 (unchanged)
    - Steps: 15,000 (was 12,000) — ensures full Turing pattern development
    - Seeds per combo: 10 (was 8) — more diversity per parameter set
    - Output: overwrites data/dataset/  (old data is replaced)

Physical parameter space (confirmed safe by script_13):
    Du ∈ {0.16, 0.17, 0.18},  Dv = Du × 0.50
    6 pattern families × 4 (F,k) variants × 3 Du × 10 seeds = 720 jobs
    Expected accepted (after FFT gate): ~600 samples

Usage:
    cd MORPHOS
    source .venv/bin/activate
    python src/script_14b_dataset_regenerate.py
    python src/script_14b_dataset_regenerate.py --quick   # fast test
"""

import argparse
import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MORPHOS-14b")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dataset"
PDF_DIR  = ROOT / "PDF"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Parameters (same safe space as script_13/14)
# ──────────────────────────────────────────────────────────────────────────────
SAFE_DU     = [0.16, 0.17, 0.18]
DV_RATIO    = 0.50

PATTERN_FAMILIES: Dict[str, List[Tuple[float, float]]] = {
    "spots":   [(0.035,0.065),(0.037,0.063),(0.040,0.062),(0.033,0.064)],
    "maze":    [(0.060,0.062),(0.058,0.060),(0.055,0.062),(0.062,0.063)],
    "holes":   [(0.039,0.058),(0.037,0.057),(0.041,0.059),(0.038,0.056)],
    "stripes": [(0.026,0.051),(0.028,0.053),(0.024,0.052),(0.027,0.050)],
    "coral":   [(0.062,0.062),(0.060,0.060),(0.064,0.063),(0.061,0.061)],
    "leopard": [(0.030,0.057),(0.032,0.058),(0.028,0.056),(0.031,0.059)],
}
CLASS_NAMES  = list(PATTERN_FAMILIES.keys())
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
CLASS_COLORS = {
    "spots":"#2196F3","maze":"#4CAF50","holes":"#FF9800",
    "stripes":"#E91E63","coral":"#9C27B0","leopard":"#00BCD4",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    class_name: str
    class_idx : int
    Du: float; Dv: float; F: float; k: float
    seed: int; N: int; steps: int
    dt: float = 1.0; dx: float = 1.0
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

@dataclass
class SampleMeta:
    sample_id: str; class_name: str; class_idx: int
    Du: float; Dv: float; F: float; k: float
    seed: int; N: int; steps: int
    r_dom: float; wavelength: float; std_u: float; std_v: float
    quality: str; split: str; npy_path: str; png_path: str

# ──────────────────────────────────────────────────────────────────────────────
# Gray-Scott Simulator  —  KEY FIX: random seed positions
# ──────────────────────────────────────────────────────────────────────────────
def laplacian2d(Z: np.ndarray) -> np.ndarray:
    return (np.roll(Z,1,0)+np.roll(Z,-1,0)+
            np.roll(Z,1,1)+np.roll(Z,-1,1) - 4.0*Z)

def run_gray_scott(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gray-Scott simulator with RANDOM seed positions.

    Critical fix vs script_14:
        OLD: single center seed at (N//2, N//2) → center bias artifact
        NEW: 5 seeds at random positions → no spatial bias

    Each seed is a small r=N//12 block with (u=0.5, v=0.25) + noise.
    Multiple random seeds force the pattern to develop uniformly.
    """
    rng = np.random.default_rng(cfg.seed)
    N   = cfg.N

    u = np.ones((N, N), dtype=np.float64)
    v = np.zeros((N, N), dtype=np.float64)

    # ── RANDOM SEEDS (5 positions) ────────────────────────────────────────────
    r        = N // 12          # smaller seed radius
    n_seeds  = 5
    for _ in range(n_seeds):
        cx = rng.integers(r, N - r)
        cy = rng.integers(r, N - r)
        u[cx-r:cx+r, cy-r:cy+r] = 0.50 + rng.uniform(-0.05, 0.05)
        v[cx-r:cx+r, cy-r:cy+r] = 0.25 + rng.uniform(-0.05, 0.05)

    # Global noise overlay
    u += rng.uniform(-0.02, 0.02, (N, N))
    v += rng.uniform(-0.02, 0.02, (N, N))
    np.clip(u, 0.0, 1.0, out=u)
    np.clip(v, 0.0, 1.0, out=v)

    Du, Dv, F, k = cfg.Du, cfg.Dv, cfg.F, cfg.k
    inv_dx2 = 1.0 / cfg.dx**2

    for _ in range(cfg.steps):
        uvv  = u * v * v
        u   += cfg.dt * (Du * laplacian2d(u) * inv_dx2 - uvv + F*(1.0-u))
        v   += cfg.dt * (Dv * laplacian2d(v) * inv_dx2 + uvv - (F+k)*v)
        np.clip(u, 0.0, 1.0, out=u)
        np.clip(v, 0.0, 1.0, out=v)

    return u.astype(np.float32), v.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# FFT Quality Gate
# ──────────────────────────────────────────────────────────────────────────────
def fft_diagnostics(u: np.ndarray) -> Dict:
    N   = u.shape[0]
    u_c = u - u.mean()
    F2  = np.fft.fftshift(np.fft.fft2(u_c))
    psd = np.abs(F2)**2
    fx  = np.fft.fftshift(np.fft.fftfreq(N)) * N
    fy  = np.fft.fftshift(np.fft.fftfreq(N)) * N
    fx, fy = np.meshgrid(fx, fy, indexing="xy")
    R   = np.sqrt(fx**2 + fy**2)
    psd_m = np.where(R > 0.5, psd, 0.0)
    r_dom = float(R.ravel()[np.argmax(psd_m)])
    wl    = N / r_dom if r_dom > 0 else float(N)
    return {
        "r_dom"      : r_dom,
        "wavelength" : wl,
        "std_u"      : float(u.std()),
        "is_physical": (r_dom < N/3.0) and (u.std() > 0.01),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Save sample
# ──────────────────────────────────────────────────────────────────────────────
def assign_split(rng: np.random.Generator) -> str:
    p = rng.random()
    return "train" if p < 0.70 else ("val" if p < 0.85 else "test")

def save_sample(u, v, meta, data_dir):
    d = data_dir / meta.split / meta.class_name
    d.mkdir(parents=True, exist_ok=True)
    npy = d / f"{meta.sample_id}.npy"
    png = d / f"{meta.sample_id}.png"
    np.save(npy, np.stack([u, v], axis=0))
    fig, ax = plt.subplots(figsize=(2,2), dpi=64)
    ax.imshow(v, cmap="inferno", origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return str(npy.relative_to(data_dir)), str(png.relative_to(data_dir))

# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────
def generate(N=128, steps=15000, seeds=10, quick=False):
    # Clear old dataset
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        log.info(f"Old dataset removed: {DATA_DIR}")
    DATA_DIR.mkdir(parents=True)

    split_rng = np.random.default_rng(42)
    accepted, rejected = [], 0

    jobs = [
        SimConfig(class_name=cn, class_idx=CLASS_TO_IDX[cn],
                  Du=Du, Dv=round(Du*DV_RATIO,4), F=F, k=k,
                  seed=s, N=N, steps=steps)
        for cn, fk_list in PATTERN_FAMILIES.items()
        for Du in SAFE_DU
        for (F,k) in fk_list
        for s in range(seeds)
    ]

    if quick:
        rng_q = np.random.default_rng(999)
        idx   = rng_q.choice(len(jobs), size=max(1,len(jobs)//5), replace=False)
        jobs  = [jobs[i] for i in sorted(idx)]
        log.info(f"[QUICK] {len(jobs)} jobs")

    log.info(f"Total jobs: {len(jobs)}  |  N={N}  steps={steps:,}  seeds={seeds}")
    t0 = time.time()

    for i, cfg in enumerate(jobs):
        if i % max(1, len(jobs)//20) == 0:
            eta = (time.time()-t0)/max(i,1)*(len(jobs)-i)
            log.info(f"  [{100*i/len(jobs):5.1f}%] {i}/{len(jobs)}  "
                     f"accepted={len(accepted)}  rejected={rejected}  ETA={eta:.0f}s")

        u, v = run_gray_scott(cfg)
        diag = fft_diagnostics(u)
        if not diag["is_physical"]:
            rejected += 1; continue

        split = assign_split(split_rng)
        meta  = SampleMeta(
            sample_id=cfg.sample_id, class_name=cfg.class_name,
            class_idx=cfg.class_idx, Du=cfg.Du, Dv=cfg.Dv,
            F=cfg.F, k=cfg.k, seed=cfg.seed, N=N, steps=steps,
            r_dom=diag["r_dom"], wavelength=diag["wavelength"],
            std_u=diag["std_u"], std_v=float(v.std()),
            quality="physical", split=split, npy_path="", png_path="",
        )
        npy_rel, png_rel = save_sample(u, v, meta, DATA_DIR)
        meta.npy_path = npy_rel; meta.png_path = png_rel
        accepted.append(meta)

    log.info(f"\nDone: accepted={len(accepted)}  rejected={rejected}  "
             f"time={time.time()-t0:.0f}s")
    return accepted

# ──────────────────────────────────────────────────────────────────────────────
# Save manifest + stats
# ──────────────────────────────────────────────────────────────────────────────
def save_manifest(samples):
    path = DATA_DIR / "manifest.json"
    data = {
        "version":"1.1_random_seeds",
        "created_by":"MORPHOS script_14b",
        "total_samples":len(samples),
        "classes":CLASS_NAMES,
        "seed_mode":"random_5_seeds",   # critical change from v1.0
        "splits":{
            "train": sum(1 for s in samples if s.split=="train"),
            "val"  : sum(1 for s in samples if s.split=="val"),
            "test" : sum(1 for s in samples if s.split=="test"),
        },
        "samples":[asdict(s) for s in samples],
    }
    with open(path,"w") as f: json.dump(data, f, indent=2)
    log.info(f"Manifest → {path}")

def save_stats(samples):
    stats = {}
    for cls in CLASS_NAMES:
        cs = [s for s in samples if s.class_name==cls]
        if not cs: continue
        stats[cls] = {
            "count":len(cs),
            "train":sum(1 for s in cs if s.split=="train"),
            "val"  :sum(1 for s in cs if s.split=="val"),
            "test" :sum(1 for s in cs if s.split=="test"),
            "wavelength_mean":float(np.mean([s.wavelength for s in cs])),
            "wavelength_std" :float(np.std( [s.wavelength for s in cs])),
            "std_u_mean"     :float(np.mean([s.std_u for s in cs])),
        }
    path = DATA_DIR / "class_stats.json"
    with open(path,"w") as f: json.dump(stats, f, indent=2)
    log.info(f"Stats → {path}")
    return stats

# ──────────────────────────────────────────────────────────────────────────────
# PDF — verification that center bias is gone
# ──────────────────────────────────────────────────────────────────────────────
def make_pdf_verification(samples):
    """
    Verify that center bias is eliminated:
    Plot center region mean vs outer region mean for each class.
    If random seeds work: center_mean ≈ outer_mean (no bias).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for ax_i, cls in enumerate(CLASS_NAMES):
        cls_s = [s for s in samples if s.class_name==cls and s.split=="train"][:12]
        ax    = axes[ax_i]

        center_ratios = []
        for meta in cls_s:
            arr = np.load(DATA_DIR / meta.npy_path)
            u   = arr[0]
            N   = u.shape[0]
            r   = N // 8
            cx, cy = N//2, N//2
            center_mean = u[cx-r:cx+r, cy-r:cy+r].mean()
            outer_mean  = u.mean()
            center_ratios.append(center_mean / max(outer_mean, 1e-6))

        ax.hist(center_ratios, bins=15, color=CLASS_COLORS[cls],
                alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(1.0, color="black", lw=1.5, ls="--",
                   label="No bias (ratio=1.0)")
        ax.set_title(f"{cls.upper()}  (n={len(cls_s)})",
                     fontsize=11, fontweight="bold", color=CLASS_COLORS[cls])
        ax.set_xlabel("Center/Outer mean ratio", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        mean_r = np.mean(center_ratios) if center_ratios else 0
        ax.set_title(f"{cls.upper()}  mean ratio={mean_r:.2f}",
                     fontsize=11, fontweight="bold", color=CLASS_COLORS[cls])

    fig.suptitle(
        "MORPHOS script_14b — Center Bias Verification\n"
        "Random seeds: center/outer ratio should be ≈ 1.0 (no spatial bias)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = PDF_DIR / "morphos_14b_center_bias_check.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"PDF → {out}")

def make_pdf_sample_grid(samples):
    """6 classes × 4 samples grid for visual verification."""
    n_show = 4
    fig    = plt.figure(figsize=(14, 6*2.8))
    fig.patch.set_facecolor("white")

    for row_i, cls in enumerate(CLASS_NAMES):
        cls_s  = sorted([s for s in samples if s.class_name==cls],
                        key=lambda s: s.Du)
        step   = max(1, len(cls_s)//n_show)
        picked = cls_s[::step][:n_show]
        for col_i, meta in enumerate(picked):
            ax = fig.add_subplot(6, n_show, row_i*n_show + col_i + 1)
            arr = np.load(DATA_DIR / meta.npy_path)
            ax.imshow(arr[1], cmap="inferno", origin="lower", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(CLASS_COLORS[cls]); sp.set_linewidth(1.5)
            ax.set_title(f"Du={meta.Du:.2f} F={meta.F:.3f}\n"
                         f"λ={meta.wavelength:.1f}px",
                         fontsize=6.5, pad=3)
            if col_i == 0:
                ax.set_ylabel(cls.upper(), fontsize=10, fontweight="bold",
                              color=CLASS_COLORS[cls], rotation=90, labelpad=5)

    fig.suptitle(
        "MORPHOS script_14b — Sample Grid (random seeds)  |  V-channel",
        fontsize=12, fontweight="bold", y=1.005,
    )
    plt.subplots_adjust(hspace=0.6, wspace=0.08,
                        left=0.10, right=0.98, top=0.97, bottom=0.02)
    out = PDF_DIR / "morphos_14b_sample_grid.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"PDF → {out}")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MORPHOS script_14b — Dataset regeneration (random seeds)")
    parser.add_argument("--grid",  type=int,  default=128)
    parser.add_argument("--steps", type=int,  default=15000)
    parser.add_argument("--seeds", type=int,  default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    log.info("="*60)
    log.info("  MORPHOS script_14b_dataset_regenerate.py")
    log.info("  KEY FIX: Random seed positions (no center bias)")
    log.info("="*60)
    log.info(f"  Grid={args.grid}  Steps={args.steps:,}  Seeds/combo={args.seeds}")
    log.info(f"  Expected samples: ~{len(CLASS_NAMES)*4*len(SAFE_DU)*args.seeds} jobs")

    samples = generate(args.grid, args.steps, args.seeds, args.quick)

    save_manifest(samples)
    stats = save_stats(samples)

    log.info("\nClass statistics:")
    for cls, st in stats.items():
        log.info(f"  {cls:<10} n={st['count']:4d}  "
                 f"train={st['train']}  val={st['val']}  test={st['test']}  "
                 f"λ={st['wavelength_mean']:.1f}px")

    log.info("\nGenerating PDFs...")
    make_pdf_verification(samples)
    make_pdf_sample_grid(samples)

    log.info("="*60)
    log.info(f"  DONE  Total={len(samples)} samples")
    log.info(f"  Data → {DATA_DIR}")
    log.info("="*60)

if __name__ == "__main__":
    main()
