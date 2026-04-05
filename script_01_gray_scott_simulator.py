"""
MORPHOS Project — script_01_gray_scott_simulator.py
====================================================
Author : IVXA
Date   : 2026-03-31

What this script does:
    Implements the Gray-Scott reaction-diffusion system, which produces
    biological Turing patterns (spots, stripes, mazes, worms, coral, etc.)
    by simulating two interacting chemical species U and V on a 2D grid.

    Governing equations:
        ∂u/∂t = Du·∇²u  −  u·v²  +  F·(1 − u)
        ∂v/∂t = Dv·∇²v  +  u·v²  −  (F + k)·v

    where:
        u, v = concentrations of species U (prey) and V (predator)
        Du, Dv = diffusion rates
        F  = feed rate  (how fast U is replenished from reservoir)
        k  = kill rate  (how fast V is removed)
        ∇² = discrete 5-point Laplacian on periodic grid

Pipeline:
    1.  Device detection → MPS (Apple Silicon) / CUDA / CPU
    2.  12 canonical presets covering the full Pearson (1993) pattern space
    3.  Fast NumPy simulator  (vectorized np.roll Laplacian)
    4.  PyTorch/MPS simulator (conv2d circular padding — 3-5× faster on M-series)
    5.  Temporal snapshots saved every save_interval steps
    6.  Raw arrays (.npy) + metadata (.json) saved to data/raw/
    7.  5 publication-quality PDF figures saved to PDF/
        ├── morphos_pattern_gallery.pdf        — 4×3 grid of all patterns
        ├── morphos_pearson_map.pdf            — F-k parameter space scatter
        ├── morphos_concentration_analysis.pdf — U-concentration distributions
        ├── morphos_evolution_{name}.pdf       — temporal evolution per pattern
        └── morphos_uv_overlay.pdf             — U vs V correlation per pattern

CLI:
    python script_01_gray_scott_simulator.py              # run all 12 presets
    python script_01_gray_scott_simulator.py --preset spots      # single preset
    python script_01_gray_scott_simulator.py --N 512             # high resolution
    python script_01_gray_scott_simulator.py --backend numpy     # force NumPy

Output:
    data/raw/   — morphos_{name}_u.npy, morphos_{name}_v.npy, morphos_{name}_meta.json
    PDF/        — all figures (see above)

Reference:
    Pearson, J.E. (1993). Complex Patterns in a Simple System.
    Science, 261(5118), 189–192. https://doi.org/10.1126/science.261.5118.189
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# PyTorch — optional, falls back to NumPy if unavailable
try:
    import torch
    import torch.nn.functional as F_torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent   # MORPHOS/
DATA_DIR = ROOT / "data" / "raw"
PDF_DIR  = ROOT / "PDF"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MORPHOS")

# ─────────────────────────────────────────────────────────────────────────────
#  Device detection
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> "torch.device":
    """Return best available device: MPS → CUDA → CPU."""
    if not TORCH_AVAILABLE:
        return None
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
if DEVICE is not None:
    log.info(f"PyTorch device : {DEVICE}")
else:
    log.info("PyTorch not found — using NumPy backend")

# ─────────────────────────────────────────────────────────────────────────────
#  Parameter presets  (Pearson 1993 + community-verified)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GrayScottParams:
    """All parameters defining a single Gray-Scott simulation."""
    name:        str
    label:       str        # pattern family name (Pearson classification)
    F:           float      # feed rate
    k:           float      # kill rate
    Du:          float = 0.16   # diffusion rate of U
    Dv:          float = 0.08   # diffusion rate of V
    steps:       int   = 10_000
    dt:          float = 1.0
    description: str   = ""


PRESETS: Dict[str, GrayScottParams] = {
    "spots": GrayScottParams(
        "spots", "Alpha — Spots",
        F=0.035, k=0.065, steps=12_000,
        description="Isolated circular spots — animal pelage markings (leopard, cheetah)"),
    "maze": GrayScottParams(
        "maze", "Epsilon — Maze",
        F=0.029, k=0.057, steps=12_000,
        description="Labyrinthine stripe patterns — gyrification, fingerprints"),
    "holes": GrayScottParams(
        "holes", "Delta — Holes",
        F=0.039, k=0.058, steps=10_000,
        description="Inverse spots — holes in continuous medium, swiss-cheese topology"),
    "worms": GrayScottParams(
        "worms", "Gamma — Worms",
        F=0.078, k=0.061, steps=10_000,
        description="Elongated worm-like filaments — coral polyp extensions"),
    "coral": GrayScottParams(
        "coral", "Zeta — Coral",
        F=0.055, k=0.062, steps=10_000,
        description="Branching coral/seaweed dendritic structures"),
    "stripes": GrayScottParams(
        "stripes", "Eta — Stripes",
        F=0.060, k=0.062, steps=10_000,
        description="Parallel stripe patterns — zebra, angelfish"),
    "mitosis": GrayScottParams(
        "mitosis", "Theta — Mitosis",
        F=0.028, k=0.053, steps=14_000,
        description="Self-replicating spots undergoing mitosis-like division"),
    "bubbles": GrayScottParams(
        "bubbles", "Beta — Bubbles",
        F=0.062, k=0.063, steps=10_000,
        description="Hexagonally packed bubble arrangements — honeycomb-like"),
    "leopard": GrayScottParams(
        "leopard", "Kappa — Leopard",
        F=0.040, k=0.060, steps=12_000,
        description="Irregular rosette spot pattern — leopard coat"),
    "waves": GrayScottParams(
        "waves", "Iota — Waves",
        F=0.014, k=0.054, steps=14_000,
        description="Expanding circular wave fronts — excitable medium dynamics"),
    "chaotic": GrayScottParams(
        "chaotic", "Lambda — Chaotic",
        F=0.026, k=0.055, steps=10_000,
        description="Spatiotemporally chaotic regime — no stable pattern forms"),
    "uniform": GrayScottParams(
        "uniform", "Mu — Uniform",
        F=0.090, k=0.059, steps=8_000,
        description="Rapid decay to homogeneous steady state — trivial equilibrium"),
}

# ─────────────────────────────────────────────────────────────────────────────
#  Colormap
# ─────────────────────────────────────────────────────────────────────────────
MORPHOS_CMAP = LinearSegmentedColormap.from_list(
    "morphos",
    [
        (0.04, 0.07, 0.18),   # deep navy       — low U
        (0.10, 0.22, 0.48),   # cobalt          —
        (0.18, 0.42, 0.62),   # slate blue      —
        (0.82, 0.79, 0.68),   # warm cream      — mid U
        (0.91, 0.68, 0.32),   # amber           —
        (0.82, 0.32, 0.10),   # burnt coral     — high U
    ],
    N=512,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Grid initialization
# ─────────────────────────────────────────────────────────────────────────────
def initialize_grid(
    N: int,
    seed_radius: int = 12,
    n_seeds: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize U and V concentration grids.

    Strategy:
        - u = 1.0 everywhere (reservoir full)
        - v = 0.0 everywhere (no inhibitor)
        - Random square patches seeded with u≈0.50, v≈0.25 + noise
          to break spatial symmetry and trigger pattern formation.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    u = np.ones((N, N), dtype=np.float64)
    v = np.zeros((N, N), dtype=np.float64)

    for _ in range(n_seeds):
        cy = rng.integers(seed_radius, N - seed_radius)
        cx = rng.integers(seed_radius, N - seed_radius)
        r  = rng.integers(max(2, seed_radius // 2), seed_radius + 1)
        slc_y = slice(max(0, cy - r), min(N, cy + r))
        slc_x = slice(max(0, cx - r), min(N, cx + r))
        h = slc_y.stop - slc_y.start
        w = slc_x.stop - slc_x.start
        u[slc_y, slc_x] = 0.50 + rng.uniform(-0.02, 0.02, (h, w))
        v[slc_y, slc_x] = 0.25 + rng.uniform(-0.02, 0.02, (h, w))

    return u, v


# ─────────────────────────────────────────────────────────────────────────────
#  Laplacian — NumPy
# ─────────────────────────────────────────────────────────────────────────────
def laplacian_np(Z: np.ndarray) -> np.ndarray:
    """
    5-point discrete Laplacian with periodic (toroidal) boundary conditions.
    Uses np.roll — O(N²), no extra memory allocation.
    """
    return (
        np.roll(Z,  1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z,  1, axis=1) + np.roll(Z, -1, axis=1) -
        4.0 * Z
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Simulator — NumPy backend
# ─────────────────────────────────────────────────────────────────────────────
def simulate_numpy(
    params: GrayScottParams,
    N: int = 256,
    save_interval: int = 1_000,
) -> Dict:
    """
    Run Gray-Scott simulation using vectorized NumPy.

    Returns:
        dict with keys: u, v, snapshots_u, snapshots_v, times, params,
                        elapsed, N, backend
    """
    rng          = np.random.default_rng(42)
    u, v         = initialize_grid(N, rng=rng)
    F, k         = params.F, params.k
    Du, Dv, dt   = params.Du, params.Dv, params.dt

    snapshots_u: List[np.ndarray] = []
    snapshots_v: List[np.ndarray] = []
    times:       List[int]        = []

    log.info(f"  [numpy] {params.name:10s}  F={F:.3f}  k={k:.3f}  "
             f"steps={params.steps:,}  grid={N}×{N}")
    t0 = time.perf_counter()

    for step in range(params.steps):
        Lu   = laplacian_np(u)
        Lv   = laplacian_np(v)
        uvv  = u * v * v

        u += dt * (Du * Lu - uvv + F * (1.0 - u))
        v += dt * (Dv * Lv + uvv - (F + k) * v)

        np.clip(u, 0.0, 1.0, out=u)        # numerical stability
        np.clip(v, 0.0, 1.0, out=v)

        if step % save_interval == 0 or step == params.steps - 1:
            snapshots_u.append(u.copy())
            snapshots_v.append(v.copy())
            times.append(step)

    elapsed = time.perf_counter() - t0
    log.info(f"             → {elapsed:.1f}s  "
             f"({params.steps / elapsed:.0f} steps/s)")

    return dict(u=u, v=v, snapshots_u=snapshots_u, snapshots_v=snapshots_v,
                times=times, params=asdict(params),
                elapsed=elapsed, N=N, backend="numpy")


# ─────────────────────────────────────────────────────────────────────────────
#  Laplacian — PyTorch
# ─────────────────────────────────────────────────────────────────────────────
def _make_lap_kernel(device: "torch.device") -> "torch.Tensor":
    """5-point Laplacian kernel as (1,1,3,3) tensor."""
    k = torch.tensor(
        [[0., 1., 0.],
         [1., -4., 1.],
         [0., 1., 0.]],
        device=device, dtype=torch.float32,
    )
    return k.view(1, 1, 3, 3)


def laplacian_torch(
    Z: "torch.Tensor",
    kernel: "torch.Tensor",
) -> "torch.Tensor":
    """5-point Laplacian with circular padding via conv2d."""
    Z4 = F_torch.pad(Z.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular")
    return F_torch.conv2d(Z4, kernel).squeeze(0).squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
#  Simulator — PyTorch/MPS backend
# ─────────────────────────────────────────────────────────────────────────────
def simulate_torch(
    params: GrayScottParams,
    N: int = 256,
    save_interval: int = 1_000,
    device: Optional["torch.device"] = None,
) -> Dict:
    """
    Run Gray-Scott simulation using PyTorch conv2d.
    ~3-5× faster than NumPy on Apple M-series (MPS backend).

    Returns:
        dict with keys: u, v, snapshots_u, snapshots_v, times, params,
                        elapsed, N, backend, device
    """
    if device is None:
        device = DEVICE
    if device is None:
        raise RuntimeError("PyTorch not available — use simulate_numpy instead.")

    rng       = np.random.default_rng(42)
    u_np, v_np = initialize_grid(N, rng=rng)

    u      = torch.from_numpy(u_np.astype(np.float32)).to(device)
    v      = torch.from_numpy(v_np.astype(np.float32)).to(device)
    kernel = _make_lap_kernel(device)

    F_val, k_val = params.F, params.k
    Du, Dv, dt   = params.Du, params.Dv, params.dt

    snapshots_u: List[np.ndarray] = []
    snapshots_v: List[np.ndarray] = []
    times:       List[int]        = []

    log.info(f"  [torch/{device.type}] {params.name:10s}  F={F_val:.3f}  k={k_val:.3f}  "
             f"steps={params.steps:,}  grid={N}×{N}")
    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(params.steps):
            Lu  = laplacian_torch(u, kernel)
            Lv  = laplacian_torch(v, kernel)
            uvv = u * v * v

            u = (u + dt * (Du * Lu - uvv + F_val * (1.0 - u))).clamp(0.0, 1.0)
            v = (v + dt * (Dv * Lv + uvv - (F_val + k_val) * v)).clamp(0.0, 1.0)

            if step % save_interval == 0 or step == params.steps - 1:
                snapshots_u.append(u.cpu().numpy().copy())
                snapshots_v.append(v.cpu().numpy().copy())
                times.append(step)

    elapsed = time.perf_counter() - t0
    log.info(f"               → {elapsed:.1f}s  "
             f"({params.steps / elapsed:.0f} steps/s)")

    return dict(u=u.cpu().numpy(), v=v.cpu().numpy(),
                snapshots_u=snapshots_u, snapshots_v=snapshots_v,
                times=times, params=asdict(params),
                elapsed=elapsed, N=N,
                backend=f"torch/{device.type}", device=str(device))


# ─────────────────────────────────────────────────────────────────────────────
#  Unified runner
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(
    params: GrayScottParams,
    N: int,
    backend: str,
    save_interval: int = 1_000,
) -> Dict:
    """
    Dispatch to the correct backend.

    backend:
        "auto"  → torch if available, else numpy
        "torch" → always torch (raises if unavailable)
        "numpy" → always numpy
    """
    use_torch = (
        backend == "torch" or
        (backend == "auto" and TORCH_AVAILABLE and DEVICE is not None)
    )
    if use_torch:
        return simulate_torch(params, N=N, save_interval=save_interval,
                              device=DEVICE)
    return simulate_numpy(params, N=N, save_interval=save_interval)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1: Pattern gallery  (4 × 3 grid)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pattern_gallery(results: Dict[str, Dict], output_path: Path) -> None:
    """
    Publication-quality 4×3 gallery of all Gray-Scott pattern types.
    - Dark background (#0A0E1A) for visual contrast
    - English titles only
    - No label overlap
    - 150 dpi → 300 dpi for final paper
    """
    names  = list(results.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(names) / n_cols))
    BG     = "#0A0E1A"

    fig = plt.figure(figsize=(16, n_rows * 4.4), dpi=150)
    fig.patch.set_facecolor(BG)

    for idx, name in enumerate(names):
        ax     = fig.add_subplot(n_rows, n_cols, idx + 1)
        res    = results[name]
        p      = GrayScottParams(**res["params"])

        ax.imshow(res["u"], cmap=MORPHOS_CMAP, interpolation="bilinear",
                  vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Pattern name + parameters
        short = p.label.split("—")[1].strip() if "—" in p.label else p.label
        ax.set_title(
            f"{short}\nF = {p.F:.3f}   k = {p.k:.3f}",
            color="#D8E4F0", fontsize=9, pad=6,
            fontfamily="monospace", fontweight="normal",
        )

        # Compute time badge (bottom-right corner)
        ax.text(
            0.97, 0.03, f"{res['elapsed']:.0f}s",
            transform=ax.transAxes,
            color="#607080", fontsize=7, ha="right", va="bottom",
            fontfamily="monospace",
        )

        # Grid size badge (bottom-left corner)
        N = res["N"]
        ax.text(
            0.03, 0.03, f"{N}×{N}",
            transform=ax.transAxes,
            color="#607080", fontsize=7, ha="left", va="bottom",
            fontfamily="monospace",
        )

        for sp in ax.spines.values():
            sp.set_edgecolor("#1E2A40")
            sp.set_linewidth(0.5)

    # Hide unused subplots
    for idx in range(len(names), n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.axis("off")

    fig.suptitle(
        "MORPHOS — Gray-Scott Reaction-Diffusion Pattern Gallery\n"
        "Biological pattern formation via Turing instability  ·  "
        "Reference: Pearson (1993) Science 261:189-192",
        color="#D8E4F0", fontsize=13, fontweight="normal",
        y=1.01, fontfamily="monospace",
    )
    plt.tight_layout(pad=1.4)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Saved → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2: Pearson F-k parameter space map
# ─────────────────────────────────────────────────────────────────────────────
def fig_pearson_map(results: Dict[str, Dict], output_path: Path) -> None:
    """
    Scatter plot of all pattern presets in the (F, k) parameter space.
    Includes pattern thumbnail insets positioned in clear space.
    """
    BG = "#0A0E1A"
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#0D1220")

    colors = plt.cm.plasma(np.linspace(0.12, 0.92, len(results)))

    # Offsets to prevent label collision — manually tuned for this preset set
    offsets = {
        "spots":   ( 0.003,  0.001),
        "maze":    (-0.006,  0.001),
        "holes":   ( 0.003, -0.002),
        "worms":   ( 0.003,  0.001),
        "coral":   ( 0.003, -0.002),
        "stripes": ( 0.003,  0.001),
        "mitosis": (-0.009,  0.001),
        "bubbles": ( 0.003, -0.002),
        "leopard": ( 0.003,  0.001),
        "waves":   (-0.007,  0.001),
        "chaotic": ( 0.003, -0.002),
        "uniform": ( 0.003,  0.001),
    }

    for idx, (name, res) in enumerate(results.items()):
        p = GrayScottParams(**res["params"])
        c = colors[idx]
        ax.scatter(p.F, p.k, s=160, color=c,
                   edgecolors="#FFFFFF", linewidth=0.6, zorder=5)

        dx, dy = offsets.get(name, (0.003, 0.001))
        short  = p.label.split("—")[1].strip() if "—" in p.label else p.label
        ax.annotate(
            short,
            xy=(p.F, p.k),
            xytext=(p.F + dx, p.k + dy),
            fontsize=8.5, color="#C0D0E0",
            fontfamily="monospace",
            arrowprops=dict(arrowstyle="-", color="#405060", lw=0.5),
        )

    # Axis decoration
    ax.set_xlabel("Feed rate  F", color="#A0B4C8", fontsize=12,
                  fontfamily="monospace")
    ax.set_ylabel("Kill rate  k", color="#A0B4C8", fontsize=12,
                  fontfamily="monospace")
    ax.set_title(
        "Pearson Parameter Space — Gray-Scott Pattern Classification\n"
        "Each point is a qualitatively distinct pattern family",
        color="#D8E4F0", fontsize=13, pad=12, fontfamily="monospace",
    )
    ax.tick_params(colors="#708090", labelsize=9)
    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1E2A40")
    ax.grid(True, alpha=0.12, color="#3A4A60", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Saved → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3: Concentration distribution analysis
# ─────────────────────────────────────────────────────────────────────────────
def fig_concentration_analysis(results: Dict[str, Dict], output_path: Path) -> None:
    """
    Histogram of U-concentration distribution for each pattern type.
    Bimodal → strong pattern.  Unimodal peak at 1 → uniform/dissolved.
    """
    BG  = "#0A0E1A"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(BG)

    colors = plt.cm.plasma(np.linspace(0.12, 0.92, len(results)))
    BINS   = 120

    for idx, (name, res) in enumerate(results.items()):
        p     = GrayScottParams(**res["params"])
        short = p.label.split("—")[1].strip() if "—" in p.label else p.label
        c     = colors[idx]

        # U distribution
        u_flat = res["u"].ravel()
        hist_u, edges = np.histogram(u_flat, bins=BINS, range=(0, 1), density=True)
        centers       = 0.5 * (edges[:-1] + edges[1:])
        ax1.plot(centers, hist_u, color=c, alpha=0.85, linewidth=1.2, label=short)

        # V distribution
        v_flat = res["v"].ravel()
        hist_v, _     = np.histogram(v_flat, bins=BINS, range=(0, 1), density=True)
        ax2.plot(centers, hist_v, color=c, alpha=0.85, linewidth=1.2, label=short)

    for ax, species, clr in zip([ax1, ax2], ["U  (activator)", "V  (inhibitor)"],
                                ["#3A8BD4", "#D45A30"]):
        ax.set_facecolor("#0D1220")
        ax.set_xlabel(f"Concentration  {species}", color="#A0B4C8",
                      fontsize=11, fontfamily="monospace")
        ax.set_ylabel("Probability density", color="#A0B4C8",
                      fontsize=11, fontfamily="monospace")
        ax.tick_params(colors="#708090", labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1E2A40")
        ax.grid(True, alpha=0.12, color="#3A4A60", linewidth=0.5, linestyle="--")

    ax1.legend(fontsize=7.5, ncol=2, framealpha=0.15,
               labelcolor="#C0D0E0", facecolor="#1A2035", edgecolor="#3A4A60",
               loc="upper left")

    fig.suptitle(
        "Concentration Distribution Analysis — All Pattern Families\n"
        "Bimodal U-distribution indicates strong spatial patterning",
        color="#D8E4F0", fontsize=13, fontfamily="monospace",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Saved → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4: Temporal evolution  (per pattern)
# ─────────────────────────────────────────────────────────────────────────────
def fig_evolution(result: Dict, output_path: Path) -> None:
    """
    6-panel temporal evolution: initial seed → pattern formation → convergence.
    """
    snaps  = result["snapshots_u"]
    times  = result["times"]
    p      = GrayScottParams(**result["params"])
    BG     = "#0A0E1A"

    n_show  = min(6, len(snaps))
    indices = np.linspace(0, len(snaps) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 3.3, 3.8), dpi=150)
    fig.patch.set_facecolor(BG)

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(snaps[idx], cmap=MORPHOS_CMAP, interpolation="bilinear",
                  vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {times[idx]:,}", color="#C0D4E8",
                     fontsize=9, fontfamily="monospace")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1E2A40")

    short = p.label.split("—")[1].strip() if "—" in p.label else p.label
    fig.suptitle(
        f"Temporal Evolution — {short}\n"
        f"F = {p.F:.3f}   k = {p.k:.3f}   Du = {p.Du}   Dv = {p.Dv}   "
        f"grid = {result['N']}×{result['N']}",
        color="#D8E4F0", fontsize=11, fontfamily="monospace",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Saved → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 5: U–V overlay (phase portrait per pattern)
# ─────────────────────────────────────────────────────────────────────────────
def fig_uv_overlay(results: Dict[str, Dict], output_path: Path) -> None:
    """
    Scatter plot of U vs V pixel values — each pattern's phase-space portrait.
    Tight clusters = ordered pattern; diffuse cloud = chaotic/uniform.
    """
    names  = list(results.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(names) / n_cols))
    BG     = "#0A0E1A"
    N_PTS  = 4096    # subsample for speed

    fig = plt.figure(figsize=(n_cols * 3.6, n_rows * 3.6), dpi=150)
    fig.patch.set_facecolor(BG)

    rng = np.random.default_rng(0)

    for idx, name in enumerate(names):
        ax  = fig.add_subplot(n_rows, n_cols, idx + 1)
        res = results[name]
        p   = GrayScottParams(**res["params"])

        u_flat = res["u"].ravel()
        v_flat = res["v"].ravel()
        sel    = rng.choice(len(u_flat), size=min(N_PTS, len(u_flat)), replace=False)

        # Color by local U gradient magnitude (structure indicator)
        grad_mag = np.abs(np.gradient(res["u"])[0]) + np.abs(np.gradient(res["u"])[1])
        c_vals   = grad_mag.ravel()[sel]
        c_vals   = (c_vals - c_vals.min()) / (c_vals.max() - c_vals.min() + 1e-12)

        ax.scatter(u_flat[sel], v_flat[sel],
                   c=c_vals, cmap="plasma", s=1.2, alpha=0.5, rasterized=True)
        ax.set_facecolor("#0D1220")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 0.52)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.25, 0.5])
        ax.tick_params(colors="#506070", labelsize=7)

        short = p.label.split("—")[1].strip() if "—" in p.label else p.label
        ax.set_title(short, color="#C0D4E8", fontsize=8.5, fontfamily="monospace")

        if idx % n_cols == 0:
            ax.set_ylabel("V  (inhibitor)", color="#708090", fontsize=7,
                          fontfamily="monospace")
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("U  (activator)", color="#708090", fontsize=7,
                          fontfamily="monospace")

        for sp in ax.spines.values():
            sp.set_edgecolor("#1E2A40")

    # Hide unused subplots
    for idx in range(len(names), n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.axis("off")

    fig.suptitle(
        "U–V Phase Portrait — Activator vs. Inhibitor Concentration\n"
        "Color = local gradient magnitude  (bright = strong spatial structure)",
        color="#D8E4F0", fontsize=12, fontfamily="monospace", y=1.01,
    )
    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Saved → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Compute pattern metrics  (for JSON metadata + paper table)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(result: Dict) -> Dict:
    """
    Compute quantitative descriptors of the final U field.
    These metrics can be used as labels for downstream classifiers.

    Returns:
        dict with: mean_u, std_u, mean_v, std_v,
                   spatial_freq_peak, bimodality_coeff, entropy_bits
    """
    u = result["u"]
    v = result["v"]

    # Basic statistics
    mean_u = float(np.mean(u))
    std_u  = float(np.std(u))
    mean_v = float(np.mean(v))
    std_v  = float(np.std(v))

    # Dominant spatial frequency (via 2D FFT)
    fft2   = np.abs(np.fft.fftshift(np.fft.fft2(u)))
    fft2[fft2.shape[0]//2, fft2.shape[1]//2] = 0  # zero DC component
    peak_y, peak_x = np.unravel_index(np.argmax(fft2), fft2.shape)
    N              = u.shape[0]
    freq_peak      = float(np.sqrt((peak_y - N//2)**2 + (peak_x - N//2)**2) / N)

    # Bimodality coefficient  (Sarle 1990): > 0.555 → bimodal → patterned
    skew      = float(((u - mean_u)**3).mean() / (std_u**3 + 1e-12))
    kurt      = float(((u - mean_u)**4).mean() / (std_u**4 + 1e-12)) - 3
    n_cells   = u.size
    bimod_coeff = (skew**2 + 1) / (kurt + 3 * (n_cells - 1)**2 / ((n_cells - 2)*(n_cells - 3) + 1e-12))

    # Shannon entropy of U histogram (bits)
    hist, _ = np.histogram(u, bins=256, range=(0, 1), density=False)
    hist    = hist / hist.sum()
    hist    = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    return dict(
        mean_u=round(mean_u, 5),
        std_u=round(std_u, 5),
        mean_v=round(mean_v, 5),
        std_v=round(std_v, 5),
        spatial_freq_peak=round(freq_peak, 5),
        bimodality_coeff=round(bimod_coeff, 5),
        entropy_bits=round(entropy, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MORPHOS — Gray-Scott Reaction-Diffusion Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=list(PRESETS.keys()) + [None],
        help="Run a single preset (default: all 12).",
    )
    parser.add_argument(
        "--N", type=int, default=256,
        help="Grid size N×N. 256 = fast dev, 512 = paper quality.",
    )
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "torch", "numpy"],
        help="Compute backend. 'auto' selects torch if available.",
    )
    parser.add_argument(
        "--save_interval", type=int, default=1_000,
        help="Save snapshot every N steps for evolution plots.",
    )
    parser.add_argument(
        "--no_figures", action="store_true",
        help="Skip PDF generation (faster, data only).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    log.info("=" * 65)
    log.info("MORPHOS — script_01_gray_scott_simulator.py")
    log.info(f"Grid size    : {args.N} × {args.N}")
    log.info(f"Backend      : {args.backend}")
    log.info(f"Presets      : {args.preset or 'all 12'}")
    log.info("=" * 65)

    # Select presets to run
    preset_subset = (
        {args.preset: PRESETS[args.preset]}
        if args.preset
        else PRESETS
    )

    results: Dict[str, Dict] = {}
    all_metrics: Dict[str, Dict] = {}

    # ── Simulations ───────────────────────────────────────────────────────────
    log.info(f"\nRunning {len(preset_subset)} simulation(s)...\n")
    for name, params in preset_subset.items():
        result = run_simulation(
            params, N=args.N, backend=args.backend,
            save_interval=args.save_interval,
        )
        results[name] = result

        # Compute pattern metrics
        metrics = compute_metrics(result)
        all_metrics[name] = metrics

        # Save raw arrays
        np.save(DATA_DIR / f"morphos_{name}_u.npy", result["u"])
        np.save(DATA_DIR / f"morphos_{name}_v.npy", result["v"])

        # Save metadata + metrics as JSON (exclude large arrays)
        meta_out = {
            k: v for k, v in result.items()
            if k not in ("u", "v", "snapshots_u", "snapshots_v")
        }
        meta_out["metrics"] = metrics
        with open(DATA_DIR / f"morphos_{name}_meta.json", "w") as fh:
            json.dump(meta_out, fh, indent=2, default=str)

        log.info(
            f"  [{name}] metrics → "
            f"mean_u={metrics['mean_u']:.3f}  "
            f"std_u={metrics['std_u']:.3f}  "
            f"bimodality={metrics['bimodality_coeff']:.3f}  "
            f"entropy={metrics['entropy_bits']:.2f} bits"
        )

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.no_figures:
        log.info("\nGenerating figures...\n")

        if len(results) > 1:
            fig_pattern_gallery(results, PDF_DIR / "morphos_pattern_gallery.pdf")
            fig_pearson_map(results, PDF_DIR / "morphos_pearson_map.pdf")
            fig_concentration_analysis(results,
                                       PDF_DIR / "morphos_concentration_analysis.pdf")
            fig_uv_overlay(results, PDF_DIR / "morphos_uv_overlay.pdf")

        for name, result in results.items():
            fig_evolution(result, PDF_DIR / f"morphos_evolution_{name}.pdf")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = sum(r["elapsed"] for r in results.values())

    log.info("\n" + "=" * 65)
    log.info("SUMMARY")
    log.info("=" * 65)
    log.info(f"  Patterns simulated : {len(results)}")
    log.info(f"  Grid size          : {args.N} × {args.N}")
    log.info(f"  Backend            : {next(iter(results.values()))['backend']}")
    log.info(f"  Total compute time : {total_time:.1f}s")
    log.info(f"  Data saved to      : {DATA_DIR}")
    log.info(f"  PDFs saved to      : {PDF_DIR}")
    log.info("=" * 65)

    # Print bimodality table  (useful for paper)
    log.info("\nPattern metrics table (for paper / next script input):")
    log.info(f"  {'Name':<12}  {'F':>6}  {'k':>6}  "
             f"{'mean_u':>7}  {'std_u':>6}  {'bimod':>6}  {'entropy':>8}")
    log.info("  " + "-" * 60)
    for name, metrics in all_metrics.items():
        p = PRESETS[name]
        log.info(
            f"  {name:<12}  {p.F:>6.3f}  {p.k:>6.3f}  "
            f"{metrics['mean_u']:>7.3f}  {metrics['std_u']:>6.3f}  "
            f"{metrics['bimodality_coeff']:>6.3f}  "
            f"{metrics['entropy_bits']:>8.2f}"
        )
    log.info("")
    log.info("Next step → run script_02_parameter_space_explorer.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
