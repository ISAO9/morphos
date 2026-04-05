# MORPHOS — PhysaNet-GNN

**PhysaNet-GNN: A *Physarum polycephalum*-Inspired Graph Neural Network for Turing Pattern Classification**

> Manuscript submitted to *Neural Networks* (Elsevier, Engineering and Applications section), 2026.  
> Author: Isao Kurosawa — isao.kurosawa [at] ivxa.ai

---

## Overview

This repository contains the code and dataset required to reproduce all scientific results reported in the manuscript. PhysaNet-GNN is a graph neural network whose edge weight update rule is derived from the tube-reinforcement dynamics of the slime mold *Physarum polycephalum*: tubes carrying high cytoplasmic flow thicken and persist, while underutilised tubes atrophy and disappear. This biological feedback law is translated into a differentiable, per-layer message-passing operation governed by three learnable scalars (α, μ, β), and applied to the classification of six Gray–Scott reaction-diffusion pattern families.

**Key results (5-seed evaluation):**

| Model | Macro-F1 | Parameters |
|---|---|---|
| PhysaNet-GNN L=6 (proposed) | 95.17 ± 0.70% | 173 K |
| CNN baseline (SE-ResNet18) | 91.79 ± 1.67% | 11.4 M |
| GAT | 92.67 ± 4.25% | ~170 K |

PhysaNet-GNN significantly outperforms the CNN baseline (Welch *t*(5.4) = 3.74, *p* = 0.012) with 66× fewer parameters. The learned μ exponent shows statistically confirmed layer-wise specialisation consistent with temporal *Physarum* biology (5/6 layers significant, Bonferroni-corrected, Cohen's *d* up to 17.2).

---

## Repository Structure

```
morphos/
├── src/
│   ├── script_01_gray_scott_simulator.py   # Gray–Scott PDE simulator
│   ├── script_14b_dataset_regenerate.py    # Dataset generation (586 samples, 6 classes)
│   └── script_16_physanet_gnn.py           # PhysaNet-GNN training & evaluation
├── data/
│   ├── raw/                                # Raw simulation outputs (.npy)
│   └── dataset/                            # Train/val/test splits
│       ├── train/
│       ├── val/
│       └── test/
├── models/                                 # Saved checkpoints (populated at runtime)
├── PDF/                                    # Output figures (populated at runtime)
├── LICENSE
└── README.md
```

The pre-trained model weights (`morphos_16_best.pt`) will be added to the `models/` directory upon acceptance of the manuscript.

---

## Requirements

This project was developed with Python 3.12 and [uv](https://github.com/astral-sh/uv) for environment management. All experiments were run on Apple Silicon (M-series) via PyTorch MPS; CPU compatibility has been verified.

```bash
# Recommended: install uv first
pip install uv

# Clone the repository and create the environment
git clone https://github.com/ivxa/morphos
cd morphos
uv venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

uv pip install torch numpy scipy matplotlib scikit-learn
```

---

## Reproducing the Results

The three scripts below are sufficient to reproduce all results reported in Tables 1–3 of the manuscript.

**Step 1 — Simulate Gray–Scott patterns**

```bash
python src/script_01_gray_scott_simulator.py
```

This generates the raw simulation outputs for all six pattern families (spots, maze, holes, stripes, coral, leopard) and saves them to `data/raw/`. Output PDFs (parameter space map, pattern gallery, temporal evolution) are written to `PDF/`.

**Step 2 — Build the dataset**

```bash
python src/script_14b_dataset_regenerate.py
```

This constructs the 586-sample benchmark dataset with a 70/15/15% stratified train/val/test split, applying Von Neumann stability gating and spatial-bias verification. The split is saved to `data/dataset/`.

**Step 3 — Train and evaluate PhysaNet-GNN**

```bash
python src/script_16_physanet_gnn.py
```

This trains the PhysaNet-GNN (L=6, d=64, 173K parameters) for up to 200 epochs with early stopping, saves the best checkpoint to `models/morphos_16_best.pt`, and generates all evaluation figures (training curves, confusion matrices, Physarum edge weight visualisation, CNN comparison) to `PDF/`.

Optional arguments:

```bash
python src/script_16_physanet_gnn.py --seeds 5 --epochs 200 --batch 32
```

---

## Dataset

The Gray–Scott benchmark dataset (586 samples, 6 classes, 128×128 grid) is fully reproducible by running scripts 01 and 14b as described above. No proprietary or externally sourced data is used; all samples are generated from publicly documented simulation parameters (Pearson, 1993).

An extended multi-system dataset incorporating FitzHugh–Nagumo patterns (720 samples, 9 classes, two PDE systems) is available upon reasonable request to the corresponding author.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{kurosawa2026physanet,
  title         = {PhysaNet-GNN: A {\it Physarum polycephalum}-Inspired Graph Neural
                   Network for Turing Pattern Classification},
  author        = {Kurosawa, Isao},
  year          = {2026},
  howpublished  = {IVXA. \url{https://github.com/ivxa/morphos}},
  note          = {Under review at Neural Networks (Elsevier)}
}
```

This entry will be updated with volume, issue, and DOI upon publication.

---

## License

The code and dataset in this repository are released under the **MIT License**. See [`LICENSE`](LICENSE) for the full text.

```
MIT License

Copyright (c) 2026 IVXA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

> **Commercial licensing:** For applications beyond the scope of the MIT License — including integration into commercial products, proprietary pipelines, or services — please contact **isao.kurosawa [at] ivxa.ai** to discuss licensing arrangements.

---

## Contact

Isao Kurosawa  
IVXA — AI Research & Consulting  
isao.kurosawa [at] ivxa.ai
