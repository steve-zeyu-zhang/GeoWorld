<div align="center"><h1>GeoWorld: Geometric World Models<br>
<sub><sup><a href="https://cvpr.thecvf.com/Conferences/2026">CVPR 2026</a></sup></sub>
</h1>

[Zeyu Zhang](https://steve-zeyu-zhang.github.io/)<sup>1</sup>, [Danning Li](https://www.linkedin.com/in/danning-li-448039229/)<sup>2</sup>, [Ian Reid](https://mbzuai.ac.ae/study/faculty/ian-reid/)<sup>2</sup>, [Richard Hartley](https://users.cecs.anu.edu.au/~hartley/)<sup>1</sup>

<sup>1</sup>The Australian National University &nbsp; <sup>2</sup>Mohamed bin Zayed University of Artificial Intelligence

[![Website](https://img.shields.io/badge/Website-Demo-fedcba?style=flat-square)](https://steve-zeyu-zhang.github.io/GeoWorld/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.23058-b31b1b?style=flat-square)](https://arxiv.org/abs/2602.23058v2)
[![BibTeX](https://img.shields.io/badge/BibTeX-Citation-eeeeee?style=flat-square)](#citation)

</div>

_Energy-based predictive world models provide a powerful approach for multi-step visual planning by reasoning over latent energy landscapes rather than generating pixels. However, existing approaches face two major challenges: (i) their latent representations are typically learned in Euclidean space, neglecting the underlying geometric and hierarchical structure among states, and (ii) they struggle with long-horizon prediction, which leads to rapid degradation across extended rollouts. To address these challenges, we introduce GeoWorld, a geometric world model that preserves geometric structure and hierarchical relations through a Hyperbolic JEPA, which maps latent representations from Euclidean space onto hyperbolic manifolds. We further introduce Geometric Reinforcement Learning for energy-based optimization, enabling stable multi-step planning in hyperbolic latent space. Extensive experiments on CrossTask and COIN demonstrate around 3% SR improvement in 3-step planning and 2% SR improvement in 4-step planning compared to the state-of-the-art V-JEPA 2._

![GeoWorld planning](https://steve-zeyu-zhang.github.io/GeoWorld/static/images/planning.svg)

## Usage

### Environment

Python 3.10+, PyTorch 2.6, and CUDA 12.4 are used for GPU training. The original V-JEPA 2 encoder and action-conditioned transformer components are included with their upstream licenses; see [Model details](docs/MODEL.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[dev]'

python scripts/download_backbone.py --model vit_large \
  --output checkpoints/vjepa2-vitl.pt
```

### Data Preparation

Download the complete annotations and obtain the original videos for [CrossTask](https://github.com/DmZhukov/CrossTask) and [COIN](https://coin-dataset.github.io/). Keep each video's original identifier as its filename, such as `data/crosstask/videos/<video_id>.mp4` and `data/coin/videos/<video_id>.mp4`.

> [!NOTE]
> Some CrossTask and COIN video links are no longer available. Please contact the authors of the respective dataset papers to request the original videos for those entries.

```bash
python scripts/download_metadata.py --dataset all --output data/metadata

python scripts/prepare_data.py --dataset crosstask \
  --metadata data/metadata/crosstask --videos data/crosstask/videos \
  --output data/crosstask/prepared
python scripts/prepare_data.py --dataset coin \
  --metadata data/metadata/coin --videos data/coin/videos \
  --output data/coin/prepared
```

Preparation writes a fixed dataset vocabulary shared across splits: 105 CrossTask actions or 778 COIN actions. Missing videos are reported explicitly. [Data preparation](docs/DATA.md) describes source downloads, split selection, timestamp handling, and the manifest format. The default CrossTask development split follows its released localization split; use explicit planning splits for benchmark comparison.

Cache the frozen encoder features before training:

```bash
for split in train val; do
  python -m geoworld.extract --config configs/crosstask.yaml \
    --manifest "data/crosstask/prepared/${split}.jsonl" \
    --output "data/crosstask/features/${split}"
done
```

For COIN, use `configs/coin.yaml`, the `data/coin/` directories, and its train/test splits. Use `--observation-mode image` to cache endpoint images for image-conditioned planning; the default uses video clips. Feature extraction fails on unreadable videos by default; `--skip-errors` explicitly permits omissions and writes an error report. Datasets, features, checkpoints, and generated outputs are excluded from Git.

### Training

```bash
# Stage 1: supervised post-training of Hyperbolic JEPA.
python -m geoworld.train --stage sft --config configs/crosstask.yaml \
  --manifest data/crosstask/features/train/manifest.jsonl --output outputs/crosstask/sft

# Stage 2: Geometric Reinforcement Learning.
python -m geoworld.train --stage grl --config configs/crosstask.yaml \
  --manifest data/crosstask/features/train/manifest.jsonl \
  --initialize outputs/crosstask/sft/last.pt --output outputs/crosstask/grl
```

The default configuration uses a frozen ViT-L encoder and a 24-layer predictor with width 1024 and 16 attention heads. SFT uses global batch size 256, 94,500 optimizer steps, and an equal weighting of teacher forcing and two-step rollout losses. GRL uses global batch size 128, 25,000 steps, discount `0.99`, and triangle weight `0.1`. Both stages use AdamW with weight decay `0.04`.

For multiple GPUs, replace `python` with `torchrun --standalone --nproc_per_node=8`. `--batch-size` is global; `--micro-batch-size` controls the batch per GPU, with gradient accumulation filling the global batch. Use `--max-steps` for a short run and `--resume outputs/crosstask/sft/last.pt` to resume the same stage.

[Method details](docs/METHOD.md) cover the training objectives, offline GRL trajectories, and numerical behavior of the triangle-inequality term. Functional checks are separate from full-dataset benchmark evaluation.

### Inference

```bash
python -m geoworld.plan --checkpoint outputs/crosstask/grl/last.pt \
  --start path/to/start.jpg --goal path/to/goal.jpg --horizon 4 \
  --output outputs/crosstask/plan.json
```

Start and goal observations may also be videos. Use `--start-time` and `--goal-time` to specify clip endpoints in seconds. CEM searches the complete fixed dataset action vocabulary, using 800 candidates, 80 elites, and 10 refinement iterations. The JSON result contains action IDs, label names, and final-state hyperbolic energy. `--features path/to/trajectory.npy` can replace the start/goal inputs when encoder features are already cached.

```bash
python -m geoworld.evaluate --checkpoint outputs/crosstask/grl/last.pt \
  --manifest data/crosstask/features/val/manifest.jsonl --horizon 4 \
  --output outputs/crosstask/evaluation.json
```

Evaluation reports sequence success rate (SR), mean action accuracy (mAcc), and mean set intersection-over-union (mIoU), all as percentages. Run `python -m pytest -q` for mathematical, model, data, and execution checks.

## Citation

```bibtex
@article{zhang2026geoworld,
  title={GeoWorld: Geometric World Models},
  author={Zhang, Zeyu and Li, Danning and Reid, Ian and Hartley, Richard},
  journal={arXiv preprint arXiv:2602.23058},
  year={2026}
}
```

## Acknowledgements

We thank the authors of the following papers and open-source projects:

- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://github.com/facebookresearch/vjepa2) for pretrained video encoders and action-conditioned transformer components.
- [CrossTask: Cross-Task Weakly Supervised Learning From Instructional Videos](https://github.com/DmZhukov/CrossTask) for instructional videos and temporal action annotations.
- [COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis](https://coin-dataset.github.io/) for instructional videos and action labels.
- [PDPP: Projected Diffusion for Procedure Planning in Instructional Videos](https://github.com/MCG-NJU/PDPP) for the reference CrossTask action-label mapping.
