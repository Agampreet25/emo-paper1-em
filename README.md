# Learning Structured Emotional Dynamics in Latent Space

This repository contains part of the code for our Findings of EMNLP 2026 paper, *Learning Structured Emotional Dynamics in Latent Space via Trajectory Modeling and Controlled Transitions*.

It covers the first two stages of the framework, building the emotional latent space and tracing conversations as trajectories through it. The prediction, intervention and cross-architecture analyses described in the paper are not included here.

## Overview

- **Stage 1, latent space.** A BERT encoder and a small projection head map each utterance to a 32-dimensional emotional state, trained with multi-label supervision on GoEmotions (57,677 utterances, 28 emotions).
- **Stage 2, trajectories.** The trained model is frozen and used to project MELD dialogues (1,344 conversations, 13,620 utterances) into the same space, turning each conversation into a path z₁ → z₂ → … → z_T.

## Repository structure

| Path | What it does |
| --- | --- |
| `main.py` | Stage 1. Trains the model, extracts latent vectors, runs geometry analysis |
| `phase2.py` | Stage 2. Projects MELD into the space and analyses trajectories |
| `configs/config.py` | All hyperparameters and paths |
| `data/dataset.py` | GoEmotions loading and multi-label aggregation |
| `data/meld_loader.py` | MELD loading, grouped by dialogue and ordered by utterance |
| `models/` | Encoder, latent projection and linear classifier |
| `analysis/` | PCA, UMAP, clustering, centroids and plots |
| `utils/` | Training, extraction, trajectory metrics and reports |

Stage 1 writes to `outputs/` and Stage 2 writes to `outputs_phase2/`.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.x. A GPU is recommended for training.

## Data

Datasets are not included. Download them and place them as follows.

- **GoEmotions** CSVs in `data/raw/` (`goemotions_1.csv`, `goemotions_2.csv`, `goemotions_3.csv`)
- **MELD** CSVs in `data/meld/` (`train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`)

## Running

```bash
python main.py       # Stage 1, trains the model and builds the latent space
python phase2.py     # Stage 2, needs the Stage 1 checkpoint
```

Use `python main.py --skip-train` to reuse a saved checkpoint and `python phase2.py --skip-encode` to reuse saved trajectories.

## Citation

If you use this code or build on this work, please cite the paper.

```bibtex
@inproceedings{saini2026learning,
  title     = {Learning Structured Emotional Dynamics in Latent Space via Trajectory Modeling and Controlled Transitions},
  author    = {Saini, Agampreet and Abusalameh, Bayan and Singh, Rahul Kumar and Cambria, Erik},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  pages     = {33329--33354},
  year      = {2026},
  url       = {https://aclanthology.org/PLACEHOLDER}
}
```
