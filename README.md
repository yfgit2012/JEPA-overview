# JEPA — Joint-Embedding Predictive Architectures

A hands-on collection of notebooks exploring **JEPA**, the self-supervised
learning paradigm introduced by Yann LeCun and Meta AI, across three modalities:
**images (I-JEPA)**, **video (V-JEPA)**, and **vision-language (VL-JEPA)**.

Each notebook is runnable, richly commented, and pairs with a focused doc page
that explains the concepts and walks through the key code.

---

## What is JEPA?

**JEPA (Joint-Embedding Predictive Architecture)** is a family of self-supervised
models that learn by **predicting the representation of one part of an input from
another part — in an abstract latent space, not in pixel/token space.**

That single design choice is what sets JEPA apart:

| Approach | What it predicts | Consequence |
|---|---|---|
| **Generative** (MAE, autoregressive) | Raw pixels / tokens | Wastes capacity on unpredictable low-level detail |
| **Contrastive** (SimCLR, CLIP) | View invariance via negatives | Needs careful augmentations / large negative batches |
| **JEPA** | **Latent representations** of masked targets | Learns high-level semantics; no pixel reconstruction, no hand-crafted negatives |

### The core recipe

```
             ┌─────────────────┐
  context ──▶│ context encoder │──┐
             └─────────────────┘  │      ┌───────────┐
                                   ├────▶ │ predictor │──▶ predicted latents ─┐
             (target locations) ──┘      └───────────┘                       │
                                                                             ▼
             ┌─────────────────┐                                        [ loss ]
  target  ──▶│ target encoder  │───────────────────▶ target latents ────────┘
             │  (EMA of ctx)   │
             └─────────────────┘
```

1. **Context encoder** embeds the visible part of the input.
2. **Predictor** guesses the latent representations of the masked / target parts.
3. **Target encoder** — an exponential-moving-average (EMA) copy of the context
   encoder — produces the "ground-truth" latents. Its EMA nature prevents
   *representation collapse* (the trivial solution of outputting a constant).
4. Training minimizes the distance between predicted and target latents.

Because the loss lives in latent space, the model is free to **discard
unpredictable detail** and focus on semantically meaningful structure.

---

## The three variants in this repo

### 🖼️ I-JEPA — Images
Predicts the latent features of several masked **image blocks** from a single
visible context block using a Vision Transformer. Learns strong image features
without augmentations or pixel reconstruction.
→ **[docs/I-JEPA.md](docs/I-JEPA.md)**

### 🎬 V-JEPA — Video
Extends the idea to **spatio-temporal** tubes in video, learning motion and
temporal dynamics. **V-JEPA 2** scales to internet-scale video and adds a
predictor head, making it a capable action-recognition model and world model.
→ **[docs/V-JEPA.md](docs/V-JEPA.md)**

### 🖼️+📝 VL-JEPA — Vision + Language
Aligns **images/video and text** in a shared latent space, so matching pairs sit
close together. Downstream tasks (captioning, VQA) are solved by comparing
embeddings rather than generating tokens.
→ **[docs/VL-JEPA.md](docs/VL-JEPA.md)**

---

## Notebooks

| Notebook | Variant | Task |
|---|---|---|
| [`I-JEPA_test_1.ipynb`](I-JEPA_test_1.ipynb) | I-JEPA | Embedding · similarity · classification (train + infer) |
| [`V-JEPA_test_1.ipynb`](V-JEPA_test_1.ipynb) | V-JEPA 2 | Feature extraction · video action recognition |
| [`vl_jepa_train_test_1.ipynb`](vl_jepa_train_test_1.ipynb) | VL-JEPA (v1) | Training a compact vision-language JEPA |
| [`vl_jepa_infer_test_1.ipynb`](vl_jepa_infer_test_1.ipynb) | VL-JEPA (v1) | Image↔text similarity inference |
| [`vl_jepa2_sft_test_1.ipynb`](vl_jepa2_sft_test_1.ipynb) | VL-JEPA 2 | Supervised fine-tuning (SFT) |
| [`vl_jepa2_infer_test_1.ipynb`](vl_jepa2_infer_test_1.ipynb) | VL-JEPA 2 | Caption · discriminative VQA · selective decoding |

---

## Getting started

The notebooks were developed with (versions may vary per notebook — see each
notebook's intro cell):

```bash
# I-JEPA / V-JEPA 
transformers==5.13.0
torch==2.12.1+cu130   torchvision==0.27.1+cu130   torchcodec==0.14.0+cu130

# VL-JEPA 2 
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install pyyaml tqdm transformers
```

A CUDA GPU is strongly recommended (each notebook starts with `!nvidia-smi`).
The VL-JEPA notebooks depend on local `vl_jepa/` and `vl_jepa2/` packages
(see the reference repos linked in each doc page and notebook).

> **Note:** These notebooks are exploratory test/learning material. Model IDs,
> dataset paths, and helper packages reflect the original experiments — adapt
> paths to your own environment before running.

---

## References

- **JEPA / world models** — LeCun, *A Path Towards Autonomous Machine Intelligence* (2022)
- **I-JEPA** — Assran et al., [*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*](https://arxiv.org/abs/2301.08243) (CVPR 2023)
- **V-JEPA** — Bardes et al., [*Revisiting Feature Prediction for Learning Visual Representations from Video*](https://arxiv.org/abs/2404.08471) (2024)
- **V-JEPA 2** — Meta AI, [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2) (2025)
- **VL-JEPA** — [arXiv:2512.10942](https://arxiv.org/pdf/2512.10942)
