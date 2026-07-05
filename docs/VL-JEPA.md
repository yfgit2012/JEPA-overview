# VL-JEPA — Vision-Language Joint-Embedding Predictive Architecture

> Notebooks:
> [`../vl_jepa_train_test_1.ipynb`](../vl_jepa_train_test_1.ipynb) ·
> [`../vl_jepa_infer_test_1.ipynb`](../vl_jepa_infer_test_1.ipynb) (v1) ·
> [`../vl_jepa2_sft_test_1.ipynb`](../vl_jepa2_sft_test_1.ipynb) ·
> [`../vl_jepa2_infer_test_1.ipynb`](../vl_jepa2_infer_test_1.ipynb) (v2)
> Paper: [arXiv:2512.10942](https://arxiv.org/pdf/2512.10942)

[← Back to README](../README.md)

---

## Concept

**VL-JEPA** brings the joint-embedding predictive idea to **vision + language**.
Images (or video frames) and text are encoded into a **shared latent space** so
that a matching image/caption pair lands close together and mismatched pairs land
far apart. Crucially, the learning signal is a **latent prediction** objective
(predict masked content in representation space) rather than generating pixels or
tokens.

Because everything lives in one embedding space, downstream tasks are solved by
**comparing embeddings**:

- **Retrieval / similarity** — nearest text for an image (or vice versa).
- **VQA** — pick the candidate answer whose embedding is closest to the query.
- **Captioning** — nearest-neighbor readout against a bank of candidate texts.

This repo contains two lineages:

| | Modality | Repo | Notebooks |
|---|---|---|---|
| **VL-JEPA (v1)** | image + text | [`mugiwaraluffy56/vl-jepa`](https://github.com/mugiwaraluffy56/vl-jepa) | train + infer |
| **VL-JEPA 2** | video frames + text | [`Soumya-Chakraborty/VL-JEPA`](https://github.com/Soumya-Chakraborty/VL-JEPA) | SFT + infer |

Both rely on **local packages** (`vl_jepa/`, `vl_jepa2/`) that must sit next to
the notebooks.

---

## VL-JEPA v1

### Architecture
A **vision encoder** and a **language encoder** each project into a common
`latent_dim`; a **predictor** ties them together. Training uses **block masking**
on image patches — the model predicts masked latent content conditioned on the
visible patches and the caption.

### Training

```python
import yaml, torch
from torch.utils.data import DataLoader
from vl_jepa.vl_jepa import build_vl_jepa
from vl_jepa.dataset import make_dummy_pairs, VLJEPADataset
from vl_jepa.masking import BlockMaskCollator
from vl_jepa.trainer import VLJEPATrainer
from vl_jepa.utils import set_seed

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("vl_jepa/config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["training"].update({"epochs": 3, "batch_size": 8, "num_workers": 0, "use_amp": False})

model    = build_vl_jepa(cfg)
pairs    = make_dummy_pairs(n=64, img_size=cfg["data"]["image_size"])   # synthetic demo data
dataset  = VLJEPADataset(pairs=pairs)
collator = BlockMaskCollator(img_size=cfg["data"]["image_size"],
                             patch_size=cfg["model"]["patch_size"])
loader   = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collator)

trainer = VLJEPATrainer(model=model, train_loader=loader, lr=1e-4, epochs=3,
                        warmup_epochs=1, use_amp=False,
                        checkpoint_dir="vljepa_model", save_every=3, device=device)
trainer.train()   # checkpoints → vljepa_model/checkpoint_epoch0003.pt
```

### Inference — image↔text similarity

```python
import torch, yaml
import torch.nn.functional as F
from vl_jepa.vl_jepa import build_vl_jepa
from vl_jepa.language_encoder import build_tokenizer

with open("vl_jepa/config.yaml") as f:
    cfg = yaml.safe_load(f)

model_id = "vljepa_model/checkpoint_epoch0003.pt"   # or None → random weights (demo)
model    = build_vl_jepa(cfg)
if model_id:
    model.load_state_dict(torch.load(model_id, map_location=device)["model_state"])
model.to(device).eval()

tokenizer = build_tokenizer(cfg["model"]["vocab_size"], cfg["model"]["max_seq_len"])
captions  = ["a dog running on the beach", "a cat sitting on a windowsill",
             "mountains covered in snow", "a cup of coffee on a wooden table"]
token_ids, attn_mask = tokenizer.batch_encode(captions)

with torch.no_grad():
    image_embs = F.normalize(model.encode_image(images), dim=-1)
    text_embs  = F.normalize(model.encode_text(token_ids, attn_mask), dim=-1)

sim = image_embs @ text_embs.T          # (N_images, N_captions)
# For a trained model, the diagonal (matching pairs) should dominate each row.
print("diagonal mean:", sim.diag().mean().item())   # ~1.0 trained, ~0.0 random
```

> The inference notebook uses `make_dummy_pairs` so it runs with **no dataset**.
> With random (untrained) weights the similarity matrix is essentially noise —
> train first to see the diagonal light up.

---

## VL-JEPA 2

A larger, **video-capable** vision-language JEPA. Frames + a text query are
encoded into a shared space; you fine-tune with SFT, then run inference in one of
several modes.

### SFT training

```python
import torch
from vl_jepa2.train.build import build_model, build_optimizer_and_scheduler, build_train_loader
from vl_jepa2.train.trainer import Trainer
from vl_jepa2.utils.config import dump_resolved_config, load_yaml, validate_train_config
from vl_jepa2.utils.seed import set_seed

cfg = load_yaml("./vl_jepa2/configs/sft_tiny.yaml")   # sft.yaml for a real run
validate_train_config(cfg)
set_seed(cfg["train"]["seed"], deterministic=True)
dump_resolved_config(cfg, cfg["train"]["output_dir"])
device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

model               = build_model(cfg)
loader              = build_train_loader(cfg)
optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

trainer = Trainer(
    model=model, optimizer=optimizer, scheduler=scheduler, train_loader=loader,
    output_dir=cfg["train"]["output_dir"],
    max_steps=cfg["train"]["max_steps"],
    grad_accum_steps=cfg["train"]["gradient_accumulation_steps"],
    clip_grad_norm=cfg["train"]["clip_grad_norm"],
    temperature=cfg["train"]["temperature"],
    precision=cfg["train"]["precision"],
)
trainer.fit(device)   # checkpoints → cfg["train"]["output_dir"]
```

### Inference modes

Load a checkpoint, then choose a mode:

```python
import json, torch
from vl_jepa2.data.dataset import VisionLanguageJsonlDataset, vl_collate
from vl_jepa2.eval.tasks import discriminative_match
from vl_jepa2.inference.decoder import NearestNeighborDecoder
from vl_jepa2.train.build import build_model
from vl_jepa2.utils.config import load_yaml, validate_train_config

cfg = load_yaml("./vl_jepa2/configs/inference_tiny.yaml")
validate_train_config(cfg, require_runtime=True, require_train=False)
device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")

model = build_model(cfg)
ckpt  = torch.load("./vl_jepa2/checkpoint/step_0000002.pt", map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)   # strict=False tolerates head/aux keys
model.to(device).eval()

ds = VisionLanguageJsonlDataset(manifest_path="samples.jsonl",
                                num_frames=cfg["data"]["num_frames"],
                                image_size=cfg["data"]["image_size"])
```

**Mode 1 — Caption (nearest-neighbor readout):**
```python
text_bank = [line.strip() for line in open("text_bank.txt") if line.strip()]
decoder   = NearestNeighborDecoder(model, text_bank); decoder.build(device)
for i in range(len(ds)):
    batch = vl_collate([ds[i]])
    # decode each sample against the text bank ...
```

**Mode 2 — Discriminative VQA (pick best candidate):**
```python
for i in range(len(ds)):
    batch = vl_collate([ds[i]])
    pred  = model.predict_embedding(batch["frames"].to(device), batch["query"])
    ans   = discriminative_match(model, pred, [batch["candidates"][0]])[0]
    print(json.dumps({"idx": i, "prediction": ans}))
```

**Mode 3 — Selective decoding** *(paper Sec. 4.6)* — segment the per-frame
embedding stream and decode only representative points. Left as an optional stub
in the notebook (`vl_jepa2.inference.selective.selective_decode_points`).

---

## Key takeaways

- VL-JEPA aligns **vision and language in one latent space** via latent
  prediction, not token generation.
- Tasks reduce to **embedding comparison**: similarity, VQA, captioning.
- **v1** = image+text (compact reference impl); **v2** = video+text, with an
  SFT stage and multiple inference modes.
- Both need their **local package** (`vl_jepa/`, `vl_jepa2/`) present; the
  notebooks ship synthetic/dummy data so they run end-to-end without a dataset.
