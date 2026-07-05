# V-JEPA — Video Joint-Embedding Predictive Architecture

> Notebook: [`../V-JEPA_test_1.ipynb`](../V-JEPA_test_1.ipynb)
> Code: [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2) ·
> HF docs: [vjepa2](https://huggingface.co/docs/transformers/main/model_doc/vjepa2)

[← Back to README](../README.md)

---

## Concept

**V-JEPA** applies the JEPA recipe to **video**: it predicts the latent
representations of masked **spatio-temporal regions** (tubes spanning space and
time) from the visible parts of a clip. By predicting in latent space rather
than reconstructing pixels, it learns motion, object permanence, and temporal
dynamics — the ingredients of a visual **world model**.

**V-JEPA 2** (Meta AI, 2025) scales this to internet-scale video and introduces
a dedicated **predictor** head. Its outputs come in two flavors:

- **Encoder outputs** — per-patch spatio-temporal features of the input clip.
- **Predictor outputs** — the model's *predicted* latents for masked regions;
  this is the piece that makes V-JEPA 2 useful for planning / world-modeling.

The pretrained backbone transfers well to **action recognition** with just a
lightweight classification head fine-tuned on a labeled dataset.

---

## Model zoo (checkpoints used in the notebook)

| Model ID | Frames/clip | Head |
|---|---|---|
| `facebook/vjepa2-vitl-fpc64-256` | 64 | none (backbone only) |
| `facebook/vjepa2-vitl-fpc16-256-ssv2` | 16 | Something-Something-v2 action head |

Video decoding uses **torchcodec** — install ffmpeg first:
```bash
conda install "ffmpeg<8" -c conda-forge
pip install torch torchvision torchcodec --index-url https://download.pytorch.org/whl/cu130
```

---

## 1. Feature extraction (encoder + predictor)

Load the backbone with `AutoModel` and read both output streams:

```python
import os, numpy as np
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"      # avoid a decoder backend clash
from torchcodec.decoders import VideoDecoder
from transformers import AutoModel, AutoVideoProcessor

model_id  = "facebook/vjepa2-vitl-fpc64-256"
processor = AutoVideoProcessor.from_pretrained(model_id)
model     = AutoModel.from_pretrained(model_id, device_map="auto",
                                      attn_implementation="sdpa")

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"
vr        = VideoDecoder(video_url)
frame_idx = np.arange(0, 64)                       # first 64 frames (T x C x H x W)
video     = vr.get_frames_at(indices=frame_idx).data
video     = processor(video, return_tensors="pt").to(model.device)

outputs = model(**video)
encoder_outputs   = outputs.last_hidden_state                       # per-patch features
predictor_outputs = outputs.predictor_output.last_hidden_state      # predicted latents
```

`model.get_vision_features()` is a shortcut equivalent to reading
`outputs.last_hidden_state`.

---

## 2. Video inference — action recognition

Use a checkpoint with a trained head and read off the top-5 actions:

```python
import torch, numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

model_id  = "facebook/vjepa2-vitl-fpc16-256-ssv2"
model     = AutoModelForVideoClassification.from_pretrained(model_id, device_map="auto")
processor = AutoVideoProcessor.from_pretrained(model_id)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
vr        = VideoDecoder(video_url)
# Sample exactly frames_per_clip frames (subsampling every 8th frame here).
frame_idx = np.arange(0, model.config.frames_per_clip, 8)
video     = vr.get_frames_at(indices=frame_idx).data

inputs = processor(video, return_tensors="pt").to(model.device)
with torch.no_grad():
    logits = model(**inputs).logits

top5_idx   = logits.topk(5).indices[0]
top5_probs = torch.softmax(logits, dim=-1).topk(5).values[0]
for idx, prob in zip(top5_idx, top5_probs):
    print(f" - {model.config.id2label[idx.item()]}: {prob:.2f}")
```

---

## 3. Classification class & training loss

The concrete `VJEPA2ForVideoClassification` class exposes the same behavior and
lets you compute a **training loss** by passing `labels`:

```python
import torch, numpy as np
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

model_id        = "facebook/vjepa2-vitl-fpc16-256-ssv2"
device          = "cuda"
video_processor = AutoVideoProcessor.from_pretrained(model_id)
model           = VJEPA2ForVideoClassification.from_pretrained(model_id).to(device)

video  = np.ones((64, 256, 256, 3))                 # placeholder clip — replace with real frames
inputs = video_processor(video, return_tensors="pt").to(device)

# Inference
with torch.no_grad():
    logits = model(**inputs).logits
print(model.config.id2label[logits.argmax(-1).item()])

# Training: pass labels to get a differentiable loss
labels = torch.ones(1, dtype=torch.long, device=device)
loss   = model(**inputs, labels=labels).loss
```

---

## Key takeaways

- V-JEPA predicts **masked spatio-temporal latents** → learns motion & dynamics.
- V-JEPA 2 exposes both **encoder** and **predictor** outputs; the predictor is
  the world-model signal.
- `AutoModel` → raw features; `AutoModelForVideoClassification` /
  `VJEPA2ForVideoClassification` → action labels (and training loss).
- Always match the model's **`frames_per_clip`** when sampling frames.
