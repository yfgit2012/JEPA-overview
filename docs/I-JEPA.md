# I-JEPA — Image Joint-Embedding Predictive Architecture

> Notebook: [`../I-JEPA_test_1.ipynb`](../I-JEPA_test_1.ipynb)
> Paper: [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) ·
> HF docs: [ijepa](https://huggingface.co/docs/transformers/en/model_doc/ijepa)

[← Back to README](../README.md)

---

## Concept

**I-JEPA** learns image representations by predicting, *in latent space*, the
features of several **target blocks** of an image from a single **context
block** — without any data augmentation, negative pairs, or pixel
reconstruction.

- **Context encoder** (a ViT) embeds the visible context patches.
- **Predictor** takes the context embeddings + positional info for the masked
  target blocks and predicts their latent representations.
- **Target encoder** is an EMA copy of the context encoder; it produces the
  target latents that the predictor is trained to match.

Masking strategy matters: I-JEPA uses **large, semantically meaningful target
blocks** and a spatially distributed context block. This pushes the model to
learn about objects and layout rather than local texture — which is why I-JEPA
features work well for downstream tasks with just a linear probe.

**Why it's efficient:** the loss is computed on a handful of target-block latents
(not every pixel), and there are no negatives to mine, so I-JEPA trains faster
than many contrastive or generative baselines at comparable quality.

---

## Model zoo

| Model ID | Backbone | Pretraining data |
|---|---|---|
| `facebook/ijepa_vith14_1k` | ViT-H/14 | ImageNet-1k |
| `facebook/ijepa_vitg16_22k` | ViT-G/16 | ImageNet-22k |

---

## 1. Image embeddings & similarity

Load a pretrained encoder and mean-pool patch tokens into one vector per image,
then compare two images with cosine similarity.

```python
import requests, torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor

model_id  = "facebook/ijepa_vitg16_22k"
processor = AutoProcessor.from_pretrained(model_id)
model     = AutoModel.from_pretrained(model_id, dtype="auto",
                                      attn_implementation="sdpa")

def infer(image):
    inputs  = processor(image, return_tensors="pt")
    outputs = model(**inputs)
    # Mean-pool the patch tokens into a single image embedding.
    return outputs.last_hidden_state.mean(dim=1)

img1 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
img2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000219578.jpg", stream=True).raw)

sim = cosine_similarity(infer(img1), infer(img2))   # in [-1, 1]
print(sim)
```

**Optional 4-bit quantization** to cut VRAM (pass into `from_pretrained`):

```python
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
# model = AutoModel.from_pretrained(model_id, quantization_config=quantization_config, dtype="auto")
```

---

## 2. Classification with a readout head

`IJepaForImageClassification` adds a classification head on top of the backbone:

```python
from transformers import AutoImageProcessor, IJepaForImageClassification
import torch
from datasets import load_dataset

image = load_dataset("huggingface/cats-image")["test"]["image"][0]
model_id = "facebook/ijepa_vith14_1k"

processor = AutoImageProcessor.from_pretrained(model_id)
model     = IJepaForImageClassification.from_pretrained(model_id)

inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print(model.config.id2label[logits.argmax(-1).item()])
```

---

## 3. Training a classifier on frozen features

The notebook trains a linear/fine-tuned classifier on I-JEPA features for a real
task: **brain-tumor MRI classification** (BRISC 2025, 4 classes).

**Prerequisites** — helper modules next to the notebook (from
[`sovit-123/ijepa`](https://github.com/sovit-123/ijepa)):
`classification_model.py`, `classification_utils.py`, `classification_datasets.py`.

**Data layout:**
```
classification_task/
├── train/<class>/*.jpg
└── test/<class>/*.jpg
```

**Backbone weights** (download once):
```bash
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar -O weights/IN1K-vit.h.14-300e.pth.tar
```

```python
from classification_model import LinearClassifier
from classification_datasets import get_datasets, get_data_loaders

dataset_train, dataset_valid, classes = get_datasets(
    train_dir="./classification_task/train",
    valid_dir="./classification_task/test",
)
train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size=1)

model = LinearClassifier(num_classes=len(classes), fine_tune="fine_tune",
                         weights="weights/IN1K-vit.h.14-300e.pth.tar").to(device)
# ... standard train/validate loop over `epochs`, then save_model + save_plots.
```

`fine_tune="fine_tune"` unfreezes the backbone. For a pure **linear probe**
(faster, less overfitting on small data), freeze the backbone instead.

---

## 4. Inference with the fine-tuned classifier

```python
checkpoint = torch.load("./classification_ft/best_classification_model.pth")
model = LinearClassifier(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
# preprocess each test image, forward pass, softmax -> argmax -> class name
```

> ⚠️ **Gotcha:** `CLASS_NAMES` in the notebook contains a `'no_tunor'` typo
> carried over from the source repo. The label list must match the **training**
> class order exactly, so it's left unchanged — just be aware of it when reading
> predictions.

---

## Key takeaways

- I-JEPA = **latent prediction of masked image blocks**, no augmentations / no
  negatives / no pixel reconstruction.
- Great **frozen features** → strong linear-probe performance downstream.
- `AutoModel` for embeddings, `IJepaForImageClassification` for labels.
