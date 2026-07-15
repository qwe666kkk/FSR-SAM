# FSR-SAM Core Model

Minimal model implementation of **FSR-SAM: Frequency-guided Saliency
Refinement with SAM for Salient Object Detection**. This repository contains
the paper architecture and a SAM-B checkpoint adapter; training, evaluation,
datasets, checkpoints, visualizations, and experiment logs are not included.

## Architecture

```text
Image
  -> SAM ViT-B encoder + MBA
  -> F1, F2, F3, F4
  -> BandFiLM(F4) and OCM(F3, F4')
  -> MCFM + SFEM -> semantic guidance S_m
  -> SAM mask decoder -> coarse logits M_c and decoder feature F_d
  -> frequency refinement F_r
  -> P_loc from F3' and F4'
  -> P_loc-gated fusion of F_r, D_f, and image detail I_d
  -> final saliency logits
```

| Paper module | Implementation |
| --- | --- |
| MBA | `model/mb_adapt.py`, injected by `model/image_encoder.py` |
| OCM | `BandFiLM2d` and `DirectionConsistentGate` |
| SFEM | `MCFM` and `model/semantic_fusion.py` |
| SAM Decoder | `model/mask_decoder.py` |
| FGPM | locator in `model/fsrsam.py` and gated prediction in `model/block.py` |

## Installation

```bash
python -m pip install -r requirements.txt
```

Download the official SAM ViT-B checkpoint separately. Model weights are not
included in this repository.

## Minimal usage

```python
import torch
from model import FSRSAM, load_sam_checkpoint

model = FSRSAM(img_size=512)
load_sam_checkpoint(model, "sam_vit_b_01ec64.pth", img_size=512)
model.eval()

image = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    final_logits, coarse_logits, auxiliary = model(image)

final_prediction = torch.sigmoid(final_logits)
print(final_prediction.shape)        # (1, 1, 512, 512)
print(coarse_logits.shape)           # (1, 1, 512, 512)
print(auxiliary["loc_multi"].shape) # (1, 1, 32, 32)
```

The public return value is `(final_logits, coarse_logits, auxiliary)`. The
`auxiliary` dictionary exposes the two localization responses (`loc_f3`,
`loc_f4`), their fused foreground confidence map (`loc_multi`), and semantic
guidance (`sem_g_all`).

## Scope

This is a core architecture release, not a full training-reproduction package.
It intentionally excludes training and evaluation pipelines, datasets,
benchmark metric implementations, experiment scripts, generated figures, and
model checkpoints.

## Citation

Add the final paper citation here after publication.

## License

Add the selected project license before publishing. Files adapted from SAM
retain their original copyright headers.
