# FSR-SAM Core Model

Minimal model implementation of **FSR-SAM: Frequency-guided Saliency
Refinement with SAM for Salient Object Detection**. 
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



## License

This project is licensed under the [Apache License 2.0](LICENSE).

Parts of the implementation are adapted from Meta's
[Segment Anything](https://github.com/facebookresearch/segment-anything)
project. Those files retain their original copyright and license notices.
