---
license: mit
base_model: Ruicheng/moge-3-vitg
library_name: moge
pipeline_tag: depth-estimation
tags:
  - depth
  - normals
  - monocular-geometry
  - safetensors
  - nuke
  - vfx
---

# MoGe-3 ViT-G, safetensors

The [MoGe-3](https://github.com/microsoft/MoGe) ViT-G checkpoint
(`Ruicheng/moge-3-vitg`, Microsoft, MIT) converted 1:1 to
[safetensors](https://github.com/huggingface/safetensors). Same weights, same
config, no quantisation; just a format that loads by memory-mapping instead of
`torch.load`, which is faster and needs no pickle.

Made for [MoGe-nuke](https://github.com/sumitchatterjee13/MoGe-nuke), a Nuke
OFX node that runs the full model (sparse refiner included) live.

## Files

| file | size | sha256 |
|---|---|---|
| `model.safetensors` | 5.00 GB | `685c5bc2bc1acfac86b928255c2c5397a7de4824870ade392e2ba1f74c2ce52b` |

Source: `Ruicheng/moge-3-vitg/model.pt`, sha256
`ce7c15417e9105c2ace7b4272e2cc69e36940921211eb7fa05d4d0bb03f0a00c`.
The safetensors metadata header carries `model_config` (the constructor
arguments for `moge.model.v3.MoGeModel`), `source`, `source_sha256`
and `moge_version` so the file is self-describing.

## Loading

```python
import json, torch
from safetensors import safe_open
from safetensors.torch import load_file
from moge.model.v3 import MoGeModel        # pip install git+https://github.com/microsoft/MoGe

path = "model.safetensors"
with safe_open(path, framework="pt") as f:
    config = json.loads(f.metadata()["model_config"])
model = MoGeModel(**config)
model.load_state_dict(load_file(path), strict=True)
model = model.cuda().eval()

out = model.infer(image_tensor)   # (3, H, W) in [0, 1], sRGB-encoded
# out["depth"], out["normal"], out["mask"], out["points"], out["intrinsics"]
```

Or with the helper shipped in MoGe-nuke:

```python
from moge_safetensors import load_moge     # MoGe-nuke/tools
model = load_moge("model.safetensors", "cuda")
```

## Conversion

```
python tools/convert_to_safetensors.py model.pt model.safetensors
```

from the MoGe-nuke repo; the script records the source hash and config in the
header, and every tensor is compared to the original after writing.

## Licence

MIT, as the original: Copyright (c) Microsoft Corporation. See the
[MoGe repository](https://github.com/microsoft/MoGe/blob/main/LICENSE).
