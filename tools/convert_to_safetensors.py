"""Convert a MoGe `model.pt` checkpoint to safetensors.

The .pt file is a dict with two entries: `model_config` (a nested dict describing
how to build MoGeModel) and `model` (the state dict). safetensors stores only
tensors, so `model_config` is JSON-encoded into the file's metadata header and
recovered at load time by `moge_safetensors.load_moge`.

Usage:
    python scripts/convert_to_safetensors.py model/model.pt model/moge-3-vitg.safetensors
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="input model.pt")
    parser.add_argument("dst", type=Path, help="output .safetensors")
    parser.add_argument("--source", default="Ruicheng/moge-3-vitg", help="HF repo id, recorded in metadata")
    parser.add_argument("--hf-revision", default="", help="HF commit sha, recorded in metadata")
    args = parser.parse_args()

    print(f"hashing {args.src} ...")
    src_sha = sha256(args.src)
    print(f"  sha256 {src_sha}")

    print(f"loading {args.src} ...")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=True)
    if set(ckpt) != {"model_config", "model"}:
        raise SystemExit(f"unexpected checkpoint layout: {sorted(ckpt)}")

    state_dict = {k: v.contiguous() for k, v in ckpt["model"].items()}
    n_params = sum(v.numel() for v in state_dict.values())
    print(f"  {len(state_dict)} tensors, {n_params / 1e9:.3f}B params")

    metadata = {
        "format": "pt",
        "model_config": json.dumps(ckpt["model_config"]),
        "moge_version": "v3",
        "source": args.source,
        "hf_revision": args.hf_revision,
        "source_file": args.src.name,
        "source_sha256": src_sha,
    }

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {args.dst} ...")
    save_file(state_dict, str(args.dst), metadata=metadata)
    print(f"  done, {args.dst.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
