"""Download the MoGe-3 ViT-G checkpoint into models/.

    python tools/download_model.py [--format safetensors|pt] [--dest models]

safetensors (default): Sumitc13/moge-3-vitg-safetensors -- the same weights as
    Microsoft's release, converted for fast mmap loading (MIT).
pt: the original Ruicheng/moge-3-vitg model.pt from Microsoft.

Both are about 5 GB. Verify with tools/verify_model.py.
"""

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCES = {
    "safetensors": ("Sumitc13/moge-3-vitg-safetensors", "model.safetensors", "moge-3-vitg.safetensors"),
    "pt": ("Ruicheng/moge-3-vitg", "model.pt", "moge-3-vitg.pt"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", choices=sorted(SOURCES), default="safetensors")
    p.add_argument("--dest", default=os.path.join(ROOT, "models"))
    args = p.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    for name in ("moge-3-vitg.safetensors", "moge-3-vitg.pt"):
        if os.path.isfile(os.path.join(args.dest, name)):
            print("already present:", os.path.join(args.dest, name))
            return
    repo, filename, local = SOURCES[args.format]
    target = os.path.join(args.dest, local)
    print("downloading {0}/{1} (about 5 GB) ...".format(repo, filename))
    cached = hf_hub_download(repo_id=repo, repo_type="model", filename=filename)
    shutil.copyfile(cached, target)
    print("saved:", target)


if __name__ == "__main__":
    main()
