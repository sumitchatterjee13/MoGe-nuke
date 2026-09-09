"""Download the MoGe-3 ViT-G checkpoint into models/moge-3-vitg.pt.

    python tools/download_model.py [--repo Ruicheng/moge-3-vitg] [--dest models]

Optional afterwards (faster loads, same weights):
    python tools/convert_to_safetensors.py models/moge-3-vitg.pt models/moge-3-vitg.safetensors
"""

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="Ruicheng/moge-3-vitg")
    p.add_argument("--dest", default=os.path.join(ROOT, "models"))
    args = p.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    target = os.path.join(args.dest, args.repo.split("/")[-1] + ".pt")
    if os.path.isfile(target):
        print("already present:", target)
        return
    print("downloading {0}/model.pt (about 5 GB) ...".format(args.repo))
    cached = hf_hub_download(repo_id=args.repo, repo_type="model", filename="model.pt")
    shutil.copyfile(cached, target)
    print("saved:", target)


if __name__ == "__main__":
    main()
