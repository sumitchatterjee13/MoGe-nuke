"""Publish the safetensors checkpoint to a Hugging Face model repo.

    python tools/hf_upload.py models/moge-3-vitg.safetensors \
        [--repo Sumitc13/moge-3-vitg-safetensors] [--public]

Needs a write token (`hf auth login`). Creates the repo (private unless
--public), uploads tools/hf_model_card.md as README.md and the checkpoint as
model.safetensors. Re-running only uploads what changed.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--repo", default="Sumitc13/moge-3-vitg-safetensors")
    p.add_argument("--public", action="store_true")
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit("not found: " + args.checkpoint)
    api = HfApi()
    who = api.whoami()
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    print("user {0}, token role {1}".format(who.get("name"), role))
    if role not in ("write", "fineGrained"):
        sys.exit("a write token is required")

    url = api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    print("repo:", url)
    api.upload_file(path_or_fileobj=os.path.join(HERE, "hf_model_card.md"),
                    path_in_repo="README.md", repo_id=args.repo, repo_type="model",
                    commit_message="model card")
    print("uploaded README.md")
    size_gb = os.path.getsize(args.checkpoint) / 1e9
    print("uploading model.safetensors ({0:.2f} GB) ...".format(size_gb))
    api.upload_file(path_or_fileobj=args.checkpoint, path_in_repo="model.safetensors",
                    repo_id=args.repo, repo_type="model",
                    commit_message="MoGe-3 ViT-G weights converted to safetensors")
    info = api.model_info(args.repo, files_metadata=True)
    for s in info.siblings:
        print("  {0:<20} {1}".format(s.rfilename, s.size))
    print("done: https://huggingface.co/{0}  ({1})".format(
        args.repo, "public" if args.public else "private -- flip in Settings when ready"))


if __name__ == "__main__":
    main()
