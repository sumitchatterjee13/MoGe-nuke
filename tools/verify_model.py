"""Print the SHA-256 of checkpoint files and name the known ones.

    python tools/verify_model.py models/moge-3-vitg.pt [more files...]
"""

import hashlib
import sys

KNOWN = {
    "ce7c15417e9105c2ace7b4272e2cc69e36940921211eb7fa05d4d0bb03f0a00c":
        "Ruicheng/moge-3-vitg  model.pt",
    "685c5bc2bc1acfac86b928255c2c5397a7de4824870ade392e2ba1f74c2ce52b":
        "Sumitc13/moge-3-vitg-safetensors  model.safetensors",
}


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        digest = sha256(path)
        print("{0}  {1}  {2}".format(digest, path,
                                      KNOWN.get(digest, "(unknown: not one of the published files)")))
