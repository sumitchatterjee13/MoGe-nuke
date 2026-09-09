"""Load the model and run inference once at the settings a session will use,
so the Triton kernel cache is populated (and the install is proven) before an
artist waits on it. Copy the resulting cache directory to air-gapped machines
that have no C compiler.

    python tools/warmup.py [--model PATH] [--refine-steps 0 3] [--sizes 1920x1080 ...]

Cache location: $TRITON_CACHE_DIR, else ~/.triton/cache (Linux) or
%USERPROFILE%\\.triton\\cache (Windows).
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "third_party", "MoGe"))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--refine-steps", type=int, nargs="+", default=[0, 3])
    p.add_argument("--sizes", nargs="+", default=["1920x1080", "2048x1152", "3840x2160"])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import cv2
    import numpy as np
    import torch

    model_path = args.model
    if not model_path:
        for name in ("moge-3-vitg.safetensors", "moge-3-vitg.pt"):
            cand = os.path.join(ROOT, "models", name)
            if os.path.isfile(cand):
                model_path = cand
                break
    if not model_path:
        sys.exit("no model in models/; pass --model")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("torch {0} cuda {1} | {2}".format(torch.__version__, torch.version.cuda, gpu))
    print("triton cache:", os.environ.get("TRITON_CACHE_DIR")
          or os.path.join(os.path.expanduser("~"), ".triton", "cache"))
    t0 = time.time()
    if model_path.lower().endswith(".safetensors"):
        from moge_safetensors import load_moge
        model = load_moge(model_path, args.device)
    else:
        from moge.model.v3 import MoGeModel
        model = MoGeModel.from_pretrained(model_path).to(args.device).eval()
    print("model loaded in {0:.1f}s: {1}".format(time.time() - t0, model_path))

    bgr = cv2.imread(os.path.join(ROOT, "docs", "sample.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for size in args.sizes:
        w, h = (int(v) for v in size.lower().split("x"))
        img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(np.ascontiguousarray(img)).to(args.device).permute(2, 0, 1)
        for steps in args.refine_steps:
            times = []
            for _ in range(2):  # first call may compile kernels; second is the real timing
                t0 = time.time()
                with torch.inference_mode():
                    out = model.infer(image, refine_steps=steps)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
            print("  {0:>10}  refine {1}  first {2:.2f}s  warm {3:.2f}s".format(
                size, steps, times[0], times[1]))
            assert torch.isfinite(out["depth"][out["mask"]]).all()
    if torch.cuda.is_available():
        print("peak GPU memory: {0:.1f} GB".format(torch.cuda.max_memory_allocated() / 1e9))
    print("OK")


if __name__ == "__main__":
    main()
