"""End-to-end test of moge_daemon.py.

Starts a daemon on a private port, then:
  1. info round trip
  2. bad requests come back as status 2 without killing the connection
  3. infer on docs/sample.jpg; compare against model.infer() run directly in
     this process with identical settings (refine_steps=0 so the comparison
     is deterministic -- the refiner is not, by ~7e-3 in depth)
  4. refine_steps=3 runs and changes depth but not normals
  5. linear vs srgb colorspace flag, nuke vs opencv axes, apply_mask=0
  6. shutdown makes the process exit

Run in the MoGe venv:  python test_moge_daemon.py
"""

import os
import subprocess
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "MoGe"))

import cv2
import numpy as np

from moge_client import Client, DaemonError

PORT = 47899
MODEL = os.environ.get("MOGE_TEST_MODEL") or os.path.join(ROOT, "models", "moge-3-vitg.pt")
IMAGE = os.path.join(ROOT, "docs", "sample.jpg")


def wait_for_daemon(port, timeout=180.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with Client(port=port) as c:
                return c.info()
        except OSError:
            time.sleep(1.0)
    raise RuntimeError("daemon did not come up")


def main():
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "moge_daemon.py"),
                             "--port", str(PORT), "--model", MODEL])
    try:
        info = wait_for_daemon(PORT)
        print("== info ==")
        print("  ", {k: info[k] for k in ("model", "device", "torch", "gpu", "loaded")})
        assert info["loaded"] and os.path.basename(info["model"]).startswith("moge-3-vitg")

        print("== bad requests ==")
        with Client(port=PORT) as c:
            for bad in ({"cmd": "nope"},
                        {"cmd": "infer", "width": 4, "height": 4},
                        {"cmd": "infer", "width": 2, "height": 2,
                         "input_colorspace": "rec709"}):
                try:
                    c.request(bad, b"\0" * (2 * 2 * 3 * 4) if bad.get("width") == 2 else b"")
                except DaemonError as exc:
                    print("   {0} -> status {1}: {2}".format(bad, exc.status,
                                                             exc.msg.strip()[:60]))
                    assert exc.status == 2
                else:
                    raise AssertionError("bad request accepted: {0}".format(bad))
            # connection still usable
            assert c.info()["loaded"]

        bgr = cv2.imread(IMAGE)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]

        print("== infer parity (refine 0, srgb input) ==")
        with Client(port=PORT) as c:
            planes, meta = c.infer(rgb, refine_steps=0, resolution_level=9,
                                   input_colorspace="srgb", normal_space="opencv",
                                   apply_mask=0)
        assert planes.shape == (5, h, w)
        print("   daemon {0:.2f}s  fov_x {1:.2f}".format(meta["elapsed"], meta["fov_x"]))

        import torch
        if MODEL.lower().endswith(".safetensors"):
            from moge_safetensors import load_moge
            model = load_moge(MODEL, "cuda")
        else:
            from moge.model.v3 import MoGeModel
            model = MoGeModel.from_pretrained(MODEL).cuda().eval()
        image = torch.tensor(rgb, device="cuda").permute(2, 0, 1)
        with torch.inference_mode():
            ref = model.infer(image, refine_steps=0, resolution_level=9, apply_mask=False)
        n_ref = ref["normal"].float().cpu().numpy()
        d_ref = ref["depth"].float().cpu().numpy()
        m_ref = ref["mask"].cpu().numpy()
        n_d = np.moveaxis(planes[0:3], 0, -1)
        d_d = planes[4]
        m_d = planes[3] > 0.5
        both = m_d & m_ref
        ang = np.degrees(np.arccos(np.clip((n_d * n_ref).sum(-1), -1, 1)))[both]
        rel = (np.abs(d_d - d_ref) / np.maximum(d_ref, 1e-6))[both]
        print("   mask agreement {0:.3f}%   normals median {1:.5f} deg   depth median "
              "{2:.5f}%".format(100 * (m_d == m_ref).mean(), np.median(ang),
                                100 * np.median(rel)))
        assert (m_d == m_ref).mean() > 0.999
        assert np.median(ang) < 0.01 and np.median(rel) < 1e-3
        del model
        torch.cuda.empty_cache()

        print("== refine 3 vs 0 ==")
        with Client(port=PORT) as c:
            p3, m3 = c.infer(rgb, refine_steps=3, input_colorspace="srgb")
            p0, m0 = c.infer(rgb, refine_steps=0, input_colorspace="srgb")
        print("   refine 3: {0:.2f}s   refine 0: {1:.2f}s".format(m3["elapsed"], m0["elapsed"]))
        valid = (p3[3] > 0.5) & (p0[3] > 0.5)
        n_change = np.abs(p3[0:3] - p0[0:3]).max()
        d_change = np.median(np.abs(p3[4] - p0[4])[valid])
        print("   normals max change {0:.2e}   depth median change {1:.4f}".format(
            n_change, d_change))
        assert n_change < 1e-5, "refiner must not touch normals"
        assert d_change > 1e-4, "refiner should change depth"

        print("== flags ==")
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        with Client(port=PORT) as c:
            p_lin, _ = c.infer(lin.astype(np.float32), refine_steps=0,
                               input_colorspace="linear")
            p_cv, _ = c.infer(rgb, refine_steps=0, input_colorspace="srgb",
                              normal_space="opencv")
            p_raw, _ = c.infer(rgb, refine_steps=0, input_colorspace="srgb",
                               apply_mask=0)
        v = (p0[3] > 0.5) & (p_lin[3] > 0.5)
        dot = np.clip((p_lin[0:3] * p0[0:3]).sum(0), -1, 1)[v]
        ang_lin = np.degrees(np.arccos(dot))
        print("   linear-flag vs srgb: normals median {0:.4f} deg  p99 {1:.3f} deg".format(
            np.median(ang_lin), np.percentile(ang_lin, 99)))
        # linear->sRGB->model vs sRGB->model: identical up to float rounding
        # in the transfer curve, so the geometry must agree closely
        assert np.median(ang_lin) < 0.05
        assert np.allclose(p_cv[0], p0[0]) and np.allclose(p_cv[1], -p0[1]) \
            and np.allclose(p_cv[2], -p0[2])
        sky = p0[3] < 0.5
        assert np.array_equal(p_raw[3], p0[3])
        if sky.any():
            assert np.abs(p0[0:3][:, sky]).max() == 0.0, "masked output must be zero in sky"
            assert np.linalg.norm(p_raw[0:3], axis=0)[sky].mean() > 0.5, "raw keeps normals"
            print("   opencv axes flip y/z; apply_mask=0 keeps raw normals; alpha unchanged")
        else:
            print("   opencv axes flip y/z; alpha unchanged (no invalid pixels in this "
                  "image, mask behaviour not exercised)")

        print("== shutdown ==")
        with Client(port=PORT) as c:
            c.shutdown()
        code = proc.wait(timeout=30)
        print("   daemon exit code", code)
        assert code == 0
        print("")
        print("ALL DAEMON CHECKS PASSED")
    finally:
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    main()
