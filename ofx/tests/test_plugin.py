"""Exercise the built MoGe3.ofx through the mini OFX host, without Nuke.

Checks, on whichever platform this runs:
  1. the plugin loads, describes, and asks for the full frame as RoI
  2. it auto-starts the daemon (on a private port), renders depth, then
     renders normals from its cache
  3. both outputs match a direct daemon request pixel for pixel
  4. the daemon it started exits once the host process is gone

    python ofx/tests/test_plugin.py            # uses ofx/build[-linux]/...

Run in the repo venv (needs numpy + opencv for the reference request).
"""

import os
import socket
import subprocess
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "daemon"))

import cv2
import numpy as np

from moge_client import Client

WIN = sys.platform == "win32"
BUILD = os.path.join(ROOT, "ofx", "build" if WIN else "build-linux")
PLUGIN = os.path.join(BUILD, "MoGe3.ofx.bundle", "Contents", "Win64" if WIN else "Linux-x86-64", "MoGe3.ofx")
HOST = os.path.join(BUILD, "tests", "mini_host.exe" if WIN else "mini_host")
PYTHON = sys.executable
DAEMON = os.path.join(ROOT, "daemon", "moge_daemon.py")
def _find_model():
    if os.environ.get("MOGE_TEST_MODEL"):
        return os.environ["MOGE_TEST_MODEL"]
    for name in ("moge-3-vitg.safetensors", "moge-3-vitg.pt"):
        cand = os.path.join(ROOT, "models", name)
        if os.path.isfile(cand):
            return cand
    return os.path.join(ROOT, "models", "moge-3-vitg.safetensors")


MODEL = _find_model()
PORT = 47897
TMP = os.path.join(ROOT, "output", "plugin_test")


def port_open(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return True
    except OSError:
        return False


def main():
    for label, path in (("plugin", PLUGIN), ("host", HOST), ("model", MODEL)):
        assert os.path.exists(path), "{0} missing: {1}".format(label, path)
    assert not port_open(PORT), "something already listens on port {0}".format(PORT)
    os.makedirs(TMP, exist_ok=True)

    bgr = cv2.imread(os.path.join(ROOT, "docs", "sample.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    src = os.path.join(TMP, "in.rgb")
    np.ascontiguousarray(rgb, dtype="<f4").tofile(src)
    out1, out2 = os.path.join(TMP, "depth.rgba"), os.path.join(TMP, "normals.rgba")

    cmd = [HOST, PLUGIN, src, str(w), str(h), out1,
           "out2=" + out2,
           "output=0", "refineSteps=0", "inputColorspace=1",
           "pythonExe=" + PYTHON, "daemonScript=" + DAEMON, "model=" + MODEL,
           "port=" + str(PORT), "autoStart=1", "exitWithHost=0"]
    print("== mini host ==")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for line in (proc.stdout + proc.stderr).splitlines():
        print("   " + line)
    print("   exit {0} in {1:.1f}s".format(proc.returncode, time.time() - t0))
    assert proc.returncode == 0, "mini host failed"
    assert "RoI for a 10x10 request -> source 0,0 {0},{1}".format(w, h) in proc.stdout, \
        "plugin must request the full source frame"
    assert "render 2 (output flipped, should be a cache hit) -> status 0" in proc.stdout

    print("== compare with a direct daemon request ==")
    assert port_open(PORT), "daemon should still be up right after the host exits"
    with Client(port=PORT) as c:
        ref, meta = c.infer(rgb, refine_steps=0, input_colorspace="srgb")
    d = np.fromfile(out1, dtype="<f4").reshape(h, w, 4)
    n = np.fromfile(out2, dtype="<f4").reshape(h, w, 4)
    assert np.array_equal(d[..., 0], ref[4]) and np.array_equal(d[..., 1], ref[4]) and np.array_equal(d[..., 2], ref[4])
    assert np.array_equal(d[..., 3], ref[3]) and np.array_equal(n[..., 3], ref[3])
    assert np.array_equal(n[..., 0], ref[0]) and np.array_equal(n[..., 1], ref[1]) and np.array_equal(n[..., 2], ref[2])
    print("   depth RGB, normals RGB and mask alpha are bit-identical to the daemon reply")
    print("   fov_x {0:.2f} deg, {1}x{2}".format(meta["fov_x"], w, h))

    with Client(port=PORT) as c:
        c.shutdown()

    print("== daemon exits with its host (second run, exitWithHost=1) ==")
    port2 = PORT + 1
    assert not port_open(port2)
    cmd2 = [a for a in cmd if not a.startswith(("port=", "exitWithHost=", "out2="))]
    cmd2 += ["port=" + str(port2), "exitWithHost=1"]
    proc = subprocess.run(cmd2, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    t0 = time.time()
    while port_open(port2) and time.time() - t0 < 30:
        time.sleep(0.5)
    assert not port_open(port2), "daemon kept running after the host exited"
    print("   gone {0:.1f}s after the host exited".format(time.time() - t0))
    print("")
    print("ALL PLUGIN CHECKS PASSED ({0})".format(sys.platform))


if __name__ == "__main__":
    main()
