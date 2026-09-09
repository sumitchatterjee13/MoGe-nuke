"""Resident MoGe-3 inference server.

Runs in the MoGe venv (torch + Triton), keeps the model on the GPU, and answers
inference requests over a localhost TCP socket. Clients: the MoGe3 OFX plugin
(../ofx) and moge_client.py. One process serves any number of Nuke sessions;
requests are serialised.

    python moge_daemon.py [--model PATH_OR_HF_ID] [--port 47821] [--device cuda]
                          [--idle-timeout SECONDS] [--parent-pid PID]

--model accepts a local .pt / .safetensors checkpoint or a Hugging Face repo id
(default: models/moge-3-vitg.pt if present, else "Ruicheng/moge-3-vitg", which
is downloaded to the Hugging Face cache on first use).

Wire protocol (v1, little-endian):

  request  b"MOGE" u32 json_len  json  u32 data_len  data
  reply    b"MOGR" i32 status u32 width u32 height u32 channels
           u32 msg_len msg  u32 data_len  data

  json (infer): width, height, and any of model, refine_steps,
      resolution_level, num_tokens, fov_x, fp16, input_colorspace
      ("linear" | "srgb"), normal_space ("nuke" | "opencv"), apply_mask
  data (infer): float32 RGB interleaved, top row first, width*height*3 values
  reply data  : float32 planar, top row first: nx, ny, nz, mask, depth
  reply msg   : JSON {"fov_x", "fov_y", "elapsed", "refine_steps", ...}

  cmd "info"      -> msg JSON describing the loaded model, no data
  cmd "load"      -> (re)load the model named in "model"
  cmd "shutdown"  -> server exits after replying

status 0 = ok; anything else = error, msg holds the text.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for extra in (os.path.join(ROOT, "tools"), os.path.join(ROOT, "third_party", "MoGe")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import numpy as np

HF_MODEL = "Ruicheng/moge-3-vitg"
LOCAL_MODEL = os.path.join(ROOT, "models", "moge-3-vitg.pt")
DEFAULT_MODEL = LOCAL_MODEL if os.path.isfile(LOCAL_MODEL) else HF_MODEL
DEFAULT_PORT = 47821
PROTOCOL = 1
REQ_MAGIC = b"MOGE"
REP_MAGIC = b"MOGR"
MAX_JSON = 1 << 20
MAX_DATA = 1 << 31

OPENCV_TO_NUKE = np.array([1.0, -1.0, -1.0], np.float32)


def log(msg):
    print("[moge-daemon] " + msg, flush=True)


def watch_parent(pid):
    """Exit when process `pid` (the Nuke that launched us) goes away.

    Frees the GPU as soon as the host closes, crash included. Windows: block on
    the process handle. Elsewhere: poll with a null signal.
    """
    def _run():
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            SYNCHRONIZE = 0x00100000
            handle = k32.OpenProcess(SYNCHRONIZE, False, int(pid))
            if not handle:
                log("cannot watch parent pid {0} (error {1}); staying up".format(
                    pid, ctypes.get_last_error()))
                return
            k32.WaitForSingleObject(handle, 0xFFFFFFFF)
        else:
            while True:
                try:
                    os.kill(int(pid), 0)
                except OSError:
                    break
                time.sleep(2.0)
        log("parent pid {0} exited -- shutting down".format(pid))
        os._exit(0)

    threading.Thread(target=_run, name="parent-watch", daemon=True).start()
    log("will exit when pid {0} does".format(pid))


def looks_like_repo_id(s):
    """'owner/name' (Hugging Face) as opposed to a filesystem path."""
    if "\\" in s or s.count("/") != 1 or ":" in s:
        return False
    owner, name = s.split("/")
    return bool(owner) and bool(name) and not name.lower().endswith((".pt", ".safetensors"))


def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------

class ProtocolError(Exception):
    pass


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k
    return bytes(buf)


def read_request(sock):
    """Return (json_dict, data_bytes) or raise."""
    magic = recv_exact(sock, 4)
    if magic != REQ_MAGIC:
        raise ProtocolError("bad magic {0!r}".format(magic))
    (jlen,) = struct.unpack("<I", recv_exact(sock, 4))
    if jlen > MAX_JSON:
        raise ProtocolError("json header too large: {0}".format(jlen))
    header = json.loads(recv_exact(sock, jlen).decode("utf-8"))
    (dlen,) = struct.unpack("<I", recv_exact(sock, 4))
    if dlen > MAX_DATA:
        raise ProtocolError("payload too large: {0}".format(dlen))
    data = recv_exact(sock, dlen) if dlen else b""
    return header, data


def pack_reply(status, msg, width=0, height=0, channels=0, data=b""):
    if isinstance(msg, dict):
        msg = json.dumps(msg)
    msg_b = msg.encode("utf-8")
    head = REP_MAGIC + struct.pack("<iIIII", status, width, height, channels, len(msg_b))
    return head + msg_b + struct.pack("<I", len(data)) + data


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def load_from_hub(repo_id, device):
    """Fetch a checkpoint from a Hugging Face repo: model.safetensors if the
    repo has one (faster, mmap-loaded), else MoGe's model.pt."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from moge.model.v3 import MoGeModel
    try:
        path = hf_hub_download(repo_id=repo_id, repo_type="model", filename="model.safetensors")
    except EntryNotFoundError:
        path = None
    if path:
        from moge_safetensors import load_moge
        return load_moge(path, device)
    return MoGeModel.from_pretrained(repo_id).to(device).eval()


class Engine:
    """Owns the model; one inference at a time."""

    def __init__(self, device):
        self.device = device
        self.model = None
        self.model_path = None
        self.lock = threading.Lock()
        self.load_time = 0.0
        self.count = 0

    def ensure_model(self, path):
        """Load `path`: a .safetensors, a .pt, or a Hugging Face repo id."""
        is_file = os.path.isfile(path)
        if is_file:
            path = os.path.abspath(path)
        elif not looks_like_repo_id(path):
            raise FileNotFoundError("model not found: " + path)
        if self.model is not None and self.model_path == path:
            return
        import torch
        from moge.model.v3 import MoGeModel

        if self.model is not None:
            log("unloading " + self.model_path)
            self.model = None
            torch.cuda.empty_cache()
        log("loading " + path)
        t0 = time.time()
        if is_file and path.lower().endswith(".safetensors"):
            from moge_safetensors import load_moge
            self.model = load_moge(path, self.device)
        elif is_file:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            model = MoGeModel(**ckpt["model_config"])
            model.load_state_dict(ckpt["model"], strict=True)
            self.model = model.to(self.device).eval()
        else:
            log("not a local file; fetching from Hugging Face (cached after the first time)")
            self.model = load_from_hub(path, self.device)
        self.model_path = path
        self.load_time = time.time() - t0
        log("model ready in {0:.1f}s".format(self.load_time))

    def info(self):
        import torch
        return {
            "protocol": PROTOCOL,
            "model": self.model_path,
            "loaded": self.model is not None,
            "device": self.device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "requests": self.count,
            "pid": os.getpid(),
        }

    def infer(self, header, data):
        import torch

        width, height = int(header["width"]), int(header["height"])
        if width <= 0 or height <= 0:
            raise ProtocolError("bad size {0}x{1}".format(width, height))
        expected = width * height * 3 * 4
        if len(data) != expected:
            raise ProtocolError("payload is {0} bytes, expected {1} for {2}x{3} RGB "
                                "float32".format(len(data), expected, width, height))
        refine_steps = int(header.get("refine_steps", 3))
        resolution_level = int(header.get("resolution_level", 9))
        num_tokens = int(header.get("num_tokens", 0))
        fov_x = float(header.get("fov_x", 0.0))
        fp16 = bool(int(header.get("fp16", 0)))
        colorspace = str(header.get("input_colorspace", "linear"))
        normal_space = str(header.get("normal_space", "nuke"))
        apply_mask = bool(int(header.get("apply_mask", 1)))
        if colorspace not in ("linear", "srgb"):
            raise ProtocolError("input_colorspace must be linear or srgb")
        if normal_space not in ("nuke", "opencv"):
            raise ProtocolError("normal_space must be nuke or opencv")
        if not 0 <= refine_steps <= 16:
            raise ProtocolError("refine_steps out of range")

        with self.lock:
            self.ensure_model(header.get("model") or self.model_path or DEFAULT_MODEL)
            rgb = np.frombuffer(data, dtype="<f4").reshape(height, width, 3)
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
            if colorspace == "linear":
                rgb = linear_to_srgb(rgb)
            rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
            image = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)

            kw = {"refine_steps": refine_steps, "use_fp16": fp16,
                  "apply_mask": apply_mask}
            if num_tokens > 0:
                kw["num_tokens"] = num_tokens
            else:
                kw["resolution_level"] = resolution_level
            if fov_x > 0:
                kw["fov_x"] = fov_x

            t0 = time.time()
            with torch.inference_mode():
                out = self.model.infer(image, **kw)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            self.count += 1

            depth = out["depth"].float().cpu().numpy()
            normal = out["normal"].float().cpu().numpy()
            mask = out["mask"].cpu().numpy().astype(np.float32)
            intr = out["intrinsics"].float().cpu().numpy()

        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        normal = np.where(np.isfinite(normal), normal, 0.0).astype(np.float32)
        if normal_space == "nuke":
            normal = normal * OPENCV_TO_NUKE
        if apply_mask:
            normal = normal * mask[..., None]
            depth = depth * mask

        planes = np.stack([normal[..., 0], normal[..., 1], normal[..., 2],
                           mask, depth], axis=0).astype("<f4")
        meta = {
            "fov_x": float(np.degrees(2 * np.arctan(0.5 / intr[0, 0]))),
            "fov_y": float(np.degrees(2 * np.arctan(0.5 / intr[1, 1]))),
            "elapsed": elapsed,
            "refine_steps": refine_steps,
            "resolution": [width, height],
            "model": self.model_path,
        }
        return meta, np.ascontiguousarray(planes).tobytes()


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

class Server:
    def __init__(self, engine, host, port, idle_timeout):
        self.engine = engine
        self.host = host
        self.port = port
        self.idle_timeout = idle_timeout
        self.last_activity = time.time()
        self.stop = threading.Event()

    def handle(self, conn, addr):
        conn.settimeout(600.0)
        try:
            while not self.stop.is_set():
                try:
                    header, data = read_request(conn)
                except (ConnectionError, socket.timeout):
                    return
                self.last_activity = time.time()
                cmd = header.get("cmd", "infer")
                try:
                    if cmd == "infer":
                        meta, out = self.engine.infer(header, data)
                        w, h = int(header["width"]), int(header["height"])
                        conn.sendall(pack_reply(0, meta, w, h, 5, out))
                        log("infer {0}x{1} refine={2} {3:.2f}s fov_x {4:.1f}".format(
                            w, h, meta["refine_steps"], meta["elapsed"], meta["fov_x"]))
                    elif cmd == "info":
                        conn.sendall(pack_reply(0, self.engine.info()))
                    elif cmd == "load":
                        with self.engine.lock:
                            self.engine.ensure_model(header.get("model") or DEFAULT_MODEL)
                        conn.sendall(pack_reply(0, self.engine.info()))
                    elif cmd == "shutdown":
                        conn.sendall(pack_reply(0, "bye"))
                        self.stop.set()
                        return
                    else:
                        conn.sendall(pack_reply(2, "unknown cmd " + repr(cmd)))
                except ProtocolError as exc:
                    conn.sendall(pack_reply(2, "bad request: {0}".format(exc)))
                except Exception:
                    tb = traceback.format_exc()
                    log("request failed:\n" + tb)
                    conn.sendall(pack_reply(1, tb))
        except Exception:
            log("connection {0} dropped:\n{1}".format(addr, traceback.format_exc()))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        srv.settimeout(1.0)
        log("listening on {0}:{1} (pid {2})".format(self.host, self.port, os.getpid()))
        while not self.stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                if self.idle_timeout and time.time() - self.last_activity > self.idle_timeout:
                    log("idle for {0}s, exiting".format(self.idle_timeout))
                    break
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self.handle, args=(conn, addr), daemon=True).start()
        srv.close()
        log("stopped")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--idle-timeout", type=float, default=0.0,
                   help="exit after this many idle seconds (0 = never)")
    p.add_argument("--lazy", action="store_true",
                   help="do not load the model until the first request")
    p.add_argument("--parent-pid", type=int, default=0,
                   help="exit when this process (the launching Nuke) exits")
    args = p.parse_args(argv)

    if args.parent_pid > 0:
        watch_parent(args.parent_pid)

    if os.path.isfile(args.model):
        # a local checkpoint never needs the network; keep huggingface_hub
        # from probing it (air-gapped machines have no route out)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA unavailable, falling back to CPU")
        device = "cpu"
    log("torch {0} | device {1} | python {2}".format(
        torch.__version__, device, sys.version.split()[0]))

    engine = Engine(device)
    if not args.lazy:
        engine.ensure_model(args.model)
    else:
        engine.model_path = args.model
    Server(engine, args.host, args.port, args.idle_timeout).serve()


if __name__ == "__main__":
    main()
