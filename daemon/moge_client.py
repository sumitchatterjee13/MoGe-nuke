"""Python client for moge_daemon.py -- the reference implementation of the
wire protocol and a handy CLI.

    python moge_client.py info
    python moge_client.py infer image.png --out result.exr [--refine-steps 3]
    python moge_client.py load --model other.safetensors
    python moge_client.py shutdown
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time

import numpy as np

REQ_MAGIC = b"MOGE"
REP_MAGIC = b"MOGR"
DEFAULT_PORT = 47821


class DaemonError(RuntimeError):
    def __init__(self, status, msg):
        super().__init__("daemon error {0}: {1}".format(status, msg))
        self.status = status
        self.msg = msg


def _recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("daemon closed the connection")
        got += k
    return bytes(buf)


class Client:
    def __init__(self, host="127.0.0.1", port=DEFAULT_PORT, timeout=600.0):
        self.sock = socket.create_connection((host, port), timeout=10.0)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def request(self, header, data=b""):
        j = json.dumps(header).encode("utf-8")
        self.sock.sendall(REQ_MAGIC + struct.pack("<I", len(j)) + j
                          + struct.pack("<I", len(data)))
        if data:
            self.sock.sendall(data)
        head = _recv_exact(self.sock, 4 + 4 * 5)
        if head[:4] != REP_MAGIC:
            raise ConnectionError("bad reply magic {0!r}".format(head[:4]))
        status, w, h, c, mlen = struct.unpack("<iIIII", head[4:])
        msg = _recv_exact(self.sock, mlen).decode("utf-8") if mlen else ""
        (dlen,) = struct.unpack("<I", _recv_exact(self.sock, 4))
        payload = _recv_exact(self.sock, dlen) if dlen else b""
        if status != 0:
            raise DaemonError(status, msg)
        return status, msg, (w, h, c), payload

    def info(self):
        _, msg, _, _ = self.request({"cmd": "info"})
        return json.loads(msg)

    def shutdown(self):
        self.request({"cmd": "shutdown"})

    def infer(self, rgb, **params):
        """rgb: (H,W,3) float32, top row first. Returns (planes (5,H,W), meta)."""
        rgb = np.ascontiguousarray(rgb, dtype="<f4")
        h, w = rgb.shape[:2]
        header = {"cmd": "infer", "width": w, "height": h}
        header.update(params)
        _, msg, (rw, rh, rc), payload = self.request(header, rgb.tobytes())
        planes = np.frombuffer(payload, dtype="<f4").reshape(rc, rh, rw)
        return planes, json.loads(msg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("info", "infer", "shutdown", "load"))
    p.add_argument("image", nargs="?")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None, help="EXR: rgb=normal, a=mask; depth in .depth.exr")
    p.add_argument("--refine-steps", type=int, default=3)
    p.add_argument("--resolution-level", type=int, default=9)
    p.add_argument("--fov-x", type=float, default=0.0)
    p.add_argument("--input-colorspace", default="srgb", choices=("linear", "srgb"))
    args = p.parse_args()

    with Client(port=args.port) as c:
        if args.cmd == "info":
            print(json.dumps(c.info(), indent=2))
        elif args.cmd == "shutdown":
            c.shutdown()
            print("daemon told to exit")
        elif args.cmd == "load":
            _, msg, _, _ = c.request({"cmd": "load", "model": args.model})
            print(msg)
        else:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            import cv2
            img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
            if img is None:
                sys.exit("cannot read " + args.image)
            img = img[..., :3].astype(np.float32)
            if img.max() > 2.0:
                img /= 255.0
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            params = {"refine_steps": args.refine_steps,
                      "resolution_level": args.resolution_level,
                      "input_colorspace": args.input_colorspace}
            if args.fov_x > 0:
                params["fov_x"] = args.fov_x
            if args.model:
                params["model"] = args.model
            t0 = time.time()
            planes, meta = c.infer(rgb, **params)
            print("round trip {0:.2f}s, inference {1:.2f}s, fov_x {2:.2f}".format(
                time.time() - t0, meta["elapsed"], meta["fov_x"]))
            if args.out:
                exr = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
                n = np.dstack([planes[2], planes[1], planes[0], planes[3]])
                cv2.imwrite(args.out, n, exr)
                d = np.dstack([planes[4], planes[4], planes[4], planes[3]])
                root, ext = os.path.splitext(args.out)
                cv2.imwrite(root + ".depth" + ext, d, exr)
                print("wrote", args.out)


if __name__ == "__main__":
    main()
