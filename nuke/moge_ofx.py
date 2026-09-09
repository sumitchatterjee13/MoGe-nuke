"""Nuke-side helpers for the MoGe3 OFX node and its daemon.

The OFX plugin (../ofx) is the node; this module only creates it from the menu
and talks to the daemon for start / status / stop. Pure stdlib so it runs in
Nuke's Python without numpy.
"""

import json
import os
import socket
import struct
import subprocess
import sys

import nuke

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFX_CLASS = "OFXcom.sumit.moge3_v1"
DEFAULT_PORT = 47821
if sys.platform == "win32":
    DEFAULT_PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe").replace("\\", "/")
else:
    DEFAULT_PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
DAEMON = os.path.join(ROOT, "daemon", "moge_daemon.py").replace("\\", "/")
HF_MODEL = "Ruicheng/moge-3-vitg"


def default_model():
    for name in ("moge-3-vitg.safetensors", "moge-3-vitg.pt"):
        path = os.path.join(ROOT, "models", name)
        if os.path.isfile(path):
            return path.replace("\\", "/")
    return HF_MODEL


def create():
    """Create the OFX node, wired to the selected node if there is one."""
    try:
        node = nuke.createNode(OFX_CLASS)
    except Exception as exc:
        nuke.message("MoGe3 OFX plugin not loaded ({0}).\n\nRun the installer from the "
                     "MoGe-nuke repo and restart Nuke. OFX_PLUGIN_PATH must include "
                     "the directory the bundle was installed to.".format(exc))
        return None
    print("[MoGe3] OFX node created -- connect it and view. First frame starts the "
          "daemon and loads the model (15-60 s); later frames ~0.5-2 s.")
    return node


def _request(header, port=DEFAULT_PORT, timeout=5.0):
    body = json.dumps(header).encode("utf-8")
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(b"MOGE" + struct.pack("<I", len(body)) + body + struct.pack("<I", 0))
        head = b""
        while len(head) < 24:
            chunk = s.recv(24 - len(head))
            if not chunk:
                raise ConnectionError("daemon closed the connection")
            head += chunk
        if head[:4] != b"MOGR":
            raise ConnectionError("bad reply")
        status, _, _, _, mlen = struct.unpack("<iIIII", head[4:])
        msg = b""
        while len(msg) < mlen:
            chunk = s.recv(mlen - len(msg))
            if not chunk:
                break
            msg += chunk
        return status, msg.decode("utf-8", "replace")


def status(port=DEFAULT_PORT, quiet=False):
    try:
        code, msg = _request({"cmd": "info"}, port)
    except OSError:
        if not quiet:
            nuke.message("No MoGe daemon on port {0}.".format(port))
        return None
    info = json.loads(msg) if code == 0 else {"error": msg}
    if not quiet:
        lines = ["MoGe daemon on port {0}".format(port)]
        for k in ("model", "device", "gpu", "torch", "requests", "pid"):
            if k in info:
                lines.append("{0}: {1}".format(k, info[k]))
        nuke.message("\n".join(lines))
    return info


def start(port=DEFAULT_PORT, model=None, python_exe=DEFAULT_PYTHON, exit_with_nuke=True):
    """Launch the daemon in its own console. No-op if one already answers.

    By default it is tied to this Nuke process and exits (freeing the GPU)
    when Nuke does.
    """
    if status(port, quiet=True):
        nuke.message("A MoGe daemon is already running on port {0}.".format(port))
        return
    model = model or default_model()
    for label, path in (("python", python_exe), ("daemon", DAEMON)):
        if not os.path.isfile(path):
            nuke.message("{0} not found:\n{1}\n\nRun the installer first.".format(label, path))
            return
    env = dict(os.environ)
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "PYTHONSTARTUP"):
        env.pop(var, None)  # Nuke's bootstrap breaks the venv interpreter
    cmd = [python_exe, DAEMON, "--model", model, "--port", str(port)]
    if exit_with_nuke:
        cmd += ["--parent-pid", str(os.getpid())]
    if sys.platform == "win32":
        subprocess.Popen(cmd, env=env, cwd=os.path.dirname(DAEMON),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
        where = "own console window"
    else:
        log_dir = os.path.join(os.environ.get("XDG_CACHE_HOME")
                               or os.path.expanduser("~/.cache"), "moge-nuke")
        os.makedirs(log_dir, exist_ok=True)
        where = os.path.join(log_dir, "daemon-{0}.log".format(os.getuid()))
        with open(where, "ab") as log:
            subprocess.Popen(cmd, env=env, cwd=os.path.dirname(DAEMON), stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print("[MoGe3] daemon starting: " + " ".join(cmd))
    nuke.message("MoGe daemon starting on port {0} ({1}).\n"
                 "The model takes 15-60 s to load.".format(port, where))


def stop(port=DEFAULT_PORT):
    try:
        _request({"cmd": "shutdown"}, port)
        print("[MoGe3] daemon on port {0} told to exit".format(port))
    except OSError:
        nuke.message("No MoGe daemon on port {0}.".format(port))
