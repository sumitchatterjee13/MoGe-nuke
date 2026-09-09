# Air-gapped / locked-down deployment

Everything MoGe-nuke does at render time stays on the machine: the plugin
talks to the daemon on `127.0.0.1` only, the daemon binds loopback only, and
a local checkpoint is loaded with no network access (the daemon sets
`HF_HUB_OFFLINE=1` for local files). The **installer** is the only part that
wants the internet, and this page replaces it with a copy-over-the-wall
procedure.

Tested on: Ubuntu 24.04 (native), Windows 11. Rocky/RHEL 8 and 9 are the
expected targets for studios; notes below.

## What crosses the air gap

| Item | Size | Produced by |
|---|---|---|
| this repository (git archive or a clone) | 3 MB | `git archive --format=tar.gz -o moge-nuke.tar.gz main` |
| wheelhouse: every Python wheel, pinned | 3-4 GB | `tools/make_wheelhouse.sh` / `.ps1` on a connected machine |
| model checkpoint | 5 GB | `tools/download_model.py` on a connected machine |
| (optional) Triton kernel cache | ~10 MB | `tools/warmup.py` on a connected machine with the same GPU class |

Verify the checkpoint after copying:

```
python tools/verify_model.py models/moge-3-vitg.safetensors
```

Known SHA-256:

| file | sha256 |
|---|---|
| `Ruicheng/moge-3-vitg` `model.pt` | `ce7c15417e9105c2ace7b4272e2cc69e36940921211eb7fa05d4d0bb03f0a00c` |
| `Sumitc13/moge-3-vitg-safetensors` `model.safetensors` | `685c5bc2bc1acfac86b928255c2c5397a7de4824870ade392e2ba1f74c2ce52b` |

## 1. On a connected machine

Same OS family and Python minor version as the target (wheels are
platform-specific; torch's are ~2.5 GB each).

```bash
git clone https://github.com/sumitchatterjee13/MoGe-nuke.git && cd MoGe-nuke
tools/make_wheelhouse.sh ../moge-wheelhouse --cuda cu128 --python 3.12   # Linux
# powershell -File tools\make_wheelhouse.ps1 ..\moge-wheelhouse -Cuda cu128   # Windows
.venv/bin/python tools/download_model.py            # after ./install.sh, or any env with huggingface_hub
```

`--cuda` must match the driver on the *target*: cu126 for driver 560+,
cu128 for 570+ (required for RTX 50-series), cu130 for 580+.

The wheelhouse contains `requirements.txt` (exact pins) and one wheel per
package, including the three MoGe dependencies that normally install from
GitHub (FlexGEMM, utils3d-moge, pipeline), built into wheels so no git is
needed on the target.

Optional: pre-compile Triton kernels for the target GPU class so no C
compiler is needed there (see step 3):

```bash
./install.sh --skip-model   # or any working env
.venv/bin/python tools/warmup.py --sizes 1920x1080 3840x2160 --refine-steps 0 3
tar czf triton-cache.tgz -C ~/.triton cache
```

## 2. On the target

Prerequisites on the box:

* NVIDIA driver (see the cu-version table above) and a GPU with 12 GB+ VRAM
* `uv` (a single static binary: copy it from
  https://github.com/astral-sh/uv/releases or `pip install uv` from the wheelhouse
  is *not* enough, uv itself is not a wheel dependency here)
* a Python 3.10-3.12 interpreter: `dnf install python3.12` on Rocky 9 / 8
  (uv cannot download one offline), or pass the path with `--python`
* to build the plugin: `gcc-c++`, `cmake` (a 5-second build, glibc-matched to
  the box; the Windows prebuilt has no Linux counterpart because a binary
  built on Ubuntu 24.04 needs glibc 2.38 and would not load on Rocky)
* for Triton at runtime: a C compiler (`gcc`) **or** the pre-warmed cache from
  step 1 unpacked into `~/.triton/cache` (or `$TRITON_CACHE_DIR`) of the
  user that runs Nuke

Then:

```bash
tar xzf moge-nuke.tar.gz && cd MoGe-nuke
cp /media/transfer/moge-3-vitg.safetensors models/
./install.sh --offline /media/transfer/moge-wheelhouse --python /usr/bin/python3.12
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -Offline D:\transfer\moge-wheelhouse -Python C:\Python312\python.exe
```

The installer never contacts the network in this mode (`uv --offline
--no-index`), and it refuses to continue if the checkpoint is missing rather
than trying to download it.

Prove the install without Nuke:

```bash
.venv/bin/python tools/warmup.py            # loads the model, runs both refine settings
.venv/bin/python daemon/test_moge_daemon.py # full protocol test against a private port
```

## 3. Triton and the C compiler

MoGe-3's refiner (FlexGEMM) is written in Triton. Triton compiles its kernels
on first use for the GPU it finds, using its bundled `ptxas` plus the system
C compiler for a small launcher stub, and caches the result under
`~/.triton/cache`. Two ways to satisfy that on a locked-down box:

* install `gcc` (Rocky: `dnf install gcc`); the first render at each new
  resolution/refine setting takes a few seconds longer, once per user;
* or ship the cache from a machine with the same GPU architecture and Triton
  version (`warmup.py` above), and set `TRITON_CACHE_DIR` to a shared,
  read-only location for all users.

`refine steps = 0` never touches Triton at all.

## 4. Render farm

* Each blade runs its own daemon. Leave the node's **daemon exits with Nuke**
  on for interactive machines; for farm blades turn it off (or run the daemon
  as a service) so consecutive tasks reuse the loaded model instead of paying
  the 15-60 s load each time. A systemd user unit is in
  `docs/moge3-daemon.service`.
* The daemon serialises requests, so N concurrent Nuke processes on one blade
  queue on one GPU; that is usually what you want.
* Port 47821 on loopback; change it on the node's Setup tab and with
  `--port` if it collides with something. Nothing listens on external
  interfaces.
* Nothing is written outside the repo directory except `~/.nuke/init.py`
  (one line), the OFX bundle in the plugin directory, the daemon log in
  `$XDG_CACHE_HOME/moge-nuke/`, and Triton's cache.

## 5. Security review notes

* Plugin: raw OFX C++; only libc (Linux) or ws2_32/kernel32 (Windows). No
  telemetry, no downloads, no writes except the Nuke output image.
* Daemon: Python; imports torch, numpy, huggingface_hub (used only when the
  model knob holds a repo id instead of a file), the vendored MoGe package.
  Listens on `127.0.0.1:<port>`; no authentication (loopback only, same as
  Nuke's own localhost services). Do not bind it to an external interface.
* All third-party code is vendored or pinned; `third_party/MoGe` is an
  unmodified copy of Microsoft's MIT release, `ofx/include/openfx` is
  OpenFX 1.5.1 (BSD-3). See `THIRD_PARTY_NOTICES.md`.
