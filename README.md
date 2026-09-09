# MoGe-nuke

Monocular depth and normals in Nuke from [MoGe-3](https://github.com/microsoft/MoGe)
(Microsoft, ViT-G, 1.25 B parameters), running the **complete** model with its
sparse volumetric refiner, as a normal pull-based node: connect it, view it,
done. No baking, no render button. Windows and Linux.

![example](docs/example.png)

**How it works.** MoGe-3's refiner is built on Triton kernels that are
JIT-compiled from Python, so it cannot live inside Nuke's own libtorch or a
`.cat` file. Instead the model runs once, resident on the GPU, in a small
Python daemon; the Nuke node is an OFX plugin that sends each frame to it over
localhost and gets depth, normals and a validity mask back. The daemon is
started automatically on first use and shuts down when Nuke does.

## Model weights

**[huggingface.co/Sumitc13/moge-3-vitg-safetensors](https://huggingface.co/Sumitc13/moge-3-vitg-safetensors)**

Microsoft's MoGe-3 ViT-G checkpoint converted 1:1 to safetensors (same
weights, MIT), self-describing (the model config travels in the file header)
and memory-mapped on load. The installer downloads it into `models/`; the
original `model.pt` from [Ruicheng/moge-3-vitg](https://huggingface.co/Ruicheng/moge-3-vitg)
works too. Checksums for both are in [docs/OFFLINE.md](docs/OFFLINE.md).

## Requirements

* NVIDIA GPU with 12 GB+ VRAM (peak measured: 7.9 GB at 1080p, refine 3).
  Tested on an RTX 5090 with Nuke 17.1; any OFX 1.4 host should work
  (Nuke 13+).
* Windows 10/11, or Linux x86-64 (tested Ubuntu 24.04; Rocky 8/9 notes in
  [docs/OFFLINE.md](docs/OFFLINE.md)).
* [uv](https://docs.astral.sh/uv/) and git on PATH.
* NVIDIA driver matching the torch CUDA build (default `cu128` needs driver
  570+; RTX 50-series needs cu128 or newer).
* Linux: `gcc-c++` and `cmake` to build the plugin (a 5-second build).
  Windows: optional Visual Studio 2022 C++ tools; a prebuilt `MoGe3.ofx` is
  included.

## Install

```bash
git clone https://github.com/sumitchatterjee13/MoGe-nuke.git
cd MoGe-nuke
./install.sh                                              # Linux
powershell -ExecutionPolicy Bypass -File install.ps1      # Windows
```

The installer

1. creates `.venv` (Python 3.12) with torch + MoGe + Triton
   (`--cuda cu126|cu128|cu130` / `-Cuda` picks the torch wheel index),
2. downloads the checkpoint (5 GB) to `models/`: the safetensors conversion
   from [Sumitc13/moge-3-vitg-safetensors](https://huggingface.co/Sumitc13/moge-3-vitg-safetensors)
   by default, or Microsoft's original `model.pt` with `--format pt` on
   `tools/download_model.py`,
3. builds the OFX plugin (or uses `ofx/prebuilt` on Windows) and copies it to
   the first writable of `$OFX_PLUGIN_PATH`, the system OFX directory
   (`/usr/OFX/Plugins`, `C:\Program Files\Common Files\OFX\Plugins`), or
   `~/.nuke/OFXPlugins`,
4. adds `nuke/` to `~/.nuke/init.py` so the menu appears.

Restart Nuke. **Nodes > ML > MoGe3 > Depth + Normals**. Connect an image and
view the node. The first frame starts the daemon and loads the model, 15-60 s;
after that a 1080p frame takes 0.5-1 s.

Air-gapped machines, render farms and security notes: [docs/OFFLINE.md](docs/OFFLINE.md).

## The node

| MoGe tab | |
| --- | --- |
| model | local `.safetensors` / `.pt`, or a Hugging Face repo id |
| output | **depth**: RGB = metric depth, A = mask. **normals**: RGB = normal, A = mask |
| refine steps | 0 (refiner off) to 5. 3 is MoGe's default. Depth only; normals are identical at every setting |
| resolution level | 0-9, detail vs speed. Ignored when num tokens > 0 |
| num tokens | ViT tokens, 1200-3600 trained range. 0 = from resolution level |
| lock fov x | degrees; 0 = estimated per frame. Only rescales depth |
| fp16 | mixed precision |
| input colorspace | **scene-linear** (default; Nuke's working space, encoded to sRGB for the model) or **already sRGB** |
| normal space | **nuke**: x right, y up, z toward camera. **opencv**: raw MoGe axes |
| apply mask | zero depth/normals where the model marks pixels invalid (sky). The mask is always in alpha |

| Setup tab | |
| --- | --- |
| python / daemon script | the venv interpreter and `daemon/moge_daemon.py`; filled in by the installer |
| port | 47821 |
| auto-start daemon | launch the daemon when nothing answers on the port |
| daemon exits with Nuke | default on: frees the GPU when Nuke closes or crashes. Off keeps the model resident across sessions |

Depth is metric (scene units as the model estimates them). Normals in `nuke`
space drop straight into relighting setups (`N . L` with L in camera space).

Both passes come from one inference, so switching `output` is free. The
reply is cached per node on the source pixels and the inference knobs.

## Daemon

`daemon/moge_daemon.py` is a plain TCP server you can also drive yourself:

```text
.venv/bin/python daemon/moge_daemon.py              # foreground, default port
.venv/bin/python daemon/moge_client.py info
.venv/bin/python daemon/moge_client.py infer image.png --out result.exr
.venv/bin/python daemon/moge_client.py shutdown
```

Nuke menu: **MoGe3 > Daemon > Start / Status / Stop**. On Windows the daemon
gets its own console window; on Linux it logs to
`$XDG_CACHE_HOME/moge-nuke/daemon-<uid>.log`. The wire protocol is documented
at the top of `moge_daemon.py`; `moge_client.py` is the reference client.

## Building the plugin

```text
./ofx/build.sh                                            # Linux
powershell -ExecutionPolicy Bypass -File ofx\build.ps1    # Windows
```

Raw OFX C API against the headers in `ofx/include/openfx`; no other
dependencies (libc on Linux, ws2_32 on Windows). Output lands in
`ofx/build[-linux]/MoGe3.ofx.bundle`. Close Nuke before reinstalling on
Windows (it locks the loaded `.ofx`). There is no Linux prebuilt on purpose:
a binary built on a recent distro needs a newer glibc than Rocky/RHEL ship,
and building on the target takes seconds.

The plugin finds the repo through `moge3.cfg` next to the `.ofx` (written by
the installer) or the `MOGE_NUKE_ROOT` environment variable. Without either,
fill in the Setup tab by hand.

## Tests

```text
.venv/bin/python daemon/test_moge_daemon.py   # protocol + parity with model.infer()
.venv/bin/python ofx/tests/test_plugin.py     # the built .ofx through a mini OFX host, no Nuke needed
.venv/bin/python tools/warmup.py              # load + time the model, populate the Triton cache
Nuke17.1 -t ofx/test_nuke_render.py           # the real thing, writes output/ofx_test/*.exr
```

`ofx/tests/mini_host.cpp` is a minimal OFX host (property, parameter, image
effect and message suites) that loads the plugin, renders one frame and
compares against a direct daemon request, so the plugin is verified bit for
bit on both platforms without a Nuke licence.

## Notes and limits

* Each new frame blocks the Viewer for the inference time (OFX renders are
  synchronous). Nuke caches the result afterwards.
* Per-frame inference: depth can drift slightly frame to frame on video
  (mostly global scale/shift). Normals are stable and refiner-independent.
* Refine steps > 0 is not bit-exact run to run (about 0.7 % in depth).
* A daemon started with "exits with Nuke" belongs to the Nuke session that
  launched it; a second session restarts it when that one closes.

## Licence

MIT (see `LICENSE`). MoGe is Copyright (c) Microsoft Corporation, MIT; the
OpenFX headers are BSD-3-Clause. See `THIRD_PARTY_NOTICES.md`. Model weights
come from Hugging Face under the MIT licence of the original release.
