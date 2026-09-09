# MoGe-3 OFX node + resident inference daemon

Date: 2026-09-09. Approved in chat ("Okay, lets build this, also I want a
dropdown menu to select how many refiner steps I want to use").

## Goal

A pull-based Nuke node (OFX) that runs the *complete* MoGe-3 model -- Triton
sparse refiner included -- on the Viewer's demand, with no bake step. The
existing bake node keeps the refiner but needs a Render button and a disk round
trip; the existing live (.cat) node is pull-based but cannot run the refiner.

## Why a daemon

The refiner is FlexGEMM = Triton = JIT-compiled from Python at runtime. Nothing
in-process (libtorch, TorchScript, TensorRT) can run it, and linking libtorch
into a plugin would collide with Foundry's own c10.dll / torch_cpu.dll by base
name. So the model runs in the existing `.venv` (torch 2.13 + triton-windows),
resident in one process, and the OFX plugin is a thin client.

## Components

1. `daemon/moge_daemon.py` -- TCP server on 127.0.0.1:<port>.
   Loads the model once (reloads only when a request names another file),
   serialises inference with a lock, idles forever (optional `--idle-timeout`).
   Also usable from any client (a Python test client is included).
2. `ofx/src/moge_ofx.cpp` -- raw OFX C API plugin
   `com.sumit.moge3` ("MoGe3", group "ML"). Filter context, float RGBA,
   no tiles / no multi-res (whole frame per render call). Per render:
   fetch the full source frame, hash it, ask the daemon (spawning it if it is
   not running), cache the 5-plane reply per instance, write the chosen pass.
3. Build: CMake + MSVC 2022, `ofx/build.ps1`, installs to
   `%OFX_PLUGIN_PATH%` = `C:\Users\sumit\.nuke\OFXPlugins\MoGe3.ofx.bundle`.
4. Nuke menu: "Depth + Normals (OFX, full model)" creates the OFX node.

## Wire protocol (v1, little-endian)

Request:  `MOGE` | u32 json_len | json | u32 data_len | data
  json: {"cmd":"infer"|"info"|"shutdown", "width","height", "model",
         "refine_steps","resolution_level","num_tokens","fov_x","fp16",
         "input_colorspace":"linear"|"srgb", "normal_space":"nuke"|"opencv",
         "apply_mask":0|1}
  data: float32 RGB interleaved, top row first, width*height*3.
Reply:    `MOGR` | i32 status (0 ok) | u32 width | u32 height | u32 channels
          | u32 msg_len | msg (utf-8 json or error text) | u32 data_len | data
  data (infer): float32 planar, top row first: nx, ny, nz, mask, depth.
  msg  (infer): {"fov_x":..., "elapsed":...}; (info): model/device/torch.

## Plugin parameters

MoGe tab: model (file), output [depth, normals], refine steps
[0 (off), 1, 2, 3 (default), 4, 5], resolution level 0-9 (9), num tokens
(0 = auto), lock fov x (0 = estimate), fp16, input colorspace
[scene-linear, sRGB], normal space [nuke, opencv], apply mask.
Setup tab: python exe, daemon script, port (47821), auto-start daemon.

## Output

RGBA float. normals: RGB = n (chosen axes), A = mask. depth: RGB = metric
depth, A = mask. Same layout as the bake node's EXRs, so the relight group
and existing comps work unchanged.

## Error handling

Daemon errors return status != 0 with the traceback as msg; the plugin shows
it via the OFX message suite and fails the render. Connect failure with
auto-start off -> clear message naming the port and the start command.

## Testing

- `daemon/test_moge_daemon.py`: protocol round trip against a live
  daemon, output parity with `model.infer()` on the alien plate, model
  reload, bad-request error path.
- Plugin: `ofx/test_nuke_render.py` under `Nuke17.1.exe -t` -- creates
  the node, auto-starts the daemon, writes depth + normals EXRs. Result
  (2026-09-09): normals 0.000 deg median vs direct inference, mask 99.998 %
  agreement, orientation and channel layout verified.

## Status

Built, installed to `~/.nuke/OFXPlugins/MoGe3.ofx.bundle`, tests passing.
Build note: VS 2022 is an incomplete install, so build.ps1 uses vcvars64 +
Ninja rather than the Visual Studio generator.
