"""Run inside `Nuke -t`: push docs/sample.jpg through the MoGe3 OFX node and
write depth + normals EXRs to output/, then print the node's knobs.

    Nuke17.1.exe -t ofx/test_nuke_render.py
"""
import os
import sys
import time

import nuke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
OUT = ROOT + "/output/ofx_test"
CLASS = "OFXcom.sumit.moge3_v1"

read = nuke.nodes.Read(file=ROOT + "/docs/sample.jpg")
read["colorspace"].setValue("sRGB")
try:
    node = nuke.createNode(CLASS, inpanel=False)
except Exception as exc:
    print("[test] FAILED to create", CLASS, ":", exc)
    sys.exit(2)
node.setInput(0, read)
for k in ("model", "output", "refineSteps", "inputColorspace", "pythonExe", "daemonScript",
          "port", "autoStart", "exitWithHost"):
    try:
        print("[test]   {0} = {1!r}".format(k, node[k].value()))
    except Exception as exc:
        print("[test]   {0}: {1}".format(k, exc))

os.makedirs(OUT, exist_ok=True)
for which, out_idx in (("depth", 0), ("normals", 1)):
    node["output"].setValue(out_idx)
    w = nuke.nodes.Write(inputs=[node], file="{0}/{1}.exr".format(OUT, which),
                         file_type="exr", datatype="32 bit float", channels="rgba")
    w["colorspace"].setValue("linear")
    t0 = time.time()
    nuke.execute(w, 1, 1)
    print("[test] wrote {0} in {1:.2f}s".format(which, time.time() - t0))
print("[test] DONE")
