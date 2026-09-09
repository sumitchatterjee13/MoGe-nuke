"""MoGe-nuke menu: Nodes > ML > MoGe3.

Nuke runs every menu.py on its plugin path; install.ps1 adds this folder via
nuke.pluginAddPath() in ~/.nuke/init.py.
"""

import os

import nuke

ICON_NAME = "MoGe.png"


def _find_icon():
    try:
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), ICON_NAME)
        if os.path.isfile(candidate):
            return candidate.replace("\\", "/")
    except NameError:  # menu.py is exec'd; __file__ is not guaranteed
        pass
    for entry in nuke.pluginPath():
        candidate = os.path.join(entry, ICON_NAME)
        if os.path.isfile(candidate):
            return candidate.replace("\\", "/")
    return ICON_NAME


ICON = _find_icon()


def _create():
    import moge_ofx
    return moge_ofx.create()


def _daemon(action):
    import moge_ofx
    return getattr(moge_ofx, action)()


_menu = nuke.menu("Nodes").addMenu("ML")
_moge = _menu.addMenu("MoGe3", ICON)
_moge.addCommand("Depth + Normals", _create, icon=ICON)
_daemon_menu = _moge.addMenu("Daemon")
_daemon_menu.addCommand("Start", lambda: _daemon("start"))
_daemon_menu.addCommand("Status", lambda: _daemon("status"))
_daemon_menu.addCommand("Stop", lambda: _daemon("stop"))
