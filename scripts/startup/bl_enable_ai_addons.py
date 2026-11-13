# SPDX-FileCopyrightText: 2026 SkCena
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Enable the bundled SkCena AI add-ons (AI Bridge + AI Chat) on every launch.
# Blender runs scripts/startup/*.py at startup, after user preferences load.
# This is idempotent: enabling an already-enabled add-on is a no-op, and it
# re-enables them even if the user disabled them without saving preferences.
# It must never break startup, so every failure is caught and only printed.
#
# NOTE: do NOT use bpy.ops.preferences.addon_enable here — operators cannot
# run during startup (restricted context, no view_layer). addon_utils.enable()
# is the operator-free equivalent and is designed for restricted context.
#
# NOTE 2: the add-ons live in scripts/addons_core/ (NOT scripts/addons/).
# Blender only scans addons_core/ under the FIRST (system) script path;
# scripts/addons/ is only scanned under the user script path.
#
# NOTE 3: startup scripts run BEFORE addon_utils._initialize_once() appends
# the add-on dirs to sys.path, so the modules are not importable yet at this
# point. Bootstrap sys.path here first (same layout as addon_utils.paths()).
"""Enable bundled AI add-ons (ai_bridge, ai_chat_panel, touch_sidebar) at startup."""

import os
import sys

import addon_utils
import bpy


def _ensure_addon_paths():
    for _i, _p in enumerate(bpy.utils.script_paths()):
        _d = os.path.join(_p, "addons_core" if _i == 0 else "addons")
        if os.path.isdir(_d) and _d not in sys.path:
            sys.path.append(_d)


def _ensure_addon_enabled(module_name):
    try:
        if module_name in bpy.context.preferences.addons:
            return
        mod = addon_utils.enable(module_name, default_set=True)
        if mod is None:
            print("[skcena-ai] could not enable add-on:", module_name)
        else:
            print("[skcena-ai] enabled bundled add-on:", module_name)
    except Exception as exc:  # noqa: BLE001 - startup must never fail
        print("[skcena-ai] could not enable add-on", module_name, ":", exc)


def _enable_all():
    _ensure_addon_paths()
    for _module in ("ai_bridge", "ai_chat_panel", "touch_sidebar"):
        _ensure_addon_enabled(_module)


def register():
    _enable_all()


def unregister():
    pass


# Startup scripts are executed at import; register() is also called by
# Blender for registerable scripts. Either path enables the add-ons (the
# second call is a no-op), so this works on both old and new Blender.
_enable_all()
