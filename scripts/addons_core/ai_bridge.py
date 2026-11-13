# ai_bridge.py — AI Bridge for Blender on Android (and desktop) v1.1.0
# BlenderMCP-style local bridge, minimal + hardened for Blender 5.x alpha.
# - TCP server on 127.0.0.1:9876, newline-delimited JSON, shared-secret token.
# - Network I/O runs on background (daemon) threads. ALL bpy calls run on the
#   main thread via a queue drained by bpy.app.timers.
# - M0.1: token+port are cached on the MAIN thread at server start; the network
#   thread never touches bpy. Never starts on "change-me": a random token is
#   generated on first enable and shown in the panel. A bad token closes the
#   connection immediately.
# - M0.5: crash guardrails — polygon/vertex caps, Subsurf level cap, render
#   size/sample caps, per-command elapsed reporting, auto-save (copy) before
#   risky ops, heavy work split into small timer chunks via the batch command.
# - M0.6: render takes an engine param; default is WORKBENCH (EEVEE on Mali is
#   unverified on the first phone test).
# - No /tmp paths anywhere: outputs default to Download/Blender on Android.
#
# SECURITY: binds loopback only + token. Anyone on the phone with the token can
# run arbitrary code inside Blender (execute_code). Never expose the port, never
# share or log the token. Change/rotate the token in Add-on Preferences.

bl_info = {
    "name": "AI Bridge",
    "author": "Muse (for anony)",
    "version": (1, 1, 2),  # 1.1.1: bind on main thread under a lock (startup double-enable race)
    # 1.1.2: "Copy token" operator (NR4 touch pass — the token label is not
    # copyable and stays truncated "..."; the operator copies it to the
    # clipboard instead). On-phone clipboard sync is UNVERIFIED — the
    # operator sets window_manager.clipboard; whether the Android port
    # forwards that to the system clipboard is a phone test, not assumed.
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > AI Bridge",
    "description": "Local token-auth JSON bridge (127.0.0.1:9876) so an AI agent can drive Blender",
    "category": "System",
}

import bpy
import json
import os
import queue
import secrets
import socket
import sys
import threading
import time
import traceback
from contextlib import redirect_stdout
from io import StringIO

from bpy.props import StringProperty, IntProperty, BoolProperty
from bpy.types import AddonPreferences, Operator, Panel

ADDON_NAME = __package__ or "ai_bridge"
DEFAULT_PORT = 9876
DEFAULT_HOST = "127.0.0.1"
MAX_MESSAGE_BYTES = 1024 * 1024  # 1 MB per JSON message

# M0.5 guardrail caps (phone-first; tune after M3 measurements).
MAX_TOTAL_POLYGONS = 200_000
MAX_TOTAL_VERTICES = 300_000
MAX_SUBSURF_LEVELS = 2
MAX_RENDER_DIM = 512
MAX_RENDER_SAMPLES = 32
MAX_EXECUTE_CODE_CHARS = 100_000
MAX_BATCH_OPS = 50
COMMAND_TIME_BUDGET_S = 120  # soft budget: reported, not a hard kill (a running
# bpy op cannot be interrupted safely; caps above are the real protection).

# ---------------------------------------------------------------- cached auth (main thread only writes)
# The network thread reads this plain dict — never bpy, never preferences.
_AUTH = {"token": None, "port": DEFAULT_PORT}

_job_queue = queue.Queue()
_server_thread = None
_server_socket = None
_server_running = False
# Serialises start/stop so a rapid enable->disable->enable cycle (which is
# exactly what Blender does at startup: the startup script enables the add-on,
# then _initialize_once() re-enables it) can never leave two threads racing
# to bind the same port. The bind happens on the main thread, inside the lock.
_server_lock = threading.Lock()
_job_counter = [0]
# Batch state: heavy work split into one op per timer tick (M0.5).
_BATCH = {"id": 0, "ops": [], "results": [], "index": 0, "done": True, "error": None}


def _next_job_id():
    _job_counter[0] += 1
    return _job_counter[0]


def _is_android():
    """True only on a real Android runtime. The old check (makedirs succeeds)
    was wrong: on a root desktop it happily created /storage/emulated/0/...
    and then treated the VM as a phone."""
    if os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"):
        return True
    return os.path.exists("/system/build.prop")


def _android_download_dir():
    """Download/Blender on Android; None on desktop."""
    if not _is_android():
        return None
    for base in ("/storage/emulated/0/Download/Blender",
                 "/sdcard/Download/Blender"):
        try:
            os.makedirs(base, exist_ok=True)
            return base
        except Exception:
            continue
    return None


def _default_render_path():
    d = _android_download_dir()
    return os.path.join(d, "ai_bridge_render.png") if d else os.path.join(
        os.path.expanduser("~"), "ai_bridge_render.png")


def _default_blend_path():
    d = _android_download_dir()
    return os.path.join(d, "ai_bridge_scene.blend") if d else os.path.join(
        os.path.expanduser("~"), "ai_bridge_scene.blend")


def _autosave_path():
    d = _android_download_dir()
    return os.path.join(d, "ai_bridge_autosave.blend") if d else os.path.join(
        os.path.expanduser("~"), "ai_bridge_autosave.blend")


def _ensure_token():
    """MAIN THREAD ONLY. Returns the token to cache. Env var wins (tests);
    otherwise prefs; generates + saves a random token when unset/sentinel."""
    env = os.environ.get("BLENDER_AI_BRIDGE_TOKEN", "").strip()
    if env:
        return env
    try:
        prefs = bpy.context.preferences.addons[ADDON_NAME].preferences
        tok = (prefs.token or "").strip()
        if not tok or tok == "change-me":
            tok = secrets.token_urlsafe(24)
            prefs.token = tok
            try:
                bpy.ops.wm.save_userpref()
            except Exception:
                pass
        return tok
    except Exception:
        return secrets.token_urlsafe(24)


def _get_port_main_thread():
    try:
        prefs = bpy.context.preferences.addons[ADDON_NAME].preferences
        return int(prefs.port or DEFAULT_PORT)
    except Exception:
        return int(os.environ.get("BLENDER_AI_BRIDGE_PORT", DEFAULT_PORT))


def _scene_totals():
    verts = polys = 0
    for o in bpy.context.scene.objects:
        if o.type == "MESH" and o.data is not None:
            try:
                verts += len(o.data.vertices)
                polys += len(o.data.polygons)
            except Exception:
                pass
    return verts, polys


def _autosave():
    """Save a COPY of the current file before a risky op. Returns path or None."""
    path = _autosave_path()
    try:
        bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
        return path if os.path.exists(path) else None
    except Exception:
        return None


# ---------------------------------------------------------------- commands (MAIN THREAD ONLY)
def _ok(data=None, **extra):
    out = {"status": "ok"}
    if data is not None:
        out["result"] = data
    out.update(extra)
    return out


def _err(message, code="error"):
    return {"status": "error", "code": code, "message": str(message)}


def _refused(message):
    return _err("refused: too heavy — %s" % message, "too_heavy")


def _scene_info(_params):
    scene = bpy.context.scene
    verts, polys = _scene_totals()
    objs = []
    for o in scene.objects:
        objs.append({"name": o.name, "type": o.type,
                     "location": [round(v, 4) for v in o.location],
                     "visible": not o.hide_viewport})
    return _ok({"scene": scene.name,
                "objects": objs,
                "object_count": len(objs),
                "total_vertices": verts,
                "total_polygons": polys,
                "frame": scene.frame_current,
                "render_engine": scene.render.engine,
                "blender_version": ".".join(str(v) for v in bpy.app.version),
                "download_dir": _android_download_dir()})


def _object_info(params):
    name = params.get("name")
    obj = bpy.data.objects.get(name) if name else None
    if obj is None:
        return _err("object not found: %r" % name, "not_found")
    info = {"name": obj.name, "type": obj.type,
            "location": list(obj.location),
            "rotation_euler": list(obj.rotation_euler),
            "scale": list(obj.scale),
            "dimensions": list(obj.dimensions),
            "materials": [m.name if m else None for m in obj.data.materials]
            if hasattr(obj.data, "materials") else []}
    if obj.type == "MESH":
        info["vertices"] = len(obj.data.vertices)
        info["polygons"] = len(obj.data.polygons)
    return _ok(info)


def _create_object(params):
    kind = (params.get("kind") or "cube").lower()
    name = params.get("name")
    loc = params.get("location") or (0, 0, 0)
    ops = {"cube": bpy.ops.mesh.primitive_cube_add,
           "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
           "cylinder": bpy.ops.mesh.primitive_cylinder_add,
           "cone": bpy.ops.mesh.primitive_cone_add,
           "torus": bpy.ops.mesh.primitive_torus_add,
           "plane": bpy.ops.mesh.primitive_plane_add,
           "monkey": bpy.ops.mesh.primitive_monkey_add,
           "light": bpy.ops.object.light_add,
           "camera": bpy.ops.object.camera_add}
    op = ops.get(kind)
    if op is None:
        return _err("unknown kind: %r (cube/sphere/cylinder/cone/torus/plane/monkey/light/camera)" % kind,
                    "bad_kind")
    _v, polys = _scene_totals()
    if polys >= MAX_TOTAL_POLYGONS:
        return _refused("scene already at %d polygons (cap %d)" % (polys, MAX_TOTAL_POLYGONS))
    kwargs = {"location": tuple(float(v) for v in loc)}
    if kind == "light":
        kwargs["type"] = params.get("light_type", "POINT")
    op(**kwargs)
    obj = bpy.context.active_object
    if name:
        obj.name = name
    _v, polys = _scene_totals()
    if polys > MAX_TOTAL_POLYGONS:
        bpy.data.objects.remove(obj, do_unlink=True)
        return _refused("new object would exceed %d polygons (cap)" % MAX_TOTAL_POLYGONS)
    try:
        bpy.ops.ed.undo_push(message="AI Bridge: create %s" % obj.name)
    except Exception:
        pass
    return _ok({"name": obj.name, "type": obj.type, "location": list(obj.location)})


def _delete_object(params):
    name = params.get("name")
    obj = bpy.data.objects.get(name) if name else None
    if obj is None:
        return _err("object not found: %r" % name, "not_found")
    bpy.data.objects.remove(obj, do_unlink=True)
    return _ok({"deleted": name})


def _set_transform(params):
    name = params.get("name")
    obj = bpy.data.objects.get(name) if name else None
    if obj is None:
        return _err("object not found: %r" % name, "not_found")
    if params.get("location") is not None:
        obj.location = tuple(float(v) for v in params["location"])
    if params.get("rotation_euler") is not None:
        obj.rotation_euler = tuple(float(v) for v in params["rotation_euler"])
    if params.get("scale") is not None:
        obj.scale = tuple(float(v) for v in params["scale"])
    return _ok({"name": obj.name, "location": list(obj.location)})


def _create_material(params):
    name = params.get("name") or "AIMaterial"
    color = params.get("color") or [0.8, 0.2, 0.2, 1.0]
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    try:
        # use_nodes is deprecated in 5.0 (still works in 5.x); guard anyway.
        mat.use_nodes = True
    except Exception:
        pass
    bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.use_nodes else None
    if bsdf is not None:
        # Socket name "Base Color" is stable across 4.x and 5.x.
        bsdf.inputs["Base Color"].default_value = tuple(float(v) for v in color)
        if "Roughness" in bsdf.inputs and params.get("roughness") is not None:
            bsdf.inputs["Roughness"].default_value = float(params["roughness"])
        if "Metallic" in bsdf.inputs and params.get("metallic") is not None:
            bsdf.inputs["Metallic"].default_value = float(params["metallic"])
    target = params.get("assign_to")
    if target:
        obj = bpy.data.objects.get(target)
        if obj is not None and hasattr(obj.data, "materials"):
            if mat.name not in [m.name if m else None for m in obj.data.materials]:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat
    return _ok({"material": mat.name, "assigned_to": target})


def _add_modifier(params):
    name = params.get("name")
    obj = bpy.data.objects.get(name) if name else None
    if obj is None:
        return _err("object not found: %r" % name, "not_found")
    mtype = (params.get("modifier_type") or "SUBSURF").upper()
    if mtype == "SUBSURF" and int(params.get("levels") or 1) > MAX_SUBSURF_LEVELS:
        return _refused("subsurf level %s exceeds cap %d" % (params.get("levels"), MAX_SUBSURF_LEVELS))
    mod = obj.modifiers.new(name=params.get("modifier_name") or mtype, type=mtype)
    if mtype == "SUBSURF" and params.get("levels") is not None:
        mod.levels = int(params["levels"])
    if mtype == "BEVEL" and params.get("width") is not None:
        mod.width = float(params["width"])
    return _ok({"object": obj.name, "modifier": mod.name, "type": mod.type})


_ENGINE_IDS = {"WORKBENCH": "BLENDER_WORKBENCH",
               "EEVEE": "BLENDER_EEVEE",
               "EEVEE_NEXT": "BLENDER_EEVEE_NEXT",
               "CYCLES": "CYCLES"}


def _render(params):
    scene = bpy.context.scene
    path = params.get("filepath") or _default_render_path()
    width = int(params.get("width") or 256)
    height = int(params.get("height") or 256)
    samples = int(params.get("samples") or 16)
    if width > MAX_RENDER_DIM or height > MAX_RENDER_DIM:
        return _refused("render %dx%d exceeds cap %dpx per side" % (width, height, MAX_RENDER_DIM))
    if samples > MAX_RENDER_SAMPLES:
        return _refused("render samples %d exceeds cap %d" % (samples, MAX_RENDER_SAMPLES))
    scene.render.filepath = path
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    requested = (params.get("engine") or "").upper()
    if requested:
        engine_id = _ENGINE_IDS.get(requested)
        if engine_id is None:
            return _err("unknown engine: %r (WORKBENCH/EEVEE/EEVEE_NEXT/CYCLES)" % requested,
                        "bad_engine")
        try:
            scene.render.engine = engine_id
        except Exception:
            return _err("engine unavailable on this build: %s" % requested, "engine_unavailable")
    else:
        # Default: WORKBENCH first (EEVEE on Mali is unverified on the phone).
        for engine in ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = engine
                break
            except Exception:
                continue
    try:
        scene.eevee.taa_render_samples = samples
    except Exception:
        pass
    try:
        scene.cycles.samples = samples
    except Exception:
        pass
    bpy.ops.render.render(write_still=True)
    exists = os.path.exists(path)
    return _ok({"filepath": path, "exists": exists,
                "size_bytes": os.path.getsize(path) if exists else 0,
                "engine": scene.render.engine})


def _save_blend(params):
    path = params.get("filepath") or _default_blend_path()
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return _ok({"filepath": path, "exists": os.path.exists(path)})


def _execute_code(params):
    code = params.get("code") or ""
    if len(code) > MAX_EXECUTE_CODE_CHARS:
        return _refused("code %d chars exceeds cap %d" % (len(code), MAX_EXECUTE_CODE_CHARS))
    autosave = _autosave()
    namespace = {"bpy": bpy}
    try:
        import mathutils
        namespace["mathutils"] = mathutils
    except Exception:
        pass
    buf = StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<ai_bridge>", "exec"), namespace)
    except Exception:
        return _err(traceback.format_exc(limit=5), "exec_error")
    out = buf.getvalue()
    result = namespace.get("result", None)
    try:
        json.dumps(result)
    except Exception:
        result = repr(result)
    try:
        bpy.ops.ed.undo_push(message="AI Bridge: execute_code")
    except Exception:
        pass
    return _ok({"stdout": out[-4000:], "result": result, "autosave": autosave})


def _batch(params):
    ops = params.get("ops") or []
    if not isinstance(ops, list) or not ops:
        return _err("batch needs a non-empty 'ops' list", "bad_request")
    if len(ops) > MAX_BATCH_OPS:
        return _refused("batch of %d ops exceeds cap %d" % (len(ops), MAX_BATCH_OPS))
    for i, step in enumerate(ops):
        if (step.get("command") or step.get("op") or step.get("type")) not in _COMMANDS:
            return _err("batch step %d: unknown command" % i, "bad_request")
        if (step.get("command") or step.get("op") or step.get("type")) in ("batch", "batch_status"):
            return _err("batch step %d: nesting not allowed" % i, "bad_request")
    _BATCH["id"] += 1
    _BATCH["ops"] = ops
    _BATCH["results"] = []
    _BATCH["index"] = 0
    _BATCH["done"] = False
    _BATCH["error"] = None
    return _ok({"batch_id": _BATCH["id"], "queued": len(ops)})


def _batch_status(params):
    return _ok({"batch_id": _BATCH["id"], "done": _BATCH["done"],
                "completed": _BATCH["index"], "total": len(_BATCH["ops"]),
                "results": _BATCH["results"], "error": _BATCH["error"]})


_COMMANDS = {
    "ping": lambda p: _ok({"pong": True, "blender": ".".join(str(v) for v in bpy.app.version)}),
    "get_scene_info": _scene_info,
    "get_object_info": _object_info,
    "create_object": _create_object,
    "delete_object": _delete_object,
    "set_transform": _set_transform,
    "create_material": _create_material,
    "add_modifier": _add_modifier,
    "render": _render,
    "save_blend": _save_blend,
    "execute_code": _execute_code,
    "batch": _batch,
    "batch_status": _batch_status,
}


def _dispatch(msg):
    """Runs on the MAIN thread only. Returns a JSON-able dict."""
    t0 = time.monotonic()
    cmd = msg.get("command") or msg.get("type") or msg.get("op")
    if not cmd:
        return _err("missing 'command'", "bad_request")
    fn = _COMMANDS.get(cmd)
    if fn is None:
        return _err("unknown command: %r" % cmd, "unknown_command")
    try:
        resp = fn(msg.get("params") or {})
    except Exception:
        resp = _err(traceback.format_exc(limit=5), "handler_error")
    elapsed = time.monotonic() - t0
    if isinstance(resp, dict):
        resp["elapsed_s"] = round(elapsed, 3)
        if elapsed > COMMAND_TIME_BUDGET_S:
            resp["budget_warning"] = "command took %.1fs (budget %ds)" % (elapsed, COMMAND_TIME_BUDGET_S)
    return resp


def _pump_queue():
    """Timer callback (main thread): drain queued jobs + one batch op per tick."""
    processed = 0
    while processed < 20:
        try:
            job = _job_queue.get_nowait()
        except queue.Empty:
            break
        try:
            job["response"] = _dispatch(job["message"])
        except Exception:
            job["response"] = _err(traceback.format_exc(limit=3), "pump_error")
        job["done"].set()
        processed += 1
    # M0.5 chunking: one batch op per timer tick keeps the UI responsive.
    if not _BATCH["done"] and _BATCH["index"] < len(_BATCH["ops"]):
        step = _BATCH["ops"][_BATCH["index"]]
        msg = {"command": step.get("command") or step.get("op") or step.get("type"),
               "params": step.get("params") or {}}
        res = _dispatch(msg)
        _BATCH["results"].append(res)
        _BATCH["index"] += 1
        if res.get("status") != "ok":
            _BATCH["error"] = res.get("message", "batch op failed")
            _BATCH["done"] = True
        elif _BATCH["index"] >= len(_BATCH["ops"]):
            _BATCH["done"] = True
    return 0.1  # keep timer alive


# ---------------------------------------------------------------- network (BACKGROUND THREADS ONLY)
# These functions receive the cached token as an argument and never touch bpy.
def _handle_client(conn, token):
    try:
        buf = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                if len(line) > MAX_MESSAGE_BYTES:
                    conn.sendall((json.dumps(_err("message too large", "too_large")) + "\n").encode())
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception as e:
                    conn.sendall((json.dumps(_err("bad JSON: %s" % e, "bad_json")) + "\n").encode())
                    continue
                if msg.get("token") != token:
                    # M0.1: bad token closes the connection immediately.
                    conn.sendall((json.dumps(_err("bad token", "auth")) + "\n").encode())
                    return
                job = {"message": msg, "done": threading.Event(), "response": None}
                _job_queue.put(job)
                if not job["done"].wait(timeout=180):
                    resp = _err("timeout waiting for main thread", "timeout")
                else:
                    resp = job["response"] or _err("no response", "internal")
                conn.sendall((json.dumps(resp) + "\n").encode())
            if len(buf) > MAX_MESSAGE_BYTES:
                return
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _serve_forever(srv, token):
    """Accept loop. `srv` is already bound + listening (bound on the main
    thread in start_server, so a second start can never race this bind)."""
    while _server_running:
        try:
            conn, _addr = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        t = threading.Thread(target=_handle_client, args=(conn, token), daemon=True)
        t.start()
    try:
        srv.close()
    except Exception:
        pass


def start_server():
    """MAIN THREAD ONLY. Caches token+port, then starts server + timer pump.
    Never starts on the 'change-me' sentinel — a random token is generated."""
    global _server_thread, _server_running, _server_socket
    with _server_lock:
        if _server_running:
            return False
        token = _ensure_token()
        port = _get_port_main_thread()
        # Bind HERE, on the main thread, while holding the lock: the old code
        # bound inside the worker thread, so a fast stop->start cycle could
        # leave two threads racing for the port (EADDRINUSE in the loser).
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((DEFAULT_HOST, port))
        srv.listen(8)
        srv.settimeout(1.0)
        _server_socket = srv
        _AUTH["token"] = token
        _AUTH["port"] = port
        _server_running = True
        if not bpy.app.timers.is_registered(_pump_queue):
            bpy.app.timers.register(_pump_queue, first_interval=0.1, persistent=True)
        _server_thread = threading.Thread(
            target=_serve_forever, args=(srv, token), daemon=True)
        _server_thread.start()
        return True


def stop_server():
    global _server_running, _server_socket, _server_thread
    with _server_lock:
        _server_running = False
        srv = _server_socket
        _server_socket = None
        thread = _server_thread
        _server_thread = None
        if srv is not None:
            try:
                srv.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                srv.close()
            except Exception:
                pass
        try:
            if bpy.app.timers.is_registered(_pump_queue):
                bpy.app.timers.unregister(_pump_queue)
        except Exception:
            pass
        # Join while still holding the lock: close() alone does NOT free the
        # port while the accept thread is still blocked in accept() (the
        # kernel keeps the socket bound until that call returns). The accept
        # loop wakes on its 1s timeout and exits, which is what frees the
        # port — without this join, an immediate rebind fails EADDRINUSE.
        if thread is not None:
            thread.join(timeout=2.0)


def get_cached_token():
    """Main-thread helper for the panel: the token currently in use."""
    return _AUTH.get("token")


# ---------------------------------------------------------------- UI
class AIBridgePreferences(AddonPreferences):
    bl_idname = ADDON_NAME

    token: StringProperty(name="Bridge token", default="change-me",
                          description="Shared secret for 127.0.0.1:9876 clients. A random one is generated on first enable — never leave it as change-me.")
    port: IntProperty(name="Port", default=DEFAULT_PORT, min=1024, max=65535)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "token")
        layout.prop(self, "port")
        layout.label(text="Binds 127.0.0.1 only. Rotate the token if it ever leaks.")


class AIBRIDGE_OT_start(Operator):
    bl_idname = "ai_bridge.start_server"
    bl_label = "Start AI Bridge"
    bl_description = "Start the 127.0.0.1:9876 JSON bridge"

    def execute(self, context):
        started = start_server()
        self.report({"INFO"}, "AI Bridge started" if started else "AI Bridge already running")
        return {"FINISHED"}


class AIBRIDGE_OT_stop(Operator):
    bl_idname = "ai_bridge.stop_server"
    bl_label = "Stop AI Bridge"

    def execute(self, context):
        stop_server()
        self.report({"INFO"}, "AI Bridge stopped")
        return {"FINISHED"}


class AIBRIDGE_OT_copy_token(Operator):
    bl_idname = "ai_bridge.copy_token"
    bl_label = "Copy token"
    bl_description = "Copy the bridge token to the system clipboard (the token label above is not copyable)"

    def execute(self, context):
        tok = get_cached_token()
        if not tok:
            try:
                tok = context.preferences.addons[ADDON_NAME].preferences.token
            except Exception:
                tok = ""
        if not tok or tok == "change-me":
            self.report({"WARNING"}, "No token to copy yet — start the bridge first")
            return {"CANCELLED"}
        context.window_manager.clipboard = tok
        self.report({"INFO"}, "Bridge token copied to clipboard")
        return {"FINISHED"}


class AIBRIDGE_OT_regen_token(Operator):
    bl_idname = "ai_bridge.regenerate_token"
    bl_label = "Regenerate token"
    bl_description = "Generate a new random token and restart the server with it"

    def execute(self, context):
        prefs = context.preferences.addons[ADDON_NAME].preferences
        prefs.token = secrets.token_urlsafe(24)
        try:
            bpy.ops.wm.save_userpref()
        except Exception:
            pass
        was_running = _server_running
        if was_running:
            stop_server()
        if was_running:
            start_server()
        self.report({"INFO"}, "New token generated" + (" — server restarted" if was_running else ""))
        return {"FINISHED"}


class AIBRIDGE_PT_panel(Panel):
    bl_label = "AI Bridge"
    bl_idname = "AIBRIDGE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Bridge"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Server: 127.0.0.1:%d" % (_AUTH.get("port") or _get_port_main_thread()))
        layout.label(text="Status: %s" % ("running" if _server_running else "stopped"))
        tok = get_cached_token()
        if tok:
            layout.label(text="Token: %s" % tok)
        else:
            layout.label(text="Token: not started yet")
        layout.operator("ai_bridge.start_server")
        layout.operator("ai_bridge.stop_server")
        layout.operator("ai_bridge.copy_token")
        layout.operator("ai_bridge.regenerate_token")


_classes = (AIBridgePreferences, AIBRIDGE_OT_start, AIBRIDGE_OT_stop,
            AIBRIDGE_OT_copy_token, AIBRIDGE_OT_regen_token, AIBRIDGE_PT_panel)


def _on_load_post(_dummy):
    # Auto-start after a .blend loads, like blender-mcp does.
    try:
        start_server()
    except Exception:
        traceback.print_exc()


@bpy.app.handlers.persistent
def _load_post_handler(_dummy):
    _on_load_post(_dummy)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.app.handlers.load_post.append(_load_post_handler)
    try:
        start_server()
    except Exception:
        traceback.print_exc()


def unregister():
    stop_server()
    try:
        bpy.app.handlers.load_post.remove(_load_post_handler)
    except ValueError:
        pass
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
