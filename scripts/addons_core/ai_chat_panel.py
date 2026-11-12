# ai_chat_panel.py — In-app AI chat panel for Blender on Android (and desktop) v1.1.0
# N-panel "AI Chat": type a prompt, the panel calls an OpenAI-compatible API
# (pure stdlib urllib — no requests, no pip), gets back structured ops, and
# applies them through ai_bridge's command dispatcher (same code the socket
# bridge uses, so socket and panel can never disagree).
#
# Safety, in order:
#   1. M0.2: the LLM HTTP call runs in a BACKGROUND thread; the result is
#      delivered via bpy.app.timers. The old 90s main-thread block (which
#      could freeze Android) is gone.
#   2. M0.3: dry-run stores the reviewed plan; an Apply button applies THAT
#      stored plan — no second LLM call.
#   3. M0.4: execute_code is NOT in the panel's allowed ops unless the
#      "Allow code" toggle is ON (default off, with a visible warning).
#      The bridge's own execute_code (token-gated socket) is unaffected.
#   4. Dry-run is ON by default: the plan is shown, nothing is applied until
#      the user taps Apply (or turns dry-run off and taps Send).
#   5. An undo checkpoint (bpy.ops.ed.undo_push) is made before any applied batch.
#   6. Ops are validated before applying (unknown op / missing object /
#      disallowed code = plan error).
#   7. The API key lives in Add-on Preferences as PLAINTEXT in userpref.blend
#      (see the preference description — use a revokable key).
#
# The LLM is asked for PLAIN JSON ops in the reply content (no native tool_calls),
# so this works with OpenAI, local llama.cpp servers, and other compatible APIs.

bl_info = {
    "name": "AI Chat",
    "author": "Muse (for anony)",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > AI Chat",
    "description": "In-app AI chat panel: plain-language prompts become Blender ops",
    "category": "System",
}

import bpy
import json
import ssl
import threading
import traceback
import urllib.request
import urllib.error

from bpy.props import StringProperty, IntProperty, BoolProperty
from bpy.types import AddonPreferences, Operator, Panel

ADDON_NAME = __package__ or "ai_chat_panel"
BRIDGE_NAME = "ai_bridge"

# Ops the panel may emit. execute_code is deliberately NOT in this set: it is
# only allowed when the per-session "Allow code" toggle is on (see _validate_plan).
ALLOWED_OPS = {
    "create_object", "delete_object", "set_transform", "create_material",
    "add_modifier", "get_scene_info", "get_object_info", "render",
    "save_blend",
}
READONLY_OPS = {"get_scene_info", "get_object_info"}

_BASE_SYSTEM_PROMPT = """You control Blender through structured ops. Reply with ONLY a JSON object:
{"ops": [{"op": "<name>", "params": {...}}, ...], "note": "<one short line>"}
No markdown, no code fences, no extra text.

Available ops (params as in the AI Bridge):
- create_object: kind cube/sphere/cylinder/cone/torus/plane/monkey/light/camera, name, location [x,y,z]
- set_transform: name, location [x,y,z], rotation_euler [rx,ry,rz] (radians), scale [x,y,z]
- delete_object: name
- create_material: name, color [r,g,b,a] 0..1, roughness, metallic, assign_to (object name)
- add_modifier: name, modifier_type SUBSURF/BEVEL, levels (max 2), width
- get_scene_info: {} — use first when you need to see the scene
- get_object_info: name
- render: filepath (omit for default), width, height (max 512 each), samples (max 32), engine WORKBENCH/EEVEE
- save_blend: filepath (omit for default)
{CODE_LINE}
Rules: keep ops minimal (usually 1-4). Never invent object names that do not exist;
call get_scene_info first if unsure. Locations are in metres, Z is up.
"""


def _system_prompt(allow_code):
    code_line = ("- execute_code: code (Python with bpy) — ONLY when no structured op can do it."
                 if allow_code else
                 "- execute_code: NOT available. Never emit it; use structured ops only.")
    return _BASE_SYSTEM_PROMPT.replace("{CODE_LINE}", code_line)


# Pending LLM result, written by the worker thread, read by the timer (main thread).
_PENDING = {"done": True, "ops": None, "note": "", "error": None, "dry_run": True}
_PENDING_LOCK = threading.Lock()


def _pending_set(**kw):
    with _PENDING_LOCK:
        _PENDING.update(kw)


def _pending_get():
    with _PENDING_LOCK:
        return dict(_PENDING)


# ---------------------------------------------------------------- bridge access (MAIN THREAD ONLY)
def _dispatch(op, params):
    """Apply one op on the main thread via ai_bridge's dispatcher (imported),
    falling back to the localhost socket bridge if the addon module is absent."""
    try:
        import ai_bridge
        return ai_bridge._dispatch({"command": op, "params": params or {}})
    except Exception:
        pass
    import socket
    try:
        prefs = bpy.context.preferences.addons[ADDON_NAME].preferences
        token = prefs.bridge_token
        port = prefs.bridge_port
    except Exception:
        token, port = "change-me", 9876
    msg = json.dumps({"token": token, "command": op, "params": params or {}}) + "\n"
    with socket.create_connection(("127.0.0.1", port), timeout=180) as s:
        s.sendall(msg.encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                raise RuntimeError("bridge connection closed")
            buf += chunk
    return json.loads(buf.decode())


def _validate_plan(ops, allow_code):
    """Dry-run validation: op names known, params sane, targets exist.
    Sequential-aware: objects created by an earlier step count as existing
    for later steps (a plan that creates a cube then materials it is valid)."""
    errors = []
    known = set(bpy.data.objects.keys())
    for i, step in enumerate(ops):
        op = step.get("op")
        params = step.get("params") or {}
        if op == "execute_code" and not allow_code:
            errors.append("step %d: execute_code is disabled (turn on 'Allow code')" % i)
            continue
        if op not in ALLOWED_OPS and op != "execute_code":
            errors.append("step %d: unknown op %r" % (i, op))
            continue
        if op == "create_object" and params.get("name"):
            known.add(params["name"])
        if op == "delete_object":
            name = params.get("name")
            if name and name not in known:
                errors.append("step %d: object not found: %r" % (i, name))
            elif name:
                known.discard(name)
        if op in ("set_transform", "add_modifier", "get_object_info"):
            name = params.get("name")
            if name and name not in known:
                errors.append("step %d: object not found: %r" % (i, name))
        if op == "create_material" and params.get("assign_to"):
            if params["assign_to"] not in known:
                errors.append("step %d: assign target not found: %r" % (i, params["assign_to"]))
    return errors


def _apply_plan(ops, dry_run, allow_code):
    """Returns (applied_results, errors). Dry-run only validates."""
    errors = _validate_plan(ops, allow_code)
    if errors:
        return [], errors
    if dry_run:
        return [{"dry_run": True, "op": s.get("op")} for s in ops], []
    try:
        bpy.ops.ed.undo_push(message="AI Chat: apply plan (%d ops)" % len(ops))
    except Exception:
        pass
    results = []
    for step in ops:
        r = _dispatch(step.get("op"), step.get("params") or {})
        results.append(r)
        if r.get("status") != "ok":
            results.append({"status": "error", "message": "stopped at failed op"})
            break
    return results, []


# ---------------------------------------------------------------- LLM client (pure stdlib; WORKER THREAD ONLY)
def _chat_completions(snap, messages):
    url = (snap["base_url"] or "").rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": snap["model"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1500,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % snap["api_key"],
    })
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RuntimeError("LLM HTTP %d: %s" % (e.code, detail))
    except Exception as e:
        raise RuntimeError("LLM request failed: %s" % e)


def _parse_plan(content):
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    plan = json.loads(content)
    ops = plan.get("ops") or []
    if not isinstance(ops, list):
        raise RuntimeError("LLM 'ops' was not a list")
    return ops, plan.get("note", "")


def _llm_worker(prompt, scene_summary, snap, dry_run):
    """BACKGROUND THREAD. Touches no bpy. Writes the result to _PENDING."""
    try:
        messages = [
            {"role": "system", "content": _system_prompt(snap["allow_code"])},
            {"role": "user", "content": "Current scene:\n%s\n\nRequest: %s" % (scene_summary, prompt)},
        ]
        data = _chat_completions(snap, messages)
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except Exception:
            raise RuntimeError("unexpected LLM response: %s" % json.dumps(data)[:500])
        ops, note = _parse_plan(content)
        _pending_set(done=True, ops=ops, note=note, error=None, dry_run=dry_run)
    except Exception as e:
        _pending_set(done=True, ops=None, note="", error=str(e), dry_run=dry_run)


def _scene_summary_for_llm():
    r = _dispatch("get_scene_info", {})
    if r.get("status") != "ok":
        return "scene info unavailable"
    res = r["result"]
    lines = ["Scene '%s', %d objects, %d verts, %d polys:" % (
        res.get("scene"), res.get("object_count", 0),
        res.get("total_vertices", 0), res.get("total_polygons", 0))]
    for o in res.get("objects", [])[:30]:
        lines.append("- %s (%s) at %s" % (o["name"], o["type"], o["location"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- UI + delivery (MAIN THREAD ONLY)
def _wrap(text, width=44):
    """Word-wrap for narrow phone screens (layout.label does not wrap)."""
    out = []
    for raw in (text or "").split("\n"):
        line = raw
        while len(line) > width:
            cut = line.rfind(" ", 0, width)
            cut = cut if cut > 10 else width
            out.append(line[:cut])
            line = line[cut:].lstrip()
        out.append(line)
    return out


def _history_add(text):
    try:
        props = bpy.context.scene.ai_chat_props
        lines = (props.history + "\n" + text).strip().split("\n")
        props.history = "\n".join(lines[-40:])
    except Exception:
        pass


def _deliver_pending():
    """Timer callback (main thread): apply the worker's result when ready."""
    p = _pending_get()
    if not p["done"]:
        return 0.2
    props = bpy.context.scene.ai_chat_props
    if p["error"]:
        _history_add("ERROR: %s" % p["error"])
        return None
    ops = p["ops"] or []
    plan_text = " | ".join("%s(%s)" % (s.get("op"), json.dumps(s.get("params") or {})) for s in ops)
    if p["dry_run"]:
        errors = _validate_plan(ops, props.allow_code)
        if errors:
            _history_add("PLAN BLOCKED: " + "; ".join(errors))
            return None
        props.pending_plan_json = json.dumps(ops)
        props.pending_note = p["note"]
        _history_add("PLAN (dry-run): " + plan_text)
        if p["note"]:
            _history_add("AI: " + p["note"])
        _history_add("Tap Apply to run this plan.")
    else:
        results, errors = _apply_plan(ops, dry_run=False, allow_code=props.allow_code)
        if errors:
            _history_add("PLAN BLOCKED: " + "; ".join(errors))
            return None
        ok = all(r.get("status") == "ok" for r in results if isinstance(r, dict))
        _history_add("APPLIED: " + plan_text)
        if p["note"]:
            _history_add("AI: " + p["note"])
        _history_add("Applied %d ops (%s)." % (len(ops), "ok" if ok else "with errors"))
    return None


class AIChatPreferences(AddonPreferences):
    bl_idname = ADDON_NAME

    base_url: StringProperty(name="API base URL", default="",
                             description="OpenAI-compatible base URL, e.g. https://api.openai.com/v1 (no trailing /chat/completions)")
    api_key: StringProperty(name="API key", default="", subtype="PASSWORD",
                            description="Stored as PLAINTEXT in Blender preferences (userpref.blend) on this device. Use a key you can revoke; never reuse a key that guards anything else.")
    model: StringProperty(name="Model", default="gpt-4o-mini")
    bridge_token: StringProperty(name="Bridge token", default="change-me",
                                 description="Only used if the AI Bridge addon module is not importable")
    bridge_port: IntProperty(name="Bridge port", default=9876, min=1024, max=65535)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "base_url")
        layout.prop(self, "api_key")
        layout.prop(self, "model")
        layout.prop(self, "bridge_token")
        layout.prop(self, "bridge_port")


class AIChatProperties(bpy.types.PropertyGroup):
    prompt: StringProperty(name="Prompt", default="", description="What should Blender do?")
    dry_run: BoolProperty(name="Dry-run (show plan first)", default=True)
    allow_code: BoolProperty(
        name="Allow code",
        default=False,
        description="Lets the AI emit execute_code (arbitrary Python inside Blender). OFF by default — only enable for a model you trust, on a scene you have saved.")
    history: StringProperty(name="History", default="")
    pending_plan_json: StringProperty(name="Pending plan", default="")
    pending_note: StringProperty(name="Pending note", default="")


class AICHAT_OT_send(Operator):
    bl_idname = "ai_chat.send"
    bl_label = "Send to AI"
    bl_description = "Send the prompt to the AI (network runs in the background; the plan is delivered when ready)"

    def execute(self, context):
        props = context.scene.ai_chat_props
        prefs = context.preferences.addons[ADDON_NAME].preferences
        prompt = (props.prompt or "").strip()
        if not prompt:
            self.report({"WARNING"}, "Type a prompt first")
            return {"CANCELLED"}
        if not prefs.base_url or not prefs.api_key:
            self.report({"WARNING"}, "Set API base URL + key in Add-on Preferences first")
            return {"CANCELLED"}
        if not _pending_get()["done"]:
            self.report({"WARNING"}, "Still waiting for the last reply")
            return {"CANCELLED"}
        _history_add("YOU: " + prompt)
        # Main thread: scene summary + snapshot plain values (worker never touches bpy).
        scene = _scene_summary_for_llm()
        snap = {"base_url": prefs.base_url, "api_key": prefs.api_key,
                "model": prefs.model, "allow_code": props.allow_code}
        _pending_set(done=False, ops=None, note="", error=None, dry_run=props.dry_run)
        t = threading.Thread(target=_llm_worker,
                             args=(prompt, scene, snap, props.dry_run), daemon=True)
        t.start()
        if not bpy.app.timers.is_registered(_deliver_pending):
            bpy.app.timers.register(_deliver_pending, first_interval=0.2)
        props.prompt = ""
        return {"FINISHED"}


class AICHAT_OT_apply(Operator):
    bl_idname = "ai_chat.apply_plan"
    bl_label = "Apply stored plan"
    bl_description = "Apply the dry-run plan shown above (no second LLM call)"

    def execute(self, context):
        props = context.scene.ai_chat_props
        raw = (props.pending_plan_json or "").strip()
        if not raw:
            self.report({"WARNING"}, "No stored plan — send a prompt first (dry-run on)")
            return {"CANCELLED"}
        try:
            ops = json.loads(raw)
        except Exception as e:
            self.report({"ERROR"}, "Stored plan is corrupt: %s" % e)
            return {"CANCELLED"}
        results, errors = _apply_plan(ops, dry_run=False, allow_code=props.allow_code)
        if errors:
            _history_add("PLAN BLOCKED: " + "; ".join(errors))
            self.report({"ERROR"}, "Plan blocked: " + "; ".join(errors)[:200])
            return {"CANCELLED"}
        ok = all(r.get("status") == "ok" for r in results if isinstance(r, dict))
        plan_text = " | ".join("%s(%s)" % (s.get("op"), json.dumps(s.get("params") or {})) for s in ops)
        _history_add("APPLIED (stored plan): " + plan_text)
        props.pending_plan_json = ""
        props.pending_note = ""
        self.report({"INFO" if ok else "WARNING"}, "Applied %d ops" % len(ops))
        return {"FINISHED"}


class AICHAT_PT_panel(Panel):
    bl_label = "AI Chat"
    bl_idname = "AICHAT_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Chat"

    def draw(self, context):
        layout = self.layout
        props = context.scene.ai_chat_props
        # Big touch-friendly prompt box + send.
        layout.prop(props, "prompt", text="")
        layout.prop(props, "dry_run")
        layout.prop(props, "allow_code")
        if props.allow_code:
            layout.label(text="WARNING: AI may run any Python.")
            layout.label(text="Save your scene first.")
        row = layout.row()
        row.scale_y = 1.6
        row.operator("ai_chat.send", text="Send to AI")
        if (props.pending_plan_json or "").strip():
            row2 = layout.row()
            row2.scale_y = 1.6
            row2.operator("ai_chat.apply_plan", text="Apply stored plan")
            for line in _wrap("Plan: " + props.pending_plan_json, 44):
                layout.label(text=line)
            if props.pending_note:
                for line in _wrap("AI: " + props.pending_note, 44):
                    layout.label(text=line)
        layout.separator()
        layout.label(text="History (latest last):")
        for raw in (props.history or "").split("\n")[-8:]:
            for line in _wrap(raw, 44):
                if line.strip():
                    layout.label(text=line)


_classes = (AIChatPreferences, AIChatProperties, AICHAT_OT_send,
            AICHAT_OT_apply, AICHAT_PT_panel)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ai_chat_props = bpy.props.PointerProperty(type=AIChatProperties)


def unregister():
    del bpy.types.Scene.ai_chat_props
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
