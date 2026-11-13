# ai_chat_panel.py — In-app AI chat panel for Blender on Android (and desktop) v1.3.0
# (v1.2.0 header notes retained below; v1.3.0 = NR4 touch pass, 2026-11-13:
#  Send/Apply buttons 1.6x -> 2.0x tall, history rendered in a box for
#  readability, and one-tap quick actions (Add Cube / Add Light / Render /
#  Save) that call the bridge directly with no LLM call.
#  Honest limit on the history text: Blender's Python UI exposes NO per-label
#  font size, so the text cannot be made bigger from an add-on — the box +
#  spacing is the closest real improvement; genuinely larger text needs the
#  global UI Scale / Resolution Scale (NR3), not a panel change.)
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
#      disallowed code / HOLLOW plan with empty params = plan error).
#      v1.2.0: a plan whose ops carry no usable params is BLOCKED at dry-run
#      instead of "applying" nothing (seen on the phone 2026-11-13: the model
#      returned create_object({}) etc., Apply reported "Applied 4 ops", and
#      the scene gained only one invisible overlapping default cube).
#   7. The API key lives in Add-on Preferences as PLAINTEXT in userpref.blend
#      (see the preference description — use a revokable key).
#   8. v1.2.0: Apply reports honestly — "Applied X of N ops" plus the step
#      it stopped at, in history and in the status bar. The old "Applied N
#      ops" counted the plan length, not successes.
#
# The LLM is asked for PLAIN JSON ops in the reply content (no native tool_calls),
# so this works with OpenAI, local llama.cpp servers, and other compatible APIs.
# v1.2.0: the system prompt now carries full Blender + bridge knowledge —
# the exact op reference, phone guardrails, and two worked examples with real
# values — so the model builds real plans instead of empty-param shapes.

bl_info = {
    "name": "AI Chat",
    "author": "Muse (for anony)",
    "version": (1, 3, 0),
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

_BASE_SYSTEM_PROMPT = """You are the AI inside "Blender AI" — Blender 5.3.0 running on an Android phone.
You drive Blender ONLY through the structured ops below. These are the exact ops
of the AI Bridge built into this app (the same ops the phone's local socket
bridge on 127.0.0.1:9876 and the MCP server expose). There is no other tool and
no other way to touch the scene. Reply with ONLY a JSON object:
{"ops": [{"op": "<name>", "params": {...}}, ...], "note": "<one short line>"}
No markdown, no code fences, no extra text, no apologies.

RULES — every op MUST have complete, concrete params. Never emit "params": {}.
Every create names the object; every transform/material/modifier names its
target; every number is a real number, never a placeholder. If something cannot
be done with these ops, say so in "note" and emit an empty ops list — never
invent an op, a param, or an object name.
- Names: use the exact names from the scene summary. Never invent an object
  name that does not exist; call get_scene_info first when unsure. When you
  create objects, give each a clear name and reuse THAT name in later steps
  (create "WallNorth", then create_material assign_to "WallNorth").
- Space: metres, Z is up, rotation_euler is in RADIANS, scale is a multiplier.
  A default cube is 2m on a side, centred at its location. "size" is NOT a
  param — size comes from set_transform scale.
- Build in order: create parts first, then materials, then transforms. Keep
  the plan small and complete: usually 2-8 ops, one object per op.

OPS (exact params — defaults in brackets):
- get_scene_info: {} — lists every object (name, type, location) + scene totals.
  Use it first when the request refers to something already in the scene.
- get_object_info: name (required).
- create_object: kind cube/sphere/cylinder/cone/torus/plane/monkey/light/camera
  [cube], name, location [x,y,z] [0,0,0]. A cone's point is up; a cylinder's
  axis is Z. Light adds a point light; camera adds a camera.
- set_transform: name (required), location [x,y,z], rotation_euler [rx,ry,rz]
  in radians, scale [x,y,z]. Only the fields you give change; the rest stay.
- delete_object: name (required).
- create_material: name, color [r,g,b,a] each 0..1, roughness 0..1, metallic
  0..1, assign_to (object name). Without assign_to the material is created
  but sits unused on nothing — ALWAYS set assign_to.
- add_modifier: name (required), modifier_type SUBSURF/BEVEL, levels (max 2),
  width (bevel, in metres).
- render: filepath (omit for the default), width/height (max 512 each),
  samples (max 32), engine WORKBENCH/EEVEE/EEVEE_NEXT/CYCLES. Default engine
  is WORKBENCH — always use it on this phone unless the user asks for another.
  Cycles is NOT in this Lite build.
- save_blend: filepath (omit for the default).
{CODE_LINE}

PHONE LIMITS — hard caps, never exceed them or the op is refused:
- Scene totals: 200,000 polygons, 300,000 vertices. Keep scenes light; prefer
  a few big simple parts over many small ones.
- SUBSURF levels max 2. Render max 512x512, max 32 samples.
- One plan: max 50 ops. execute_code: max 100,000 chars, and the scene is
  auto-saved before it runs.
- Files: never use /tmp — on Android it fails. Omit filepath and the app uses
  Download/Blender.

EXAMPLE 1 — "make a simple house with walls and roof" (copy this shape):
{"ops": [
  {"op": "create_object", "params": {"kind": "cube", "name": "HouseBase", "location": [0, 0, 1]}},
  {"op": "set_transform", "params": {"name": "HouseBase", "scale": [3, 2.5, 1]}},
  {"op": "create_object", "params": {"kind": "cone", "name": "HouseRoof", "location": [0, 0, 3.2]}},
  {"op": "set_transform", "params": {"name": "HouseRoof", "scale": [2.6, 2.2, 1.2]}},
  {"op": "create_material", "params": {"name": "WallMat", "color": [0.85, 0.7, 0.5, 1.0], "roughness": 0.9, "assign_to": "HouseBase"}},
  {"op": "create_material", "params": {"name": "RoofMat", "color": [0.6, 0.15, 0.1, 1.0], "roughness": 0.8, "assign_to": "HouseRoof"}}
], "note": "Made a simple house: brick base with a cone roof."}

EXAMPLE 2 — "make a wooden table" (one leg shown; add 3 more the same way):
{"ops": [
  {"op": "create_object", "params": {"kind": "cube", "name": "TableTop", "location": [0, 0, 0.75]}},
  {"op": "set_transform", "params": {"name": "TableTop", "scale": [1.1, 0.7, 0.05]}},
  {"op": "create_object", "params": {"kind": "cylinder", "name": "TableLeg1", "location": [-0.9, -0.55, 0.35]}},
  {"op": "set_transform", "params": {"name": "TableLeg1", "scale": [0.05, 0.05, 0.35]}},
  {"op": "create_material", "params": {"name": "WoodMat", "color": [0.55, 0.35, 0.2, 1.0], "roughness": 0.7, "assign_to": "TableTop"}}
], "note": "Made a wooden table top with one leg; copy TableLeg1 three more times for the other corners."}
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
    for later steps (a plan that creates a cube then materials it is valid).
    v1.2.0: also rejects HOLLOW plans — an op with no usable params (e.g.
    create_object({})) is a plan error, not something to "apply"."""
    errors = []
    if not ops:
        return ["plan has no ops"]
    # Required params. need_any: at least one of the listed must be present
    # and non-empty. need_all: every listed param must be present.
    need_any = {"create_object": ("kind", "name", "location"),
                "create_material": ("name", "color", "assign_to")}
    need_all = {"set_transform": ("name",), "delete_object": ("name",),
                "add_modifier": ("name",), "get_object_info": ("name",),
                "execute_code": ("code",)}
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
        if op in need_any and not any(params.get(k) for k in need_any[op]):
            errors.append("step %d: %s has no usable params — give %s" %
                          (i, op, " or ".join(need_any[op])))
        for k in need_all.get(op, ()):
            if not params.get(k):
                errors.append("step %d: %s needs %r" % (i, op, k))
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


def _apply_summary(results, total):
    """Honest Apply report: (ok_count, fail_reason). _apply_plan stops at the
    first failed op and appends a 'stopped at failed op' marker, so the first
    non-ok result carrying a 'code' is the real failure (v1.2.0)."""
    ok = 0
    fail = ""
    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("status") == "ok":
            ok += 1
        elif "code" in r and not fail:
            fail = str(r.get("message") or r.get("code"))[:160]
    return ok, fail


# ---------------------------------------------------------------- LLM client (pure stdlib; WORKER THREAD ONLY)
def _chat_completions(snap, messages):
    url = (snap["base_url"] or "").rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": snap["model"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4000,
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


def _extract_json_block(content):
    """Pull the JSON object out of a model reply that may carry prose or
    markdown fences around it. Returns the candidate string, or "".
    v1.1.1: the old code only stripped a leading fence, so a reply like
    "Here is the plan: ```json {...} ``` Hope this helps!" died in
    json.loads with a bare "Expecting value" error (seen on the phone
    2026-11-13, second "make an simple house" send)."""
    text = (content or "").strip()
    if not text:
        return ""
    # Drop markdown fence lines (``` or ```json), line by line.
    lines = [ln for ln in text.split("\n") if ln.strip() not in ("```", "```json")]
    text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _parse_plan(content):
    candidate = _extract_json_block(content)
    try:
        plan = json.loads(candidate)
    except Exception as e:
        raise RuntimeError(
            "LLM plan was not JSON (%s). Model said (first 300 chars): %s"
            % (e, (content or "")[:300]))
    if not isinstance(plan, dict):
        raise RuntimeError("LLM plan was not a JSON object")
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
        ok_count, fail = _apply_summary(results, len(ops))
        ok = ok_count == len(ops)
        _history_add("APPLIED: " + plan_text)
        if p["note"]:
            _history_add("AI: " + p["note"])
        _history_add("Applied %d of %d ops%s." %
                     (ok_count, len(ops), (" — stopped: " + fail) if fail else ""))
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


def _quick(op, params, label):
    """One-tap quick action: run a bridge op directly (no LLM call) and
    return (ok, short message) for the operator to report + log."""
    r = _dispatch(op, params)
    if isinstance(r, dict) and r.get("status") == "ok":
        res = r.get("result") or {}
        detail = res.get("filepath") or res.get("name") or ""
        return True, "%s: ok%s" % (label, (" — " + str(detail)) if detail else "")
    msg = ""
    if isinstance(r, dict):
        msg = str(r.get("message") or r.get("code") or r)
    return False, "%s failed: %s" % (label, msg[:160])


class AICHAT_OT_quick_cube(Operator):
    bl_idname = "ai_chat.quick_cube"
    bl_label = "Add Cube"
    bl_description = "Add a cube at the origin (no AI call)"

    def execute(self, context):
        ok, msg = _quick("create_object", {"kind": "cube", "location": [0, 0, 1]}, "Add Cube")
        _history_add("Quick: " + msg)
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class AICHAT_OT_quick_light(Operator):
    bl_idname = "ai_chat.quick_light"
    bl_label = "Add Light"
    bl_description = "Add a point light above the scene (no AI call)"

    def execute(self, context):
        ok, msg = _quick("create_object", {"kind": "light", "location": [2, 2, 4]}, "Add Light")
        _history_add("Quick: " + msg)
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class AICHAT_OT_quick_render(Operator):
    bl_idname = "ai_chat.quick_render"
    bl_label = "Render"
    bl_description = "Render with the default engine (Workbench, 256x256) to Download/Blender (no AI call)"

    def execute(self, context):
        ok, msg = _quick("render", {}, "Render")
        _history_add("Quick: " + msg)
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class AICHAT_OT_quick_save(Operator):
    bl_idname = "ai_chat.quick_save"
    bl_label = "Save"
    bl_description = "Save the .blend file to the default location (no AI call)"

    def execute(self, context):
        ok, msg = _quick("save_blend", {}, "Save")
        _history_add("Quick: " + msg)
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


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
        ok_count, fail = _apply_summary(results, len(ops))
        ok = ok_count == len(ops)
        plan_text = " | ".join("%s(%s)" % (s.get("op"), json.dumps(s.get("params") or {})) for s in ops)
        _history_add("APPLIED (stored plan): " + plan_text)
        _history_add("Applied %d of %d ops%s." %
                     (ok_count, len(ops), (" — stopped: " + fail) if fail else ""))
        props.pending_plan_json = ""
        props.pending_note = ""
        self.report({"INFO" if ok else "WARNING"}, "Applied %d of %d ops" % (ok_count, len(ops)))
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
        row.scale_y = 2.0
        row.operator("ai_chat.send", text="Send to AI")
        if (props.pending_plan_json or "").strip():
            row2 = layout.row()
            row2.scale_y = 2.0
            row2.operator("ai_chat.apply_plan", text="Apply stored plan")
            for line in _wrap("Plan: " + props.pending_plan_json, 44):
                layout.label(text=line)
            if props.pending_note:
                for line in _wrap("AI: " + props.pending_note, 44):
                    layout.label(text=line)
        layout.separator()
        layout.label(text="Quick actions (no AI call):")
        qrow1 = layout.row(align=True)
        qrow1.scale_y = 1.6
        qrow1.operator("ai_chat.quick_cube", text="Add Cube")
        qrow1.operator("ai_chat.quick_light", text="Add Light")
        qrow2 = layout.row(align=True)
        qrow2.scale_y = 1.6
        qrow2.operator("ai_chat.quick_render", text="Render")
        qrow2.operator("ai_chat.quick_save", text="Save")
        layout.separator()
        # History in a box: Blender's Python UI has no per-label font size,
        # so grouping + spacing is the real readability win available here.
        layout.label(text="History (latest last):")
        box = layout.box()
        for raw in (props.history or "").split("\n")[-8:]:
            for line in _wrap(raw, 44):
                if line.strip():
                    box.label(text=line)


_classes = (AIChatPreferences, AIChatProperties, AICHAT_OT_send,
            AICHAT_OT_apply, AICHAT_OT_quick_cube, AICHAT_OT_quick_light,
            AICHAT_OT_quick_render, AICHAT_OT_quick_save, AICHAT_PT_panel)


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
