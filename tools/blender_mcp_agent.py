#!/usr/bin/env python3
# blender_mcp_agent.py — a REAL MCP agent loop for Blender on Android (v1.0.0).
# Runs in the phone's own Linux environment (Termux) and drives Blender the
# way the Blender MCP demos do: plan -> act -> observe -> fix -> done.
#
# The user types a prompt ("make a low-poly city"); the agent asks an LLM
# what to do, executes the MCP tools, feeds the results back, and loops until
# the scene is built. This is NOT the panel's one-shot plan: the agent sees
# every tool result and can correct its own mistakes.
#
# Architecture: this file is an MCP CLIENT. It spawns
# blender_android_mcp_lite.py as a subprocess (stdio JSON-RPC), does the
# handshake, and calls its tools. It NEVER talks to the bridge socket
# directly and NEVER reimplements bridge ops.
#
# LLM: plain urllib POST to an OpenAI-compatible /chat/completions endpoint.
# The model is asked for ONE JSON block per step (NOT native tool_calls —
# free models are unreliable at native function calling):
#   {"calls": [{"tool": "<name>", "params": {...}}, ...],
#    "done": false, "note": "<one short line>"}
#
# Config (environment only — tokens/keys never on the command line):
#   BLENDER_BRIDGE_TOKEN  REQUIRED (same token as the AI Bridge addon panel)
#   BLENDER_AGENT_API_KEY LLM key; GROQ_API_KEY is accepted as an alias
#   BLENDER_AGENT_BASE_URL  default https://api.groq.com/openai/v1
#   BLENDER_AGENT_MODEL    default llama-3.3-70b-versatile
#   BLENDER_AGENT_TIMEOUT_S  default 120 (LLM HTTP timeout)
#   BLENDER_AGENT_MAX_STEPS  default 25
#   BLENDER_ALLOW_CODE     0 (default) or 1 — when 1, the execute_code tool
#                          is listed and the agent may use it (warning printed)
#
# Run (Termux — Blender open, AI Bridge Status: running):
#   export GROQ_API_KEY=<key from console.groq.com — free, email only, no card>
#   export BLENDER_BRIDGE_TOKEN=<token from Blender's AI Bridge panel>
#   python3 blender_mcp_agent.py "make a low-poly city"
#
# Exit codes: 0 = done (LLM said done), 1 = stopped with a failure
# (LLM error, max steps, same call failing 3 times, unparseable replies),
# 2 = missing config (no API key / no bridge token).
#
# Testing: --mock runs a canned scripted LLM through the SAME code path
# (same parse, same hollow-check, same MCP calls) — no key needed, but the
# bridge must be up. See test_agent.py.
#
# Security: the API key and bridge token are NEVER printed or logged. The
# agent writes no files; all state is in memory + stdout.

import json
import os
import select
import subprocess
import sys
import urllib.error
import urllib.request

VERSION = "1.0.0"
NAME = "blender-mcp-agent"

HERE = os.path.dirname(os.path.abspath(__file__))
LITE_SERVER = os.path.join(HERE, "blender_android_mcp_lite.py")

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_STEPS = 25
MAX_CALLS_PER_STEP = 5
MAX_SAME_FAILURES = 3

# ---------------------------------------------------------------- errors
class AgentError(Exception):
    """Fatal agent problem (config, MCP transport, LLM HTTP)."""


class LLMError(AgentError):
    """The LLM endpoint answered with an error (429/503/...)."""


# ---------------------------------------------------------------- JSON block
def extract_json_block(text):
    """Robust JSON-object extraction, same idea as the panel's
    _extract_json_block: drop code-fence lines, take the first '{' to the
    last '}'. Raises ValueError with a short explanation on failure."""
    if not isinstance(text, str):
        raise ValueError("LLM reply was not text")
    lines = [ln for ln in text.splitlines()
             if ln.strip() not in ("```", "```json", "```JSON")]
    cleaned = "\n".join(lines)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the LLM reply")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError("LLM JSON did not parse: %s" % e)


# ---------------------------------------------------------------- hollow calls
def hollow_reason(tool, params):
    """Return a short reason when a call carries no usable params, else
    None. Same rule as the panel's hollow-plan rejection (v1.2.0): an op
    that cannot do anything must never reach Blender."""
    if not isinstance(params, dict):
        return "params must be an object"
    if tool == "create_object":
        if not (params.get("kind") or params.get("name")
                or params.get("location")):
            return ("create_object needs at least one of kind/name/location "
                    "(got empty params)")
    elif tool in ("set_transform", "delete_object", "add_modifier",
                  "get_object_info"):
        if not params.get("name"):
            return "%s needs 'name' (got %s)" % (tool, params)
    elif tool == "create_material":
        if not (params.get("name") or params.get("color")
                or params.get("assign_to")):
            return ("create_material needs name/color/assign_to "
                    "(got %s)" % params)
    elif tool == "execute_code":
        if not params.get("code"):
            return "execute_code needs non-empty 'code'"
    elif tool == "batch":
        if not params.get("ops"):
            return "batch needs a non-empty 'ops' list"
    return None


# ---------------------------------------------------------------- system prompt
_OPS_REFERENCE = """OPS (exact params — defaults in brackets):
- get_scene_info: {} — lists every object (name, type, location) + scene totals. Use it first when unsure about names.
- get_object_info: name (required).
- create_object: kind cube/sphere/cylinder/cone/torus/plane/monkey/light/camera [cube], name, location [x,y,z] [0,0,0]. A cone's point is up; a cylinder's axis is Z.
- set_transform: name (required), location [x,y,z], rotation_euler [rx,ry,rz] in radians, scale [x,y,z]. Only the fields you give change.
- delete_object: name (required).
- create_material: name, color [r,g,b,a] each 0..1, roughness 0..1, metallic 0..1, assign_to (object name). ALWAYS set assign_to or the material sits unused.
- add_modifier: name (required), modifier_type SUBSURF/BEVEL, levels (max 2), width (bevel, metres).
- render: filepath (omit for the default), width/height (max 512 each), samples (max 32), engine WORKBENCH/EEVEE/EEVEE_NEXT/CYCLES. Default WORKBENCH — always use it on this phone. Cycles is NOT in this Lite build.
- save_blend: filepath (omit for the default).
- batch: ops (list of {"command": ..., "params": {...}}) — runs one op per Blender timer tick; use for many small ops at once.
{CODE_LINE}

RULES — every call MUST have complete, concrete params. Never emit "params": {}.
- Names: use the exact names from the scene summary. Never invent an object name that does not exist; call get_scene_info first when unsure. When you create objects, give each a clear name and reuse THAT name in later steps.
- Space: metres, Z is up, rotation_euler is in RADIANS, scale is a multiplier. A default cube is 2m on a side, centred at its location. "size" is NOT a param — size comes from set_transform scale.
- Build in order: create parts first, then materials, then transforms. Keep each step small: at most 5 calls per step, one object per call. Prefer a few big simple parts over many small ones.
- If something cannot be done with these ops, set "done": true and explain in "note" — never invent a tool, a param, or an object name.

PHONE LIMITS — hard caps, never exceed them or the op is refused:
- Scene totals: 200,000 polygons, 300,000 vertices. SUBSURF levels max 2. Render max 512x512, max 32 samples. One batch: max 50 ops. execute_code: max 100,000 chars, scene auto-saved first.
- Files: never use /tmp — on Android it fails. Omit filepath and the app uses Download/Blender.

EXAMPLE — "make a simple house" (copy this shape, one step at a time):
{"calls": [
  {"tool": "create_object", "params": {"kind": "cube", "name": "HouseBase", "location": [0, 0, 1]}},
  {"tool": "set_transform", "params": {"name": "HouseBase", "scale": [3, 2.5, 1]}},
  {"tool": "create_object", "params": {"kind": "cone", "name": "HouseRoof", "location": [0, 0, 3.2]}},
  {"tool": "set_transform", "params": {"name": "HouseRoof", "scale": [2.6, 2.2, 1.2]}},
  {"tool": "create_material", "params": {"name": "WallMat", "color": [0.85, 0.7, 0.5, 1.0], "roughness": 0.9, "assign_to": "HouseBase"}},
  {"tool": "create_material", "params": {"name": "RoofMat", "color": [0.6, 0.15, 0.1, 1.0], "roughness": 0.8, "assign_to": "HouseRoof"}}
], "done": false, "note": "House base and roof created with materials."}
"""


def system_prompt(allow_code):
    code_line = ("- execute_code: code (Python with bpy) — ONLY when no structured "
                 "tool can do it. The scene is auto-saved before it runs."
                 if allow_code else
                 "- execute_code: NOT available. Never emit it; use the structured tools only.")
    return ("You are the AI driving \"Blender AI\" — Blender 5.3.0 on an Android "
            "phone — through MCP tools. The user gives you one task; you build "
            "it step by step, checking each tool result before the next step. "
            "Reply with ONLY one JSON object per step, no markdown, no fences, "
            "no extra text:\n"
            "{\"calls\": [{\"tool\": \"<name>\", \"params\": {...}}, ...], "
            "\"done\": false, \"note\": \"<one short line>\"}\n"
            "Set \"done\": true (with empty calls) only when the task is fully "
            "built. If a tool result says an object does not exist or a call "
            "failed, fix the next step instead of repeating the same failing "
            "call.\n\n" + _OPS_REFERENCE.replace("{CODE_LINE}", code_line))


# ---------------------------------------------------------------- MCP client
class MCPClient:
    """Minimal JSON-RPC 2.0 stdio client for blender_android_mcp_lite.py."""

    def __init__(self, token, allow_code):
        env = dict(os.environ)
        env["BLENDER_BRIDGE_TOKEN"] = token
        env["BLENDER_ALLOW_CODE"] = "1" if allow_code else "0"
        self.proc = subprocess.Popen(
            [sys.executable, LITE_SERVER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self._next_id = 0
        self.tools = []

    def _readline(self, timeout):
        r, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not r:
            raise AgentError("MCP server did not answer within %ss" % timeout)
        line = self.proc.stdout.readline()
        if not line:
            err = ""
            try:
                err = self.proc.stderr.read()[:500]
            except Exception:
                pass
            raise AgentError("MCP server closed stdout (stderr: %s)" % err)
        return line

    def request(self, method, params=None, timeout=60):
        self._next_id += 1
        rid = self._next_id
        self.proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": rid, "method": method,
             "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self._readline(timeout))
        if resp.get("id") != rid:
            raise AgentError("MCP reply id mismatch: %s" % resp)
        if "error" in resp:
            raise AgentError("MCP error %s: %s"
                             % (resp["error"].get("code"),
                                resp["error"].get("message")))
        return resp.get("result", {})

    def notify(self, method, params=None):
        self.proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "method": method,
             "params": params or {}}) + "\n")
        self.proc.stdin.flush()

    def handshake(self):
        res = self.request("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": NAME, "version": VERSION}})
        self.notify("notifications/initialized")
        tools = self.request("tools/list")
        self.tools = [t["name"] for t in tools.get("tools", [])]
        return res.get("serverInfo", {})

    def call_tool(self, name, args):
        """Returns (ok, result_text). ok=False means the tool reported an
        error (isError) or the bridge returned status:error."""
        res = self.request("tools/call",
                           {"name": name, "arguments": args or {}},
                           timeout=200)
        parts = res.get("content", [])
        text = parts[0].get("text", "") if parts else ""
        if res.get("isError"):
            return False, text
        try:
            data = json.loads(text) if text.startswith("{") else {"text": text}
        except json.JSONDecodeError:
            return True, text
        if isinstance(data, dict) and data.get("status") == "error":
            return False, text
        return True, text

    def stop(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ---------------------------------------------------------------- LLM client
def llm_chat(base_url, model, api_key, messages, timeout_s):
    """One POST to an OpenAI-compatible /chat/completions endpoint.
    Returns the assistant's reply text. Raises LLMError on any HTTP or
    transport failure (the agent stops — it never retries forever)."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4000,
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:300]
        except Exception:
            pass
        raise LLMError("LLM HTTP %s: %s" % (e.code, detail or e.reason))
    except urllib.error.URLError as e:
        raise LLMError("LLM request failed: %s" % e.reason)
    except (TimeoutError, OSError) as e:
        raise LLMError("LLM request failed: %s" % e)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("LLM reply had no usable content: %s"
                       % str(data)[:300])


# ---------------------------------------------------------------- mock LLM
# Canned scripted replies for --mock host testing. The mock returns plain
# text (sometimes garbage) and goes through the SAME parse / hollow-check /
# MCP-call path as the real LLM.
def mock_reply(scenario, step):
    if scenario == "bad_tool":
        return json.dumps({"calls": [{"tool": "fly_to_moon", "params": {}}],
                           "done": False, "note": "Trying a tool that does not exist."})
    if scenario == "garbage":
        return "this is not json {{{ garbage {{{"
    if scenario == "hollow":
        if step == 0:
            return json.dumps({"calls": [{"tool": "create_object", "params": {}}],
                               "done": False, "note": "Empty params (bad)."})
        if step == 1:
            return json.dumps({"calls": [
                {"tool": "create_object",
                 "params": {"kind": "cube", "name": "HollowHouse",
                            "location": [1, 0, 0.5]}},
                {"tool": "set_transform",
                 "params": {"name": "HollowHouse", "scale": [1.5, 1.5, 1]}},
            ], "done": False, "note": "House with filled params."})
        if step == 2:
            return json.dumps({"calls": [{"tool": "get_scene_info", "params": {}}],
                               "done": False, "note": "Checking the scene."})
        return json.dumps({"calls": [], "done": True, "note": "Hollow house built."})
    if scenario == "never_done":
        return json.dumps({"calls": [{"tool": "get_scene_info", "params": {}}],
                           "done": False, "note": "Still going."})
    # default scenario
    if step == 0:
        return json.dumps({"calls": [
            {"tool": "create_object",
             "params": {"kind": "cube", "name": "MockHouse",
                        "location": [0, 0, 1]}},
            {"tool": "set_transform",
             "params": {"name": "MockHouse", "scale": [2, 2, 1]}},
        ], "done": False, "note": "Building the mock house."})
    if step == 1:
        return json.dumps({"calls": [{"tool": "get_scene_info", "params": {}}],
                           "done": False, "note": "Checking the scene."})
    return json.dumps({"calls": [], "done": True, "note": "Mock house built."})


# ---------------------------------------------------------------- agent loop
def run_agent(prompt, base_url, model, api_key, max_steps, timeout_s,
              bridge_token, allow_code, mock_scenario=None):
    """Returns (exit_code, summary_dict). Prints progress to stdout."""
    print("blender-mcp-agent v%s — model %s via %s (max_steps=%d, allow_code=%s)"
          % (VERSION, model, base_url, max_steps, allow_code), flush=True)
    if allow_code:
        print("WARNING: execute_code is ON — the LLM may run arbitrary Python "
              "in Blender (the bridge auto-saves first).", flush=True)
    client = MCPClient(bridge_token, allow_code)
    try:
        info = client.handshake()
        print("MCP server: %s v%s, tools: %s"
              % (info.get("name"), info.get("version"),
                 ", ".join(client.tools)), flush=True)
    except AgentError as e:
        print("AGENT STOPPED: MCP handshake failed: %s" % e, flush=True)
        client.stop()
        return 1, {"error": "mcp handshake failed: %s" % e}

    # The agent reads the scene itself first, so the LLM never has to guess
    # which objects exist.
    scene_note = ""
    try:
        ok, text = client.call_tool("get_scene_info", {})
        if ok:
            data = json.loads(text)
            objs = data.get("result", {}).get("objects", [])
            names = [o.get("name") for o in objs]
            scene_note = ("Current scene: %d objects: %s. "
                          % (len(names), ", ".join(names) if names else "none"))
            print("scene: %s" % scene_note, flush=True)
    except AgentError as e:
        print("AGENT STOPPED: could not read the scene: %s" % e, flush=True)
        client.stop()
        return 1, {"error": "scene read failed: %s" % e}

    messages = [
        {"role": "system", "content": system_prompt(allow_code)},
        {"role": "user", "content": scene_note + "Task: " + prompt},
    ]
    created = []
    failures = []
    fail_counts = {}
    consecutive_empty = 0
    malformed = 0
    done = False
    stop_reason = None

    for step in range(max_steps):
        if mock_scenario is not None:
            reply = mock_reply(mock_scenario, step)
        else:
            try:
                reply = llm_chat(base_url, model, api_key, messages, timeout_s)
            except LLMError as e:
                print("AGENT STOPPED: %s" % e, flush=True)
                print("Tip: free-model 429/503s are provider capacity, not the "
                      "agent — wait a minute and re-run, or switch model.",
                      flush=True)
                client.stop()
                return 1, {"error": str(e), "created": created,
                           "failures": failures}
        messages.append({"role": "assistant", "content": reply})
        try:
            plan = extract_json_block(reply)
        except ValueError as e:
            malformed += 1
            print("step %d: LLM reply was not valid JSON (%s)"
                  % (step + 1, e), flush=True)
            if malformed >= 2:
                stop_reason = ("LLM replied with unparseable JSON twice — "
                               "stopping instead of looping")
                break
            messages.append({"role": "user", "content":
                             "Your last reply was not valid JSON (%s). Reply "
                             "with ONLY the JSON object "
                             "{\"calls\": [...], \"done\": false, "
                             "\"note\": \"...\"}." % e})
            continue
        if not isinstance(plan, dict) or not isinstance(plan.get("calls"), list) \
                or not isinstance(plan.get("done"), bool):
            malformed += 1
            print("step %d: LLM reply had the wrong shape (need {\"calls\": [], "
                  "\"done\": bool})" % (step + 1), flush=True)
            if malformed >= 2:
                stop_reason = ("LLM reply had the wrong shape twice — "
                               "stopping instead of looping")
                break
            messages.append({"role": "user", "content":
                             "Your reply must be exactly {\"calls\": "
                             "[{\"tool\": ..., \"params\": {...}}], "
                             "\"done\": false, \"note\": \"...\"}. "
                             "Try again."})
            continue
        calls = plan["calls"]
        note = plan.get("note", "")
        if not calls and not plan["done"]:
            consecutive_empty += 1
            print("step %d: LLM sent no calls and did not finish (%s)"
                  % (step + 1, note), flush=True)
            if consecutive_empty >= 2:
                stop_reason = ("LLM sent no calls twice in a row without "
                               "finishing — stopping")
                break
            messages.append({"role": "user", "content":
                             "You sent no calls and did not finish. Either "
                             "call tools or set \"done\": true."})
            continue
        consecutive_empty = 0
        if len(calls) > MAX_CALLS_PER_STEP:
            print("step %d: %d calls given, running the first %d only"
                  % (step + 1, len(calls), MAX_CALLS_PER_STEP), flush=True)
            calls = calls[:MAX_CALLS_PER_STEP]
        results = []
        for call in calls:
            tool = call.get("tool") if isinstance(call, dict) else None
            params = call.get("params") if isinstance(call, dict) else None
            label = "%s(%s)" % (tool, json.dumps(params or {}))
            if not isinstance(tool, str) or not isinstance(params, dict):
                results.append({"tool": str(tool),
                                "result": "error: each call must be "
                                "{\"tool\": \"<name>\", \"params\": {...}}"})
                print("  step %d: %s -> error (bad call shape)"
                      % (step + 1, label), flush=True)
                continue
            reason = hollow_reason(tool, params)
            if reason is not None:
                results.append({"tool": tool,
                                "result": "error: hollow call rejected: %s. "
                                "Re-issue this call with filled params."
                                % reason})
                print("  step %d: %s -> rejected (hollow: %s)"
                      % (step + 1, label, reason), flush=True)
                continue
            if tool not in client.tools:
                key = (tool, json.dumps(params, sort_keys=True))
                fail_counts[key] = fail_counts.get(key, 0) + 1
                failures.append("%s: unknown tool" % tool)
                results.append({"tool": tool,
                                "result": "error: unknown tool '%s'. "
                                "Available tools: %s"
                                % (tool, ", ".join(client.tools))})
                print("  step %d: %s -> error (unknown tool)"
                      % (step + 1, label), flush=True)
                if fail_counts[key] >= MAX_SAME_FAILURES:
                    stop_reason = ("the same call failed %d times "
                                   "(%s) — stopping instead of looping"
                                   % (MAX_SAME_FAILURES, label))
                    break
                continue
            try:
                ok, text = client.call_tool(tool, params)
            except AgentError as e:
                print("AGENT STOPPED: MCP transport failed: %s" % e, flush=True)
                client.stop()
                return 1, {"error": "mcp transport failed: %s" % e,
                           "created": created, "failures": failures}
            short = text[:1200]
            results.append({"tool": tool, "result": short})
            if ok:
                print("  step %d: %s -> ok" % (step + 1, label), flush=True)
                if tool == "create_object":
                    try:
                        name = json.loads(text).get("result", {}).get("name")
                    except (ValueError, AttributeError):
                        name = None
                    if name:
                        created.append(name)
            else:
                key = (tool, json.dumps(params, sort_keys=True))
                fail_counts[key] = fail_counts.get(key, 0) + 1
                failures.append("%s: %s" % (label, short[:200]))
                print("  step %d: %s -> error" % (step + 1, label), flush=True)
                if fail_counts[key] >= MAX_SAME_FAILURES:
                    stop_reason = ("the same call failed %d times "
                                   "(%s) — stopping instead of looping"
                                   % (MAX_SAME_FAILURES, label))
                    break
        messages.append({"role": "user", "content":
                         "Tool results:\n" + json.dumps(results)})
        if stop_reason is not None:
            break
        if plan["done"]:
            done = True
            print("step %d: done — %s" % (step + 1, note), flush=True)
            break
    client.stop()

    if done:
        print("AGENT DONE: %s" % (note or "task complete"), flush=True)
        print("created: %s" % (", ".join(created) if created else "none"),
              flush=True)
        if failures:
            print("failures along the way: %s" % "; ".join(failures), flush=True)
        return 0, {"created": created, "failures": failures}
    reason = stop_reason or ("stopped at max steps (%d) without finishing"
                             % max_steps)
    print("AGENT STOPPED: %s" % reason, flush=True)
    print("created so far: %s" % (", ".join(created) if created else "none"),
          flush=True)
    if failures:
        print("failures: %s" % "; ".join(failures), flush=True)
    print("re-run with: python3 blender_mcp_agent.py %r" % prompt, flush=True)
    return 1, {"error": reason, "created": created, "failures": failures}


# ---------------------------------------------------------------- CLI
def _parse_args(argv):
    prompt = None
    mock = False
    mock_scenario = "default"
    max_steps = None
    model = None
    base_url = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--mock":
            mock = True
        elif arg == "--mock-scenario" and i + 1 < len(argv):
            i += 1
            mock_scenario = argv[i]
        elif arg == "--max-steps" and i + 1 < len(argv):
            i += 1
            max_steps = int(argv[i])
        elif arg == "--model" and i + 1 < len(argv):
            i += 1
            model = argv[i]
        elif arg == "--base-url" and i + 1 < len(argv):
            i += 1
            base_url = argv[i]
        elif arg.startswith("--"):
            print("unknown flag: %s" % arg, file=sys.stderr)
            return None
        elif prompt is None:
            prompt = arg
        else:
            prompt += " " + arg
        i += 1
    return {"prompt": prompt, "mock": mock, "mock_scenario": mock_scenario,
            "max_steps": max_steps, "model": model, "base_url": base_url}


def main(argv=None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args is None:
        return 2
    prompt = args["prompt"]
    if prompt is None:
        if sys.stdin.isatty():
            print("usage: python3 blender_mcp_agent.py \"<prompt>\" [--mock] "
                  "[--max-steps N] [--model NAME] [--base-url URL]",
                  file=sys.stderr)
            return 2
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("no prompt given", file=sys.stderr)
        return 2

    bridge_token = os.environ.get("BLENDER_BRIDGE_TOKEN", "").strip()
    if not bridge_token:
        print("BLENDER_BRIDGE_TOKEN is not set — refusing to start",
              file=sys.stderr)
        print("Set it to the same token shown in Blender's AI Bridge panel.",
              file=sys.stderr)
        return 2
    allow_code = os.environ.get("BLENDER_ALLOW_CODE", "0").strip() == "1"

    if args["mock"]:
        api_key = None
    else:
        api_key = (os.environ.get("BLENDER_AGENT_API_KEY", "").strip()
                   or os.environ.get("GROQ_API_KEY", "").strip())
        if not api_key:
            print("no LLM API key — set GROQ_API_KEY (or BLENDER_AGENT_API_KEY)",
                  file=sys.stderr)
            print("Get a free key at console.groq.com (email only, no card).",
                  file=sys.stderr)
            return 2
    base_url = (args["base_url"]
                or os.environ.get("BLENDER_AGENT_BASE_URL", "").strip()
                or DEFAULT_BASE_URL)
    model = (args["model"]
             or os.environ.get("BLENDER_AGENT_MODEL", "").strip()
             or DEFAULT_MODEL)
    try:
        timeout_s = int(os.environ.get("BLENDER_AGENT_TIMEOUT_S",
                                       str(DEFAULT_TIMEOUT_S)))
    except ValueError:
        timeout_s = DEFAULT_TIMEOUT_S
    try:
        max_steps = (args["max_steps"] if args["max_steps"] is not None
                     else int(os.environ.get("BLENDER_AGENT_MAX_STEPS",
                                            str(DEFAULT_MAX_STEPS))))
    except ValueError:
        max_steps = DEFAULT_MAX_STEPS

    code, _summary = run_agent(
        prompt, base_url, model, api_key, max_steps, timeout_s,
        bridge_token, allow_code,
        mock_scenario=args["mock_scenario"] if args["mock"] else None)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
