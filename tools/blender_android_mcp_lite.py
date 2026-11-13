#!/usr/bin/env python3
# blender_android_mcp_lite.py — stdlib-only MCP server for Blender on Android
# (v1.0.0, "blender-android-lite"). Drop-in replacement for
# blender_android_mcp.py (official MCP SDK v2) on phones where pip cannot
# install the SDK: Termux/Android has no prebuilt wheel for rpds-py (a Rust
# extension the SDK pulls in), so `pip install "mcp[cli]"` fails there.
# This file uses NOTHING but the Python standard library.
#
# Runs in the phone's own Linux environment (Termux) and forwards MCP tool
# calls to the AI Bridge inside Blender over 127.0.0.1:<port> + token.
# Transport: JSON-RPC 2.0 over stdio, newline-delimited JSON (same framing as
# the SDK server). No Blender, no bpy here — this process only speaks TCP to
# the bridge. Diagnostics go to stderr only; stdout is the MCP transport.
#
# Config (environment, never on the command line — tokens must not leak into
# process lists or shell history):
#   BLENDER_BRIDGE_HOST   default 127.0.0.1
#   BLENDER_BRIDGE_PORT   default 9876
#   BLENDER_BRIDGE_TOKEN  REQUIRED (same token as the AI Bridge addon)
#   BLENDER_ALLOW_CODE    0 (default) or 1 — when 1, the execute_code tool is
#                         listed; when 0 it does not exist in tools/list.
#
# Run (Termux, no pip install needed):
#   BLENDER_BRIDGE_TOKEN=<token from Blender's AI Bridge panel> \
#     python3 blender_android_mcp_lite.py
# Check the bridge without starting MCP:
#   BLENDER_BRIDGE_TOKEN=<token> python3 blender_android_mcp_lite.py --ping
#
# Protocol coverage: initialize, ping, tools/list, tools/call, plus empty
# resources/list, prompts/list, resourceTemplates/list. No sampling,
# elicitation, roots, or logging support — tools only (same scope the SDK
# server was used for). Unknown method -> JSON-RPC -32601. Malformed JSON
# line -> -32700 (the SDK server silently drops such lines; this server
# answers so clients never hang). Bad tool params -> a bridge-style error
# dict as the tool result, isError true; the server never crashes on a
# bad request.
#
# Security: the token is sent only to 127.0.0.1. It is never printed or
# logged anywhere.

import json
import os
import socket
import sys

VERSION = "1.0.0"
NAME = "blender-android-lite"

INSTRUCTIONS = (
    "Control Blender running on this phone through its local AI Bridge. "
    "Tools mirror the bridge commands. Read get_scene_info first when unsure "
    "about object names. Renders default to the Workbench engine and are capped "
    "at 512px / 32 samples by the bridge. execute_code is only available when "
    "the server was started with BLENDER_ALLOW_CODE=1."
)

HOST = os.environ.get("BLENDER_BRIDGE_HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = int(os.environ.get("BLENDER_BRIDGE_PORT", "9876"))
TOKEN = os.environ.get("BLENDER_BRIDGE_TOKEN", "").strip()
ALLOW_CODE = os.environ.get("BLENDER_ALLOW_CODE", "0").strip() == "1"

# MCP protocol versions this server understands. The client asks for one in
# initialize; echo it back when known, else answer with the default (the
# version the SDK server negotiated in the M1 tests).
KNOWN_PROTOCOL_VERSIONS = {
    "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05", "2024-10-07",
}
DEFAULT_PROTOCOL_VERSION = "2025-11-25"

# Tool definitions, byte-identical in content to the SDK server's tools/list
# output (captured live from blender_android_mcp.py, mcp 2.3.0): same names,
# descriptions and inputSchemas. execute_code is appended only when
# BLENDER_ALLOW_CODE=1.
_TOOLS_JSON = r"""[
  {
    "description": "Scene summary: object list with locations, vertex/face totals, engine, version.",
    "inputSchema": {
      "properties": {},
      "type": "object",
      "title": "get_scene_infoArguments"
    },
    "name": "get_scene_info"
  },
  {
    "description": "Details of one object: type, transform, dimensions, materials, mesh counts.",
    "inputSchema": {
      "properties": {
        "name": {
          "title": "Name",
          "type": "string"
        }
      },
      "required": [
        "name"
      ],
      "type": "object",
      "title": "get_object_infoArguments"
    },
    "name": "get_object_info"
  },
  {
    "description": "Create a primitive: cube, sphere, cylinder, cone, torus, plane, monkey,\n    light, camera. location is [x, y, z] in metres (Z up).",
    "inputSchema": {
      "properties": {
        "kind": {
          "title": "Kind",
          "type": "string"
        },
        "name": {
          "default": "",
          "title": "Name",
          "type": "string"
        },
        "location": {
          "anyOf": [
            {
              "items": {
                "type": "number"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Location"
        }
      },
      "required": [
        "kind"
      ],
      "type": "object",
      "title": "create_objectArguments"
    },
    "name": "create_object"
  },
  {
    "description": "Delete one object by name.",
    "inputSchema": {
      "properties": {
        "name": {
          "title": "Name",
          "type": "string"
        }
      },
      "required": [
        "name"
      ],
      "type": "object",
      "title": "delete_objectArguments"
    },
    "name": "delete_object"
  },
  {
    "description": "Move/rotate/scale an object. Rotation is Euler radians [rx, ry, rz].",
    "inputSchema": {
      "properties": {
        "name": {
          "title": "Name",
          "type": "string"
        },
        "location": {
          "anyOf": [
            {
              "items": {
                "type": "number"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Location"
        },
        "rotation_euler": {
          "anyOf": [
            {
              "items": {
                "type": "number"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Rotation Euler"
        },
        "scale": {
          "anyOf": [
            {
              "items": {
                "type": "number"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scale"
        }
      },
      "required": [
        "name"
      ],
      "type": "object",
      "title": "set_transformArguments"
    },
    "name": "set_transform"
  },
  {
    "description": "Create (or reuse) a Principled BSDF material. color is [r, g, b, a] 0..1.\n    assign_to attaches it to that object.",
    "inputSchema": {
      "properties": {
        "name": {
          "title": "Name",
          "type": "string"
        },
        "color": {
          "anyOf": [
            {
              "items": {
                "type": "number"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Color"
        },
        "roughness": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Roughness"
        },
        "metallic": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Metallic"
        },
        "assign_to": {
          "default": "",
          "title": "Assign To",
          "type": "string"
        }
      },
      "required": [
        "name"
      ],
      "type": "object",
      "title": "create_materialArguments"
    },
    "name": "create_material"
  },
  {
    "description": "Add a modifier (e.g. SUBSURF with levels <= 2, BEVEL with width).",
    "inputSchema": {
      "properties": {
        "name": {
          "title": "Name",
          "type": "string"
        },
        "modifier_type": {
          "title": "Modifier Type",
          "type": "string"
        },
        "levels": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Levels"
        },
        "width": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Width"
        }
      },
      "required": [
        "name",
        "modifier_type"
      ],
      "type": "object",
      "title": "add_modifierArguments"
    },
    "name": "add_modifier"
  },
  {
    "description": "Render the current view to a file (max 512px per side, max 32 samples).\n    engine: WORKBENCH (default), EEVEE, EEVEE_NEXT, CYCLES. On the phone the\n    file lands in Download/Blender when filepath is omitted.",
    "inputSchema": {
      "properties": {
        "filepath": {
          "default": "",
          "title": "Filepath",
          "type": "string"
        },
        "width": {
          "default": 256,
          "title": "Width",
          "type": "integer"
        },
        "height": {
          "default": 256,
          "title": "Height",
          "type": "integer"
        },
        "samples": {
          "default": 16,
          "title": "Samples",
          "type": "integer"
        },
        "engine": {
          "default": "",
          "title": "Engine",
          "type": "string"
        }
      },
      "type": "object",
      "title": "renderArguments"
    },
    "name": "render"
  },
  {
    "description": "Save the .blend file (Download/Blender when filepath is omitted).",
    "inputSchema": {
      "properties": {
        "filepath": {
          "default": "",
          "title": "Filepath",
          "type": "string"
        }
      },
      "type": "object",
      "title": "save_blendArguments"
    },
    "name": "save_blend"
  },
  {
    "description": "Queue many bridge ops to run one per Blender timer tick (keeps the phone\n    UI responsive). Each op: {\"command\": ..., \"params\": {...}}. Use the\n    batch_status tool result pattern via get_scene_info to confirm effects;\n    the bridge also exposes batch_status over the socket directly.",
    "inputSchema": {
      "properties": {
        "ops": {
          "items": {
            "additionalProperties": true,
            "type": "object"
          },
          "title": "Ops",
          "type": "array"
        }
      },
      "required": [
        "ops"
      ],
      "type": "object",
      "title": "batchArguments"
    },
    "name": "batch"
  }
]"""
_EXECUTE_CODE_JSON = r"""{
  "description": "Run arbitrary Python inside Blender (bpy available). Only registered\n        when the server started with BLENDER_ALLOW_CODE=1. The bridge auto-saves\n        a copy of the scene first.",
  "inputSchema": {
    "properties": {
      "code": {
        "title": "Code",
        "type": "string"
      }
    },
    "required": [
      "code"
    ],
    "type": "object",
    "title": "execute_codeArguments"
  },
  "name": "execute_code"
}"""
TOOLS = json.loads(_TOOLS_JSON)
EXECUTE_CODE_TOOL = json.loads(_EXECUTE_CODE_JSON)


def _bridge(command, params=None, timeout=180):
    """One bridge call: new TCP connection, one JSON line, one response line.
    Returns the bridge response dict. Transport problems become error dicts
    (never exceptions) so MCP clients always get a readable result."""
    if not TOKEN:
        return {"status": "error", "code": "no_token",
                "message": "BLENDER_BRIDGE_TOKEN is not set"}
    msg = json.dumps({"token": TOKEN, "command": command,
                      "params": params or {}}) + "\n"
    try:
        with socket.create_connection((HOST, PORT), timeout=timeout) as s:
            s.sendall(msg.encode())
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    return {"status": "error", "code": "bridge_closed",
                            "message": "Blender bridge closed the connection"}
                buf += chunk
            return json.loads(buf.decode())
    except socket.timeout:
        return {"status": "error", "code": "bridge_timeout",
                "message": "Blender bridge did not answer in %ds" % timeout}
    except ConnectionRefusedError:
        return {"status": "error", "code": "bridge_unreachable",
                "message": ("cannot reach Blender bridge at %s:%d — is Blender running "
                            "with the AI Bridge addon enabled and the server started?"
                            % (HOST, PORT))}
    except Exception as e:  # noqa: BLE001 - transport errors become dicts
        return {"status": "error", "code": "bridge_error",
                "message": "bridge call failed: %s" % e}


def _clean(resp):
    """Bridge responses already carry status/result/error; pass through."""
    if isinstance(resp, dict):
        return resp
    return {"status": "ok", "result": resp}


# ---------------------------------------------------------------- validation
# The SDK server validates tool arguments with pydantic before calling the
# bridge. This server does the same checks by hand: required arguments must
# be present, and values must have the declared JSON type. A failure returns
# a bridge-style error dict (never an exception, never a crash).

def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _check_args(tool, args):
    """Return (params_dict, error_dict). error_dict is None on success.
    params_dict mirrors exactly what the SDK tool functions send to the
    bridge (same defaults, same omit-when-empty rules)."""
    if not isinstance(args, dict):
        return None, {"status": "error", "code": "bad_params",
                      "message": "%s: arguments must be an object" % tool}

    def need(name):
        if name not in args or args[name] is None:
            return "%s: missing required argument '%s'" % (tool, name)
        return None

    def want_str(name):
        v = args.get(name)
        if v is None:
            return None
        if not isinstance(v, str):
            return "%s: argument '%s' must be a string" % (tool, name)
        return None

    def want_num(name):
        v = args.get(name)
        if v is None:
            return None
        if not _is_number(v):
            return "%s: argument '%s' must be a number" % (tool, name)
        return None

    def want_int(name):
        v = args.get(name)
        if v is None:
            return None
        if not _is_int(v):
            return "%s: argument '%s' must be an integer" % (tool, name)
        return None

    def want_num_list(name):
        v = args.get(name)
        if v is None:
            return None
        if not isinstance(v, list) or not all(_is_number(x) for x in v):
            return "%s: argument '%s' must be a list of numbers" % (tool, name)
        return None

    if tool == "get_scene_info":
        return {}, None
    if tool == "get_object_info":
        err = need("name") or want_str("name")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        return {"name": args["name"]}, None
    if tool == "create_object":
        err = need("kind") or want_str("kind") or want_str("name") \
            or want_num_list("location")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        params = {"kind": args["kind"]}
        if args.get("name"):
            params["name"] = args["name"]
        if args.get("location") is not None:
            params["location"] = args["location"]
        return params, None
    if tool == "delete_object":
        err = need("name") or want_str("name")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        return {"name": args["name"]}, None
    if tool == "set_transform":
        err = need("name") or want_str("name") or want_num_list("location") \
            or want_num_list("rotation_euler") or want_num_list("scale")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        params = {"name": args["name"]}
        if args.get("location") is not None:
            params["location"] = args["location"]
        if args.get("rotation_euler") is not None:
            params["rotation_euler"] = args["rotation_euler"]
        if args.get("scale") is not None:
            params["scale"] = args["scale"]
        return params, None
    if tool == "create_material":
        err = need("name") or want_str("name") or want_num_list("color") \
            or want_num("roughness") or want_num("metallic") or want_str("assign_to")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        params = {"name": args["name"]}
        if args.get("color") is not None:
            params["color"] = args["color"]
        if args.get("roughness") is not None:
            params["roughness"] = args["roughness"]
        if args.get("metallic") is not None:
            params["metallic"] = args["metallic"]
        if args.get("assign_to"):
            params["assign_to"] = args["assign_to"]
        return params, None
    if tool == "add_modifier":
        err = need("name") or need("modifier_type") or want_str("name") \
            or want_str("modifier_type") or want_int("levels") or want_num("width")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        params = {"name": args["name"], "modifier_type": args["modifier_type"]}
        if args.get("levels") is not None:
            params["levels"] = args["levels"]
        if args.get("width") is not None:
            params["width"] = args["width"]
        return params, None
    if tool == "render":
        err = want_str("filepath") or want_int("width") or want_int("height") \
            or want_int("samples") or want_str("engine")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        params = {"width": args.get("width", 256),
                  "height": args.get("height", 256),
                  "samples": args.get("samples", 16)}
        if args.get("filepath"):
            params["filepath"] = args["filepath"]
        if args.get("engine"):
            params["engine"] = args["engine"]
        return params, None
    if tool == "save_blend":
        err = want_str("filepath")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        params = {}
        if args.get("filepath"):
            params["filepath"] = args["filepath"]
        return params, None
    if tool == "batch":
        err = need("ops")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        if not isinstance(args["ops"], list) \
                or not all(isinstance(o, dict) for o in args["ops"]):
            return None, {"status": "error", "code": "bad_params",
                          "message": "batch: argument 'ops' must be a list of objects"}
        return {"ops": args["ops"]}, None
    if tool == "execute_code":
        err = need("code") or want_str("code")
        if err:
            return None, {"status": "error", "code": "bad_params", "message": err}
        return {"code": args["code"]}, None
    return None, {"status": "error", "code": "bad_params",
                  "message": "unknown tool: %s" % tool}


# Bridge command + timeout per tool (same as the SDK server).
_TOOL_COMMANDS = {
    "get_scene_info": ("get_scene_info", 30),
    "get_object_info": ("get_object_info", 30),
    "create_object": ("create_object", 60),
    "delete_object": ("delete_object", 30),
    "set_transform": ("set_transform", 30),
    "create_material": ("create_material", 30),
    "add_modifier": ("add_modifier", 30),
    "render": ("render", 180),
    "save_blend": ("save_blend", 60),
    "batch": ("batch", 60),
    "execute_code": ("execute_code", 180),
}


def _available_tools():
    tools = list(TOOLS)
    if ALLOW_CODE:
        tools.append(EXECUTE_CODE_TOOL)
    return tools


def _call_tool(name, args):
    """Run one tool. Returns (text, is_error): text is the JSON of the
    bridge result dict (or of a local error dict); is_error matches the SDK
    server's behaviour — False for bridge results (even bridge status:error
    dicts, which are results, not protocol failures), True for unknown tools
    and bad params."""
    known = {t["name"] for t in _available_tools()}
    if name not in known:
        return "Unknown tool: %s" % name, True
    params, err = _check_args(name, args)
    if err is not None:
        return json.dumps(err, indent=2), True
    command, timeout = _TOOL_COMMANDS[name]
    result = _clean(_bridge(command, params, timeout=timeout))
    return json.dumps(result, indent=2), False


# ---------------------------------------------------------------- JSON-RPC
def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _result(rid, result):
    _send({"jsonrpc": "2.0", "id": rid, "result": result})


def _error(rid, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    _send({"jsonrpc": "2.0", "id": rid, "error": err})


def _handle(msg):
    """Handle one parsed JSON-RPC message. Returns True to keep serving."""
    if not isinstance(msg, dict):
        _error(None, -32600, "Invalid Request")
        return True
    method = msg.get("method")
    rid = msg.get("id")
    is_notification = "id" not in msg
    if not isinstance(method, str):
        if not is_notification:
            _error(rid, -32600, "Invalid Request")
        return True
    params = msg.get("params") or {}
    if not isinstance(params, dict):
        if not is_notification:
            _error(rid, -32602, "Invalid params")
        return True

    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in KNOWN_PROTOCOL_VERSIONS \
            else DEFAULT_PROTOCOL_VERSION
        if not is_notification:
            _result(rid, {
                "protocolVersion": version,
                "capabilities": {
                    "prompts": {"listChanged": False},
                    "resources": {"listChanged": False, "subscribe": False},
                    "tools": {"listChanged": False},
                },
                "instructions": INSTRUCTIONS,
                "serverInfo": {"name": NAME, "version": VERSION},
            })
        return True
    if method == "ping":
        if not is_notification:
            _result(rid, {})
        return True
    if method in ("notifications/initialized", "notifications/cancelled",
                  "notifications/progress", "notifications/roots/list_changed"):
        return True
    if method == "tools/list":
        if not is_notification:
            _result(rid, {"tools": _available_tools()})
        return True
    if method == "tools/call":
        if is_notification:
            return True
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            _error(rid, -32602, "Invalid params",
                   "tools/call requires a string 'name'")
            return True
        text, is_error = _call_tool(name, args)
        _result(rid, {"content": [{"type": "text", "text": text}],
                      "isError": is_error})
        return True
    if method == "resources/list":
        if not is_notification:
            _result(rid, {"resources": []})
        return True
    if method == "resources/templates/list":
        if not is_notification:
            _result(rid, {"resourceTemplates": []})
        return True
    if method == "prompts/list":
        if not is_notification:
            _result(rid, {"prompts": []})
        return True
    if method.startswith("notifications/"):
        return True
    if not is_notification:
        _error(rid, -32601, "Method not found", method)
    return True


def _serve_stdio():
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            return 0
        if line == "":
            return 0  # EOF: client went away
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _error(None, -32700, "Parse error")
            continue
        try:
            _handle(msg)
        except BrokenPipeError:
            return 0
        except Exception as e:  # noqa: BLE001 - never die on a bad request
            print("lite server error: %s" % e, file=sys.stderr)
            if isinstance(msg, dict) and "id" in msg:
                try:
                    _error(msg.get("id"), -32603, "Internal error")
                except BrokenPipeError:
                    return 0
    return 0


def _ping() -> int:
    resp = _bridge("ping", timeout=10)
    print(json.dumps(resp), file=sys.stderr)
    if resp.get("status") == "ok":
        print("BRIDGE OK: %s:%d reachable, Blender %s"
              % (HOST, PORT, resp.get("result", {}).get("blender", "?")))
        return 0
    print("BRIDGE FAIL: %s" % resp.get("message", resp))
    return 1


def main() -> int:
    if "--ping" in sys.argv:
        if not TOKEN:
            print("BLENDER_BRIDGE_TOKEN is not set", file=sys.stderr)
            return 2
        return _ping()
    if not TOKEN:
        print("BLENDER_BRIDGE_TOKEN is not set — refusing to start without a token",
              file=sys.stderr)
        print("Set it to the same token shown in Blender's AI Bridge panel.",
              file=sys.stderr)
        return 2
    print("blender-android-lite MCP server v%s -> bridge %s:%d (allow_code=%s)"
          % (VERSION, HOST, PORT, ALLOW_CODE), file=sys.stderr)
    return _serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
