#!/usr/bin/env python3
# blender_mcp_http.py — HTTP front-end for the stdio MCP server (v1.0.0).
# Lets HTTP MCP clients (notably the AiRA Android app, whose McpClient
# speaks JSON-RPC 2.0 over HTTP POST) drive Blender through the same
# stdlib-only lite MCP server that Termux agents use over stdio.
#
# This file uses NOTHING but the Python standard library (http.server,
# json, os, subprocess, sys, threading).
#
# Architecture: on startup it spawns ONE persistent
# blender_android_mcp_lite.py subprocess (stdio JSON-RPC) and holds it for
# the server's lifetime. Each HTTP POST carries exactly one JSON-RPC
# message; the wrapper forwards it down the stdio pipe (under ONE global
# lock — a single stdio pair is strictly sequential, never interleave two
# requests) and returns the one response line as application/json.
# The lite server (and through it the Blender bridge) does all the real
# work; this wrapper adds no tools and no Blender logic.
#
# Config (environment):
#   BLENDER_MCP_HTTP_HOST  default 127.0.0.1 (loopback ONLY — never 0.0.0.0)
#   BLENDER_MCP_HTTP_PORT  default 9877
#   MCP_HTTP_TOKEN         optional — when set, every POST must carry
#                          "Authorization: Bearer <token>" (else HTTP 401)
#   BLENDER_MCP_SERVER_PY  optional override for the lite server path
#                          (default: blender_android_mcp_lite.py next to
#                          this file)
#   BLENDER_BRIDGE_HOST / BLENDER_BRIDGE_PORT / BLENDER_BRIDGE_TOKEN /
#   BLENDER_ALLOW_CODE     passed straight through to the lite server.
#                          BLENDER_BRIDGE_TOKEN is REQUIRED — the wrapper
#                          refuses to start without it (exit 2).
#
# Run (Termux, Blender open, AI Bridge Status: running):
#   BLENDER_BRIDGE_TOKEN=<token from Blender's AI Bridge panel> \
#     python3 blender_mcp_http.py
# AiRA then connects to http://127.0.0.1:9877 (or .../mcp) from its Tools
# screen, with the MCP_HTTP_TOKEN as the token if one is set.
#
# Security: binds loopback only; when MCP_HTTP_TOKEN is unset, loopback-only
# access is the protection — any process on the same phone could call it,
# so set a token if that matters to you. Tokens are never printed or logged.

import json
import os
import select
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "1.0.0"
NAME = "blender-mcp-http"

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SERVER_PY = os.path.join(HERE, "blender_android_mcp_lite.py")

HOST = os.environ.get("BLENDER_MCP_HTTP_HOST", "127.0.0.1").strip() or "127.0.0.1"
try:
    PORT = int(os.environ.get("BLENDER_MCP_HTTP_PORT", "9877"))
except ValueError:
    print("BLENDER_MCP_HTTP_PORT is not a number", file=sys.stderr)
    sys.exit(2)
HTTP_TOKEN = os.environ.get("MCP_HTTP_TOKEN", "").strip()
SERVER_PY = os.environ.get("BLENDER_MCP_SERVER_PY", "").strip() or DEFAULT_SERVER_PY
BRIDGE_TOKEN = os.environ.get("BLENDER_BRIDGE_TOKEN", "").strip()

# The bridge's own command budget is 120s; a tool call (render) may take up
# to 180s inside the lite server, so the HTTP read waits a little longer.
RPC_READ_TIMEOUT = 200


class McpProcess:
    """Owns the persistent lite-server subprocess and its stdio pipe.

    One global lock serializes every round-trip: write one JSON line, read
    exactly one response line. Reads use select + os.read on the raw fd so
    a line already buffered in Python never looks like a timeout.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._buf = b""
        self._restarts = 0

    def start(self):
        env = dict(os.environ)
        self._proc = subprocess.Popen(
            [sys.executable, SERVER_PY],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, env=env)  # stderr inherits: lite diagnostics show
        self._buf = b""
        print("blender-mcp-http: MCP subprocess started (pid %d)"
              % self._proc.pid, file=sys.stderr)

    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _restart_once(self):
        """Restart the subprocess once. Returns True if a live proc exists."""
        if self._alive():
            return True
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._restarts += 1
        if self._restarts > 10:
            print("blender-mcp-http: subprocess died too often, giving up",
                  file=sys.stderr)
            return False
        print("blender-mcp-http: MCP subprocess died, restarting "
              "(restart #%d)" % self._restarts, file=sys.stderr)
        try:
            self.start()
        except Exception as e:  # noqa: BLE001 - startup failure is reported
            print("blender-mcp-http: restart failed: %s" % e, file=sys.stderr)
            return False
        return self._alive()

    def _readline(self, timeout):
        """Read one newline-terminated line from the subprocess, or None."""
        import time
        deadline = time.time() + timeout
        while b"\n" not in self._buf:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            r, _, _ = select.select([self._proc.stdout.fileno()], [], [],
                                    remaining)
            if not r:
                return None
            try:
                chunk = os.read(self._proc.stdout.fileno(), 65536)
            except OSError:
                return None
            if not chunk:
                return None  # EOF: subprocess closed stdout
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def roundtrip(self, line, timeout=RPC_READ_TIMEOUT):
        """Send one JSON line, return the response line (str), or None."""
        with self._lock:
            for attempt in range(2):  # first try + one restart retry
                if not self._alive():
                    if not self._restart_once():
                        return None
                try:
                    self._proc.stdin.write(line.encode() + b"\n")
                    self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    if attempt == 0 and self._restart_once():
                        continue
                    return None
                resp = self._readline(timeout)
                if resp is None:
                    # No answer: the subprocess may have died mid-call.
                    if attempt == 0 and self._restart_once():
                        continue
                    return None
                try:
                    return resp.decode()
                except UnicodeDecodeError:
                    return None
            return None

    def notify(self, line):
        """Fire-and-forget a notification (no reply is ever read)."""
        with self._lock:
            if not self._alive():
                self._restart_once()
            if not self._alive():
                return
            try:
                self._proc.stdin.write(line.encode() + b"\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass  # a notification that cannot be delivered is dropped

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass


MCP = McpProcess()


def _jsonrpc_error(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code,
                                                   "message": message}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep request lines on stderr
        sys.stderr.write("blender-mcp-http: %s - %s\n"
                         % (self.address_string(), fmt % args))

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _auth_ok(self):
        if not HTTP_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        # Constant-time compare; never log the presented value.
        import hmac
        return hmac.compare_digest(auth, "Bearer " + HTTP_TOKEN)

    def do_GET(self):
        if self.path not in ("/", "/mcp"):
            self._send_json(404, {"status": "error", "code": "not_found",
                                  "message": "unknown path: %s" % self.path})
            return
        self._send_json(200, {"status": "ok", "server": NAME,
                              "version": VERSION,
                              "mcp_subprocess_pid":
                                  MCP._proc.pid if MCP._proc else None,
                              "auth": bool(HTTP_TOKEN)})

    def do_POST(self):
        if self.path not in ("/", "/mcp"):
            self._send_json(404, _jsonrpc_error(
                None, -32601, "Method not found: %s" % self.path))
            return
        if not self._auth_ok():
            self._send_json(401, _jsonrpc_error(
                None, -32000, "Unauthorized: missing or wrong Bearer token"))
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            msg = json.loads(raw.decode())
        except (ValueError, UnicodeDecodeError):
            self._send_json(200, _jsonrpc_error(None, -32700, "Parse error"))
            return
        if not isinstance(msg, dict):
            self._send_json(200, _jsonrpc_error(None, -32600,
                                                "Invalid Request"))
            return
        if "id" not in msg:
            # Notification: forward, never wait for a reply.
            MCP.notify(json.dumps(msg))
            self._send_empty(202)
            return
        rid = msg.get("id")
        resp_line = MCP.roundtrip(json.dumps(msg))
        if resp_line is None:
            self._send_json(500, _jsonrpc_error(
                rid, -32603,
                "MCP server did not answer in %ds" % RPC_READ_TIMEOUT))
            return
        try:
            resp = json.loads(resp_line)
        except ValueError:
            self._send_json(500, _jsonrpc_error(
                rid, -32603, "MCP server returned invalid JSON"))
            return
        self._send_json(200, resp)

    # AiRA only POSTs, but reject other verbs cleanly instead of hanging.
    def do_PUT(self):
        self._send_json(405, _jsonrpc_error(None, -32601, "Method not found"))

    def do_DELETE(self):
        self._send_json(405, _jsonrpc_error(None, -32601, "Method not found"))


def main() -> int:
    if not BRIDGE_TOKEN:
        print("BLENDER_BRIDGE_TOKEN is not set — refusing to start without a token",
              file=sys.stderr)
        print("Set it to the same token shown in Blender's AI Bridge panel.",
              file=sys.stderr)
        return 2
    if not os.path.isfile(SERVER_PY):
        print("MCP server file not found: %s" % SERVER_PY, file=sys.stderr)
        return 2
    if HOST not in ("127.0.0.1", "localhost", "::1"):
        print("refusing to bind a non-loopback host: %s" % HOST,
              file=sys.stderr)
        return 2
    MCP.start()
    if not MCP._alive():
        print("MCP subprocess exited immediately — check BLENDER_BRIDGE_TOKEN",
              file=sys.stderr)
        return 2
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print("blender-mcp-http v%s listening on %s:%d (token auth: %s, "
          "MCP subprocess pid %d)"
          % (VERSION, HOST, PORT, "on" if HTTP_TOKEN else "off",
             MCP._proc.pid), file=sys.stderr)
    print("AiRA: Tools -> MCP server URL http://%s:%d (or /mcp)"
          % (HOST, PORT), file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        MCP.stop()
    print("blender-mcp-http: stopped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
