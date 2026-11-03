#!/usr/bin/env python3
"""Runs the web demos' end-to-end specs in headless Chrome, over the DevTools protocol.

    driver.py --chrome BIN --profile DIR --base URL --out DIR [--serve DIR] [--commit SHA] SPEC...

With --serve, the driver serves DIR at --base, which must then be http://localhost:<port>/.

Each SPEC is a demo's `web/e2e.json`:

    {
      "path": "sd15/",              page under --base, as deployed on demo.tract.rs
      "timeout": 1800,              seconds for the whole demo
      "progress": "<js>",           logged whenever its value changes
      "fail": "<js>",               a string when the page shows an error, else falsy
      "steps": [
        {"wait": "<js>"},           poll until truthy (and `fail` stays falsy)
        {"run": "<js>"},            evaluate once
        {"check": "<js>"}           null or "" when good, else the reason it is not; or
                                    {fail: <that>, info: <a measurement to log>}
      ]
    }

Expressions run in the page, awaited, with the spec itself bound to `SPEC`. Exceptions thrown in
the page or its workers fail the demo. Each demo gets a screenshot in --out, and results.json
and summary.md sum them up. Standard library only, Python 3.9.
"""

import argparse
import base64
import functools
import http.server
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request

POLL = 1.0
# A hung browser fails the demo instead of the job's timeout.
CALL_TIMEOUT = 120


class Tab:
    """One page target and the workers it spawns, on a flat DevTools session."""

    def __init__(self, port, url):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/json/new?about:blank", method="PUT")
        self.target = json.load(urllib.request.urlopen(req, timeout=30))
        self.port = port
        path = self.target["webSocketDebuggerUrl"].split(f":{port}", 1)[1]
        self.sock = socket.create_connection(("127.0.0.1", port))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        self.buf = b""
        while b"\r\n\r\n" not in self.buf:
            self.buf += self.sock.recv(4096)
        self.buf = self.buf.split(b"\r\n\r\n", 1)[1]
        self.next_id = 0
        self.errors = []
        self.console_errors = []
        self.call("Runtime.enable")
        self.call("Log.enable")
        self.call("Page.enable")
        self.call("Target.setAutoAttach", autoAttach=True, waitForDebuggerOnStart=False, flatten=True)
        self.call("Page.navigate", url=url)

    def close(self):
        try:
            self.sock.close()
        finally:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/close/{self.target['id']}", timeout=30)

    def _frame(self):
        """Pops one whole frame off the buffer: (header byte, payload), or None for a partial one."""
        if len(self.buf) < 2:
            return None
        n, at = self.buf[1] & 0x7F, 2
        if n == 126:
            if len(self.buf) < 4:
                return None
            n, at = struct.unpack(">H", self.buf[2:4])[0], 4
        elif n == 127:
            if len(self.buf) < 10:
                return None
            n, at = struct.unpack(">Q", self.buf[2:10])[0], 10
        if len(self.buf) < at + n:
            return None
        head, payload, self.buf = self.buf[0], self.buf[at : at + n], self.buf[at + n :]
        return head, payload

    def _send(self, obj):
        data = json.dumps(obj).encode()
        n = len(data)
        if n < 126:
            head = bytes([0x81, 0x80 | n])
        elif n < 65536:
            head = bytes([0x81, 0x80 | 126]) + struct.pack(">H", n)
        else:
            head = bytes([0x81, 0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _recv(self, timeout):
        """The next message; socket.timeout after `timeout` seconds, with the buffer intact."""
        deadline = time.time() + timeout
        msg = b""
        while True:
            frame = self._frame()
            if frame is None:
                left = deadline - time.time()
                if left <= 0:
                    raise socket.timeout()
                self.sock.settimeout(left)
                chunk = self.sock.recv(1 << 20)
                if not chunk:
                    raise ConnectionError("DevTools connection closed")
                self.buf += chunk
                continue
            head, payload = frame
            if head & 0x0F in (0x8, 0x9, 0xA):
                continue
            msg += payload
            if head & 0x80:
                return json.loads(msg)

    def _post(self, method, session=None, **params):
        self.next_id += 1
        msg = {"id": self.next_id, "method": method, "params": params}
        if session:
            msg["sessionId"] = session
        self._send(msg)
        return self.next_id

    def call(self, method, **params):
        i = self._post(method, **params)
        deadline = time.time() + CALL_TIMEOUT
        while True:
            try:
                m = self._recv(deadline - time.time())
            except socket.timeout:
                raise TimeoutError(f"Chrome did not answer {method} within {CALL_TIMEOUT} s") from None
            if m.get("id") == i and "sessionId" not in m:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})
            self._event(m)

    def _event(self, m):
        method, params, session = m.get("method"), m.get("params", {}), m.get("sessionId")
        where = "worker" if session else "page"
        if method == "Target.attachedToTarget":
            child = params["sessionId"]
            self._post("Runtime.enable", session=child)
            self._post("Runtime.runIfWaitingForDebugger", session=child)
        elif method == "Runtime.exceptionThrown":
            d = params["exceptionDetails"]
            text = d.get("exception", {}).get("description") or d.get("text", "")
            self.errors.append(f"{where} exception: {text}")
            log(f"  {where} exception: {first_lines(text)}")
        elif method == "Runtime.consoleAPICalled" and params.get("type") in ("error", "warning", "log", "info"):
            args = " ".join(str(a.get("value", a.get("description", ""))) for a in params.get("args", []))
            if params["type"] == "error":
                self.console_errors.append(args)
            log(f"  {where} console.{params['type']}: {first_lines(args)}")
        elif method == "Log.entryAdded":
            e = params["entry"]
            log(f"  {where} log {e.get('level')}/{e.get('source')}: {first_lines(e.get('text', ''))}")

    def eval(self, expr, spec):
        wrapped = f"(async (SPEC) => ({expr}))({json.dumps(spec)})"
        r = self.call("Runtime.evaluate", expression=wrapped, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError(d.get("exception", {}).get("description") or d.get("text"))
        return r.get("result", {}).get("value")

    def pump(self, seconds):
        """Handles the events that arrive within `seconds`."""
        deadline = time.time() + seconds
        try:
            while True:
                self._event(self._recv(deadline - time.time()))
        except socket.timeout:
            pass

    def screenshot(self, path):
        self.call("Emulation.setDeviceMetricsOverride", width=1280, height=1000, deviceScaleFactor=1, mobile=False)
        shot = self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        with open(path, "wb") as f:
            f.write(base64.b64decode(shot["data"]))


def log(text):
    print(text, flush=True)


def first_lines(text, n=4):
    lines = str(text).splitlines()
    return "\n    ".join(lines[:n]) + (" …" if len(lines) > n else "")


def run_demo(port, base, spec_path, out, commit):
    spec = json.load(open(spec_path))
    name = spec["path"].strip("/")
    url = base.rstrip("/") + "/" + spec["path"]
    deadline = time.time() + spec.get("timeout", 600)
    result = {"demo": name, "url": url, "ok": False, "reason": None, "seconds": None}
    t0 = time.time()
    log(f"== {name}: {url}")

    tab = Tab(port, url)
    try:
        last = None

        # Page.navigate returns before the document commits: until then the steps would be
        # evaluated against about:blank.
        while not tab.eval(f"location.href.startsWith({json.dumps(url)}) && document.readyState !== 'loading'", spec):
            if time.time() > deadline:
                raise AssertionError("the page did not load")
            tab.pump(POLL)

        if commit:
            # The page fetches it: the CDN turns away clients that are not browsers.
            built = tab.eval("fetch('build.json', { cache: 'no-store' }).then((r) => r.json())", spec)
            if built.get("commit") != commit or built.get("dirty"):
                raise AssertionError(f"build.json is {built}, expected {commit}")

        def poll_once():
            nonlocal last
            tab.pump(POLL)
            if tab.errors:
                raise AssertionError(tab.errors[0])
            if spec.get("progress"):
                now = tab.eval(spec["progress"], spec)
                if now != last:
                    log(f"  {time.time() - t0:7.1f}s {first_lines(now, 2)}")
                    last = now
            if spec.get("fail"):
                why = tab.eval(spec["fail"], spec)
                if why:
                    raise AssertionError(f"page error: {why}")

        for step in spec["steps"]:
            if "wait" in step:
                while not tab.eval(step["wait"], spec):
                    if time.time() > deadline:
                        raise AssertionError(f"timed out waiting for {step['wait']}")
                    poll_once()
                poll_once()
            elif "run" in step:
                tab.eval(step["run"], spec)
            elif "check" in step:
                why = tab.eval(step["check"], spec)
                info = ""
                if isinstance(why, dict):
                    why, info = why.get("fail"), f" ({why.get('info')})"
                if why:
                    raise AssertionError(f"check failed: {why}{info}")
                log(f"  check ok: {step.get('name', step['check'][:60])}{info}")
        result["ok"] = True
    except Exception as e:
        result["reason"] = str(e)
        if tab.console_errors:
            first = " ".join(tab.console_errors[0].splitlines()[:2])
            result["reason"] = f"{first} / {str(e).splitlines()[0]}"
        log(f"  FAIL {first_lines(e, 6)}")
    finally:
        result["seconds"] = round(time.time() - t0, 1)
        try:
            tab.screenshot(os.path.join(out, f"{name}.png"))
        except Exception as e:
            log(f"  no screenshot: {e}")
        tab.close()
    log(f"  {'ok' if result['ok'] else 'FAILED'} in {result['seconds']} s")
    return result


class NoStore(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def serve(directory, base):
    port = int(re.match(r"http://localhost:(\d+)/?$", base).group(1))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(NoStore, directory=directory))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"serving {directory} at {base}")


def start_chrome(binary, profile, port, log_path):
    flags = [
        "--headless=new",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--enable-unsafe-webgpu",
        "--use-mock-keychain",
        "--password-store=basic",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
        "about:blank",
    ]
    stderr = open(log_path, "w")
    proc = subprocess.Popen([binary] + flags, stdout=stderr, stderr=stderr)
    for _ in range(60):
        try:
            version = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=5))
            log(f"{version['Browser']}, profile {profile}")
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    sys.exit("Chrome did not open its DevTools port")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--base", required=True, help="site root, as http://localhost:8123/ or https://demo.tract.rs/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--serve", help="directory to serve at --base")
    ap.add_argument("--commit", help="fail unless every demo's build.json names this clean commit")
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("specs", nargs="+")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.serve:
        serve(args.serve, args.base)

    chrome_log = os.path.join(args.out, "chrome.log")
    chrome = start_chrome(args.chrome, args.profile, args.port, chrome_log)
    try:
        results = [run_demo(args.port, args.base, s, args.out, args.commit) for s in args.specs]
    finally:
        chrome.terminate()
        try:
            chrome.wait(10)
        except subprocess.TimeoutExpired:
            chrome.kill()

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"### Web demos on {args.base}\n\n| demo | result | time |\n|---|---|---|\n")
        for r in results:
            reason = re.sub(r"\s+", " ", r["reason"] or "")[:200].replace("|", "\\|")
            f.write(f"| {r['demo']} | {'✅' if r['ok'] else '❌ ' + reason} | {r['seconds']} s |\n")
    failed = [r["demo"] for r in results if not r["ok"]]
    if failed:
        with open(chrome_log, errors="replace") as f:
            tail = [l for l in f.read().splitlines() if "CVDisplayLinkCreateWithCGDisplay" not in l][-40:]
        log("== tail of chrome.log\n" + "\n".join(tail))
    log(f"failed: {', '.join(failed)}" if failed else "all demos ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
