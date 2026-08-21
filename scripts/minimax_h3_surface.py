#!/usr/bin/python3
"""Local browser surface for the MiniMax-H3 MLX CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
GENERATOR = REPO / "scripts" / "generate.py"
PYTHON = REPO / ".venv" / "bin" / "python"
CHECKPOINT = WORK / "checkpoints" / "minimax-h3-fl2va"
TRANSFORMER = WORK / "models" / "minimax-h3-mlx-6bit-streamed-adaln"
DEFAULT_OUTPUT = REPO.parent.parent / "outputs" / "minimax-h3-output.mp4"
PORT = 8765
RUNTIME_ASSETS_ENV = "MINIMAX_H3_RUNTIME_ASSETS"
CURRENT_RUNTIME_ID = "current"
BETA_RUNTIME_ID = "beta-0.6"
DEFAULT_RUNTIME_ID = CURRENT_RUNTIME_ID
RUNTIME_LABELS = {
    CURRENT_RUNTIME_ID: "Current",
    BETA_RUNTIME_ID: "Beta 0.6",
}

SIZES = {
    "0.2 MP — 608 × 352": "0.2",
    "0.3 MP — 736 × 416": "0.3",
    "0.4 MP — 864 × 480": "0.4",
    "0.5 MP — 960 × 544": "0.5",
    "0.98 MP — 1344 × 768": "0.98",
}

PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniMax H3 — Local MLX</title>
<style>
:root { color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif; }
body { margin: 0; background: #f4f5f7; color: #1d1f23; }
main { width: min(760px, calc(100% - 40px)); margin: 36px auto; }
.card { background: white; border: 1px solid #d8dbe0; border-radius: 16px; padding: 28px; box-shadow: 0 8px 30px #00000012; }
h1 { margin: 0 0 6px; font-size: 25px; }
.subtitle { margin: 0 0 24px; color: #68707b; }
label { display: block; margin: 16px 0 7px; font-weight: 600; }
textarea, input, select { box-sizing: border-box; width: 100%; border: 1px solid #c9cdd4; border-radius: 9px; padding: 10px 12px; font: inherit; background: white; color: inherit; }
textarea { min-height: 150px; resize: vertical; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.runtime-row { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.actions { display: flex; align-items: center; gap: 10px; margin-top: 22px; }
button { border: 0; border-radius: 9px; padding: 10px 17px; font: inherit; font-weight: 600; cursor: pointer; }
#generate { background: #147ef5; color: white; }
#stop { background: #e4e6ea; color: #20242a; }
button:disabled { opacity: .45; cursor: default; }
#status { margin-left: auto; color: #68707b; }
pre { min-height: 120px; max-height: 260px; overflow: auto; background: #17191d; color: #d9e2ee; border-radius: 9px; padding: 13px; white-space: pre-wrap; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
.note { margin-top: 12px; color: #68707b; font-size: 13px; }
@media (prefers-color-scheme: dark) {
  body { background: #16181b; color: #f4f5f7; }
  .card, input, select, textarea { background: #24272c; border-color: #424750; }
  #stop { background: #3a3e45; color: #f4f5f7; }
}
</style>
</head>
<body>
<main><section class="card">
<h1>MiniMax H3</h1>
<p class="subtitle">Local MLX text-to-video</p>
<label for="prompt">Prompt</label>
<textarea id="prompt">A cinematic live-action scene with natural movement and atmosphere.</textarea>
<div class="row runtime-row">
  <div><label for="runtime">Runtime</label>
    <select id="runtime">
      <option value="current">Current</option>
      <option value="beta-0.6">Beta 0.6</option>
    </select>
  </div>
  <div><label for="size">Size</label>
    <select id="size">
      <option value="0.2">0.2 MP — 608 × 352</option>
      <option value="0.3">0.3 MP — 736 × 416</option>
      <option value="0.4">0.4 MP — 864 × 480</option>
      <option value="0.5">0.5 MP — 960 × 544</option>
      <option value="0.98">0.98 MP — 1344 × 768</option>
    </select>
  </div>
  <div><label for="length">Length (seconds)</label>
    <input id="length" type="number" min="5" max="15" step="1" value="5">
  </div>
</div>
<label for="output">Output path</label>
<input id="output" value="__DEFAULT_OUTPUT__">
<div class="actions">
  <button id="generate" onclick="generate()">Generate</button>
  <button id="stop" onclick="stopJob()" disabled>Stop</button>
  <span id="status">Ready</span>
</div>
<p class="note">The default is 0.2 MP to reduce memory pressure. H3 supports 5–15 seconds. Beta 0.6 uses the named runtime profile configured by MINIMAX_H3_RUNTIME_ASSETS.</p>
<label for="log">Progress</label>
<pre id="log">Ready.</pre>
</section></main>
<script>
const $ = id => document.getElementById(id);
let running = false;
function setRunning(value) {
  running = value;
  $('generate').disabled = value;
  $('stop').disabled = !value;
}
async function generate() {
  const prompt = $('prompt').value.trim();
  const length = Number($('length').value);
  const output = $('output').value.trim();
  if (!prompt) return alert('Enter a prompt first.');
  if (!Number.isInteger(length) || length < 5 || length > 15) return alert('Length must be 5–15 seconds.');
  if (!output) return alert('Enter an output path.');
  const response = await fetch('/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({runtime: $('runtime').value, prompt, length, output, megapixels: $('size').value})});
  const result = await response.json();
  if (!response.ok) return alert(result.error || 'Could not start generation.');
  $('log').textContent = '';
  $('status').textContent = `Running · ${result.runtime_label || 'Selected runtime'}…`;
  setRunning(true);
}
async function stopJob() {
  await fetch('/stop', {method: 'POST'});
  $('status').textContent = 'Stopping…';
}
async function poll() {
  try {
    const response = await fetch('/status');
    const result = await response.json();
    $('log').textContent = result.log || '';
    $('log').scrollTop = $('log').scrollHeight;
    const runtimeLabel = result.runtime_label || 'Selected runtime';
    if (result.running) {
      $('status').textContent = `Running · ${runtimeLabel}…`;
      setRunning(true);
    } else if (running) {
      setRunning(false);
      $('status').textContent = result.exit_code === 0 ? `Finished · ${runtimeLabel}` : `Failed (${result.exit_code}) · ${runtimeLabel}`;
    }
  } catch (error) {
    $('status').textContent = 'Disconnected';
  }
  setTimeout(poll, 1000);
}
poll();
</script>
</body></html>'''

JOB_LOCK = threading.Lock()
JOB = {
    "process": None,
    "log": "",
    "exit_code": None,
    "runtime_id": DEFAULT_RUNTIME_ID,
    "runtime_label": RUNTIME_LABELS[DEFAULT_RUNTIME_ID],
}


def send_json(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def build_generation_command(
    runtime_id: str,
    prompt: str,
    megapixels: str,
    length: int,
    output: Path,
) -> list[str]:
    """Build the child request while keeping named-runtime admission in generate.py."""
    command = [str(PYTHON), "-u", str(GENERATOR), prompt]
    if runtime_id == BETA_RUNTIME_ID:
        command.extend(["--runtime", BETA_RUNTIME_ID])
    elif runtime_id == CURRENT_RUNTIME_ID:
        command.extend([
            "--checkpoint", str(CHECKPOINT),
            "--transformer", str(TRANSFORMER),
        ])
    else:
        raise ValueError(f"unsupported runtime: {runtime_id}")
    command.extend([
        "--megapixels", megapixels,
        "--duration", str(length),
        "--output", str(output),
    ])
    return command


def consume_output(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        with JOB_LOCK:
            JOB["log"] += line
            JOB["log"] = JOB["log"][-200_000:]
    exit_code = process.wait()
    with JOB_LOCK:
        JOB["exit_code"] = exit_code
        JOB["process"] = None


def start_job(data: dict) -> tuple[dict, int]:
    runtime_id = str(data.get("runtime", DEFAULT_RUNTIME_ID)).strip() or DEFAULT_RUNTIME_ID
    prompt = str(data.get("prompt", "")).strip()
    output = Path(str(data.get("output", "")).strip()).expanduser()
    try:
        length = int(data.get("length"))
    except (TypeError, ValueError):
        return {"error": "Length must be a whole number from 5 to 15."}, 400
    megapixels = str(data.get("megapixels", "0.2"))
    if runtime_id not in RUNTIME_LABELS:
        return {"error": "Choose a supported runtime."}, 400
    if not prompt:
        return {"error": "Enter a prompt first."}, 400
    if not 5 <= length <= 15:
        return {"error": "Length must be from 5 to 15 seconds."}, 400
    if megapixels not in SIZES.values():
        return {"error": "Choose a supported size preset."}, 400
    if not output.name:
        return {"error": "Enter an output path."}, 400

    with JOB_LOCK:
        if JOB["process"] is not None and JOB["process"].poll() is None:
            return {"error": "A generation is already running."}, 409
        output.parent.mkdir(parents=True, exist_ok=True)
        command = build_generation_command(runtime_id, prompt, megapixels, length, output)
        environment = {**os.environ, "PYTHONPATH": str(REPO)}
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        JOB["process"] = process
        JOB["log"] = (
            f"runtime: {RUNTIME_LABELS[runtime_id]} ({runtime_id})\n"
            + "$ " + " ".join(command) + "\n\n"
        )
        JOB["exit_code"] = None
        JOB["runtime_id"] = runtime_id
        JOB["runtime_label"] = RUNTIME_LABELS[runtime_id]
    threading.Thread(target=consume_output, args=(process,), daemon=True).start()
    return {
        "ok": True,
        "runtime_id": runtime_id,
        "runtime_label": RUNTIME_LABELS[runtime_id],
    }, 200


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.replace("__DEFAULT_OUTPUT__", json.dumps(str(DEFAULT_OUTPUT))).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/status":
            with JOB_LOCK:
                process = JOB["process"]
                running = process is not None and process.poll() is None
                send_json(
                    self,
                    {
                        "running": running,
                        "log": JOB["log"],
                        "exit_code": JOB["exit_code"],
                        "runtime_id": JOB["runtime_id"],
                        "runtime_label": JOB["runtime_label"],
                    },
                )
        else:
            send_json(self, {"error": "Not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length) or b"{}")
        path = urlparse(self.path).path
        if path == "/generate":
            payload, status = start_job(data)
            send_json(self, payload, status)
        elif path == "/stop":
            with JOB_LOCK:
                process = JOB["process"]
                if process is not None and process.poll() is None:
                    process.terminate()
            send_json(self, {"ok": True})
        else:
            send_json(self, {"error": "Not found"}, 404)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"MiniMax H3 surface: {url}", flush=True)
    if "--open" in sys.argv:
        subprocess.Popen(["open", url])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
