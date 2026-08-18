#!/usr/bin/env python3
"""Run the local MiniMax H3 Render Lab browser surface.

Start from the repository root with::

    ./.venv/bin/python tools/render_lab/server.py

The server binds only to loopback and starts the existing ``scripts/generate.py`` CLI in one child
process per render.  It is intentionally a standard-library surface because this checkout does not
ship Streamlit.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.render_lab.runner import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    FIRST_LAST,
    I2V,
    RenderBusyError,
    RenderController,
    RenderRequest,
    RenderValidationError,
    T2V,
    UploadedImage,
    history_rows,
    parse_additional_loras_payload,
    resolve_output_root,
)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniMax H3 — Render Lab</title>
<style>
:root { color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif; }
body { margin: 0; background: #f4f5f7; color: #1d1f23; }
main { width: min(1040px, calc(100% - 32px)); margin: 28px auto 60px; }
.card { background: white; border: 1px solid #d8dbe0; border-radius: 16px; padding: 24px; margin: 14px 0; box-shadow: 0 8px 30px #00000010; }
h1 { margin: 0 0 4px; font-size: 26px; } h2 { font-size: 17px; margin: 0 0 14px; }
.subtitle, .note, .meta { color: #68707b; font-size: 13px; }
label { display: block; margin: 13px 0 6px; font-weight: 650; }
textarea, input, select { box-sizing: border-box; width: 100%; border: 1px solid #c9cdd4; border-radius: 9px; padding: 10px 12px; font: inherit; background: white; color: inherit; }
input[type="checkbox"] { width: auto; padding: 0; }
textarea { min-height: 120px; resize: vertical; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.image-slot { border: 1px dashed #aeb5bf; border-radius: 9px; padding: 10px; }
.hidden { display: none !important; }
.actions { display: flex; align-items: center; gap: 10px; margin-top: 18px; }
button { border: 0; border-radius: 9px; padding: 10px 17px; font: inherit; font-weight: 650; cursor: pointer; }
#render { background: #147ef5; color: white; } button:disabled { opacity: .45; cursor: default; }
#status { margin-left: auto; color: #68707b; }
.pill { display: inline-block; border-radius: 999px; padding: 3px 8px; background: #e9edf2; font-size: 12px; }
.pill.ok { background: #d8f5df; color: #145b29; } .pill.bad { background: #ffe0e0; color: #8b1a1a; }
pre { min-height: 120px; max-height: 330px; overflow: auto; background: #17191d; color: #d9e2ee; border-radius: 9px; padding: 13px; white-space: pre-wrap; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
video { width: min(100%, 720px); border-radius: 9px; background: #111; }
table { width: 100%; border-collapse: collapse; font-size: 13px; } th, td { border-bottom: 1px solid #e3e6ea; padding: 9px 7px; text-align: left; vertical-align: top; }
.small { font-size: 12px; } .warning { padding: 10px; border-radius: 9px; background: #fff2cc; color: #6e5200; margin-top: 12px; }
.lora-row { display: grid; grid-template-columns: minmax(0, 1fr) 140px auto; gap: 10px; align-items: end; margin: 10px 0; }
.lora-row label { margin: 0 0 6px; }
.secondary { background: #e9edf2; color: #1d1f23; }
@media (max-width: 700px) { .grid, .grid.three { grid-template-columns: 1fr; } }
@media (max-width: 700px) { .lora-row { grid-template-columns: 1fr; } }
@media (prefers-color-scheme: dark) {
 body { background: #16181b; color: #f4f5f7; } .card, input, select, textarea { background: #24272c; border-color: #424750; }
 .pill { background: #3a3e45; } .pill.ok { background: #164b27; color: #c9f4d3; } .pill.bad { background: #5c2222; color: #ffd4d4; }
 th, td { border-color: #3b3f46; } .warning { background: #4b3e1e; color: #f5d98d; }
}
</style>
</head>
<body>
<main>
<section class="card">
  <h1>MiniMax H3 Render Lab</h1>
  <p class="subtitle">Local operator surface around the existing MLX generation CLI. One child render at a time; every run keeps its own evidence.</p>
  <div id="runtime" class="meta">Loading runtime identity…</div>
  <div id="runtime-warning" class="warning hidden"></div>
</section>

<section class="card">
  <h2>Render controls</h2>
  <label for="mode">Mode</label>
  <select id="mode"></select>
  <label for="prompt">Prompt</label>
  <textarea id="prompt" placeholder="Describe the video to generate.">A cinematic live-action scene with natural movement and atmosphere.</textarea>
  <label for="text-encoder">Text encoder</label>
  <select id="text-encoder"></select>
  <div id="text-encoder-hint" class="meta">Canonical Qwen3-VL is the default H3 conditioning path.</div>
  <label for="conditioning-artifact">Conditioning artifact (optional)</label>
  <input id="conditioning-artifact" type="text" spellcheck="false" placeholder="/path/to/conditioning-artifact.npz">
  <div class="meta">Replay is currently T2V-only. The artifact must match the visible prompt and the selected Canonical Qwen3-VL encoder/checkpoint identity. Leave blank to encode with the live text encoder.</div>
  <div class="grid">
    <div id="image1-slot" class="image-slot hidden"><label for="image1">First image</label><input id="image1" type="file" accept="image/*"></div>
    <div id="image2-slot" class="image-slot hidden"><label for="image2">Last image</label><input id="image2" type="file" accept="image/*"></div>
  </div>
  <div class="grid three">
    <div class="dimension-control">
      <label for="width">Width (pixels)</label>
      <input id="width-range" type="range" min="128" max="1344" step="32" aria-label="Width slider">
      <input id="width" type="number" min="128" max="1344" step="1" inputmode="numeric" aria-describedby="width-error">
      <div id="width-error" class="meta"></div>
    </div>
    <div class="dimension-control">
      <label for="height">Height (pixels)</label>
      <input id="height-range" type="range" min="128" max="1344" step="32" aria-label="Height slider">
      <input id="height" type="number" min="128" max="1344" step="1" inputmode="numeric" aria-describedby="height-error">
      <div id="height-error" class="meta"></div>
    </div>
    <div><label>Geometry</label><div id="geometry" class="meta">Width and height are independently selectable.</div></div>
  </div>
  <div class="grid three">
    <div><label for="steps">Inference steps</label><input id="steps" type="number" min="2" max="40" step="1"></div>
    <div><label for="duration">Duration (seconds)</label><input id="duration" type="number" min="5" max="15" step="1"></div>
    <div><label for="seed">Seed</label><input id="seed" type="number" step="1"></div>
  </div>
  <div class="grid three">
    <div><label for="turbo-preset">Turbo preset</label><select id="turbo-preset"></select><div id="turbo-preset-details" class="meta"></div></div>
    <div class="meta">Turbo owns scheduling, adapter family, runtime variant, scheduler shifts, and NFE.</div>
    <div class="meta">Additional LoRAs contribute model deltas only and keep their own scales.</div>
  </div>
  <div>
    <label>Additional LoRAs</label>
    <div id="additional-lora-rows"></div>
    <button id="add-additional-lora" class="secondary" type="button" onclick="addAdditionalLora()">+ Add LoRA</button>
    <div class="small">Rows are serialized in display order. Remove a row to omit it from the request.</div>
  </div>
  <div class="grid three">
    <div><label for="output-root">Output root</label><input id="output-root" spellcheck="false"></div>
    <div><label for="output-name">Output name</label><input id="output-name" value="render.mp4" spellcheck="false"></div>
  </div>
  <div class="actions"><button id="render" onclick="renderJob()">Render</button><span id="status">Ready</span></div>
  <p class="note">Width and height must each be positive, between 128 and 1344 pixels, and divisible by 32. Sliders are snapped to 32-pixel increments; numeric entries are validated before launch.</p>
</section>

<section class="card">
  <h2>Live process output</h2>
  <div id="run-meta" class="meta">No active run.</div>
  <pre id="log">Ready.</pre>
  <div id="benchmark"></div>
  <div id="preview"></div>
</section>

<section class="card">
  <h2>Recent run history</h2>
  <div id="history" class="meta">Loading…</div>
</section>
</main>
<script>
const $ = id => document.getElementById(id);
let config = null;
let activeRunId = null;
const STATUS_POLL_INTERVAL_MS = 1000;
const TERMINAL_STATUSES = new Set(['succeeded', 'failed']);
let statusPollTimer = null;
let statusPollInFlight = false;

function setStatus(text, bad=false) { $('status').textContent = text; $('status').style.color = bad ? '#b42318' : ''; }
function dimensionContract() {
  return config.resolution_contract || {min_dimension: 128, max_dimension: 1344, step: 32};
}
function validateDimension(axis) {
  const value = Number($(axis).value);
  const contract = dimensionContract();
  let error = '';
  if (!Number.isInteger(value)) error = `${axis} must be an integer`;
  else if (value <= 0) error = `${axis} must be positive`;
  else if (value < contract.min_dimension || value > contract.max_dimension) error = `${axis} must be between ${contract.min_dimension} and ${contract.max_dimension}`;
  else if (value % contract.step !== 0) error = `${axis} must be divisible by ${contract.step}`;
  $(`${axis}-error`).textContent = error;
  return {value, error};
}
function syncDimensionFromSlider(axis) {
  $(axis).value = $(`${axis}-range`).value;
  validateDimension(axis);
  updateGeometry();
}
function syncDimensionFromNumber(axis) {
  const result = validateDimension(axis);
  if (!result.error) $(`${axis}-range`).value = result.value;
  updateGeometry();
}
function updateGeometry() {
  const width = validateDimension('width');
  const height = validateDimension('height');
  if (width.error || height.error) {
    $('geometry').textContent = 'Enter valid width and height values.';
    return;
  }
  const g = config.geometry_contract;
  const ratio = g.spatial_compression_ratio;
  const patch = g.dit_patch_size;
  const lh = height.value / ratio, lw = width.value / ratio;
  const tokens = Number.isInteger(lh) && Number.isInteger(lw) ? (lh / patch[1]) * (lw / patch[2]) : null;
  $('geometry').textContent = `${width.value} × ${height.value}` + (tokens ? ` · latent grid ${lh} × ${lw} · ${tokens} spatial tokens/frame` : '');
}
function updateMode() {
  const mode = $('mode').value;
  $('image1-slot').classList.toggle('hidden', mode === 'T2V');
  $('image2-slot').classList.toggle('hidden', mode !== 'FIRST_LAST');
  updateTextEncoderControls();
}
function selectedTextEncoder() {
  const id = $('text-encoder').value;
  return (config.text_encoders || []).find(item => item.id === id) || null;
}
function updateTextEncoderControls() {
  const mode = $('mode').value;
  const heretic = (config.text_encoders || []).find(item => item.experimental);
  const current = selectedTextEncoder();
  const hereticDisabled = Boolean(heretic && (mode !== 'T2V' || !heretic.available));
  for (const option of $('text-encoder').options) {
    const item = (config.text_encoders || []).find(entry => entry.id === option.value);
    option.disabled = Boolean(item && item.experimental && (mode !== 'T2V' || !item.available));
  }
  if (hereticDisabled && current && current.experimental) {
    $('text-encoder').value = (config.text_encoders || []).find(item => !item.experimental).id;
  }
  const selected = selectedTextEncoder();
  if (mode !== 'T2V' && heretic) {
    $('text-encoder-hint').textContent = 'Heretic is currently text-only; image-conditioned modes require Canonical Qwen3-VL.';
  } else if (selected && selected.experimental && !selected.available) {
    $('text-encoder-hint').textContent = selected.disabled_reason || selected.hint;
  } else {
    $('text-encoder-hint').textContent = selected ? selected.hint : 'Canonical Qwen3-VL is the default H3 conditioning path.';
  }
}
let additionalLoras = [];
let normalStepsBeforeTurbo = null;
function htmlEscape(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
}
function syncAdditionalLoraRows() {
  for (const [index, row] of additionalLoras.entries()) {
    const path = $(`additional-lora-path-${index}`);
    const scale = $(`additional-lora-scale-${index}`);
    if (path) row.path = path.value;
    if (scale) row.scale = scale.value;
  }
}
function renderAdditionalLoraRows() {
  const container = $('additional-lora-rows');
  container.innerHTML = additionalLoras.map((row, index) => `
    <div class="lora-row" data-index="${index}">
      <div><label for="additional-lora-path-${index}">Path</label><input id="additional-lora-path-${index}" type="text" spellcheck="false" placeholder="/path/to/style.safetensors" value="${htmlEscape(row.path)}"></div>
      <div><label for="additional-lora-scale-${index}">Scale</label><input id="additional-lora-scale-${index}" type="number" min="0" step="0.01" value="${htmlEscape(row.scale)}"></div>
      <button class="secondary remove-additional-lora" type="button" data-index="${index}">Remove</button>
    </div>`).join('');
  for (const button of container.querySelectorAll('.remove-additional-lora')) {
    button.onclick = () => removeAdditionalLora(Number(button.dataset.index));
  }
}
function addAdditionalLora() {
  syncAdditionalLoraRows();
  additionalLoras.push({path: '', scale: '1.0'});
  renderAdditionalLoraRows();
  $(`additional-lora-path-${additionalLoras.length - 1}`).focus();
}
function removeAdditionalLora(index) {
  syncAdditionalLoraRows();
  additionalLoras.splice(index, 1);
  renderAdditionalLoraRows();
}
function selectedAdditionalLoras() {
  syncAdditionalLoraRows();
  return additionalLoras.map(row => ({path: row.path, scale: row.scale}));
}
function selectedTurboPreset() {
  const id = $('turbo-preset').value;
  return (config.turbo_presets || []).find(item => item.id === id) || null;
}
function updateTurboPresetDetails() {
  const preset = selectedTurboPreset();
  if (!preset || preset.id === 'none') {
    $('turbo-preset-details').textContent = 'None / Reference · existing normal Render Lab behavior.';
    return;
  }
  const variant = preset.runtime_variant ? ` · ${preset.runtime_variant}` : '';
  const geometry = preset.recommended_geometry ? ` · ${preset.recommended_geometry}` : '';
  const scheduler = preset.scheduler ? ` · scheduler ${preset.scheduler.video_shift}/${preset.scheduler.audio_shift}` : '';
  const scale = preset.default_scale == null ? '' : ` · fixed scale ${preset.default_scale}`;
  const asset = preset.logical_asset ? ` · asset ${preset.logical_asset}` : '';
  $('turbo-preset-details').textContent = `${preset.label} · ${preset.summary}${asset}${scale}${scheduler}${variant}${geometry}`;
}
function updateTurboPresetControls() {
  const preset = selectedTurboPreset();
  const selected = Boolean(preset && preset.id !== 'none');
  if (selected) {
    if (normalStepsBeforeTurbo === null) normalStepsBeforeTurbo = $('steps').value;
    $('steps').value = preset.nfe;
    $('steps').disabled = true;
  } else {
    if (normalStepsBeforeTurbo !== null) {
      $('steps').value = normalStepsBeforeTurbo;
      normalStepsBeforeTurbo = null;
    }
    $('steps').disabled = false;
  }
  updateTurboPresetDetails();
}
function validateDimensionsBeforeLaunch() {
  const width = validateDimension('width');
  const height = validateDimension('height');
  if (width.error || height.error) {
    setStatus(width.error || height.error, true);
    return false;
  }
  return true;
}
function renderBenchmark(benchmark) {
  if (!benchmark || !benchmark.run_id) return '';
  const state = benchmark.success ? '<span class="pill ok">succeeded</span>' : '<span class="pill bad">failed</span>';
  const forwards = benchmark.actual_transformer_forward_count == null ? 'unavailable' : benchmark.actual_transformer_forward_count;
  const peak = benchmark.peak_mlx_memory_bytes == null ? 'unavailable' : `${benchmark.peak_mlx_memory_bytes.toLocaleString()} B`;
  return `<p>${state} · ${Number(benchmark.total_elapsed_seconds || 0).toFixed(2)} s · ${forwards} observed transformer forwards · peak MLX ${peak}</p>`;
}
function previewFor(snapshot) {
  const artifact = snapshot && snapshot.benchmark && snapshot.benchmark.output_artifact;
  if (!artifact || artifact.kind !== 'mp4') return '';
  const root = encodeURIComponent(snapshot.output_root || $('output-root').value);
  return `<h3>Video preview</h3><video controls src="/api/artifact?root=${root}&run=${encodeURIComponent(snapshot.run_id)}&file=${encodeURIComponent(artifact.path.split('/').pop())}"></video>`;
}
function stopStatusPolling() {
  if (statusPollTimer !== null) { clearTimeout(statusPollTimer); statusPollTimer = null; }
}
function scheduleStatusPolling(delay = STATUS_POLL_INTERVAL_MS) {
  stopStatusPolling();
  statusPollTimer = setTimeout(() => { statusPollTimer = null; refreshStatus(); }, delay);
}
function startStatusPolling() {
  stopStatusPolling();
  refreshStatus();
}
function isTerminalSnapshot(snapshot) {
  return Boolean(snapshot && snapshot.run_id && !snapshot.running && TERMINAL_STATUSES.has(snapshot.status));
}
async function refreshStatus() {
  if (statusPollInFlight) return;
  statusPollInFlight = true;
  try {
    const response = await fetch('/api/status'); const snapshot = await response.json();
    if (snapshot.run_id) {
      activeRunId = snapshot.run_id;
      const terminal = isTerminalSnapshot(snapshot);
      const active = Boolean(snapshot.running);
      $('run-meta').textContent = `${snapshot.run_id} · ${snapshot.status}${active ? ' · child process active' : ''}`;
      $('log').textContent = (snapshot.stdout || '') + (snapshot.stderr ? `\n--- stderr ---\n${snapshot.stderr}` : '');
      $('log').scrollTop = $('log').scrollHeight;
      $('benchmark').innerHTML = renderBenchmark(snapshot.benchmark);
      $('preview').innerHTML = terminal ? previewFor(snapshot) : '';
      if (active) { setStatus('Rendering…'); $('render').disabled = true; scheduleStatusPolling(); }
      else if (terminal) {
        $('render').disabled = false;
        setStatus(snapshot.status === 'succeeded' ? 'Finished' : 'Failed', snapshot.status !== 'succeeded');
        refreshHistory();
        stopStatusPolling();
      } else { scheduleStatusPolling(); }
    } else { $('render').disabled = false; stopStatusPolling(); }
  } catch (_) { setStatus('Disconnected', true); scheduleStatusPolling(); }
  finally { statusPollInFlight = false; }
}
function renderHistory(rows) {
  if (!rows.length) return 'No completed or failed runs yet.';
  const body = rows.map(row => {
    const artifact = row.artifact_name ? `<a href="/api/artifact?root=${encodeURIComponent(row.output_root || $('output-root').value)}&run=${encodeURIComponent(row.run_id)}&file=${encodeURIComponent(row.artifact_name)}" target="_blank">${row.artifact_name}</a>` : '—';
    return `<tr><td>${row.timestamp || '—'}</td><td>${row.mode || '—'}</td><td>${row.resolution || '—'}</td><td>${row.steps ?? '—'}</td><td>${row.seed ?? '—'}</td><td>${row.elapsed_seconds == null ? '—' : Number(row.elapsed_seconds).toFixed(2) + ' s'}</td><td>${row.status || '—'}</td><td>${artifact}</td></tr>`;
  }).join('');
  return `<table><thead><tr><th>Timestamp</th><th>Mode</th><th>Resolution</th><th>Steps</th><th>Seed</th><th>Elapsed</th><th>Status</th><th>Artifact</th></tr></thead><tbody>${body}</tbody></table>`;
}
async function refreshHistory() {
  try { const root = encodeURIComponent($('output-root').value); const response = await fetch(`/api/history?root=${root}`); $('history').innerHTML = renderHistory(await response.json()); }
  catch (_) { $('history').textContent = 'History unavailable.'; }
}
async function renderJob() {
  const mode = $('mode').value;
  if (!validateDimensionsBeforeLaunch()) return;
  const form = new FormData();
  form.set('mode', mode); form.set('prompt', $('prompt').value);
  form.set('text_encoder_id', $('text-encoder').value);
  const conditioningArtifactPath = $('conditioning-artifact').value.trim();
  if (conditioningArtifactPath) form.set('conditioning_artifact_path', conditioningArtifactPath);
  form.set('width', $('width').value); form.set('height', $('height').value);
  form.set('steps', $('steps').value); form.set('duration_seconds', $('duration').value); form.set('seed', $('seed').value);
  form.set('output_root', $('output-root').value); form.set('output_name', $('output-name').value);
  form.set('turbo_preset_id', $('turbo-preset').value);
  form.set('additional_loras', JSON.stringify(selectedAdditionalLoras()));
  form.set('turbo_enabled', 'false');
  form.set('turbo_steps', '');
  if (mode !== 'T2V' && $('image1').files[0]) form.append('image1', $('image1').files[0]);
  if (mode === 'FIRST_LAST' && $('image2').files[0]) form.append('image2', $('image2').files[0]);
  $('render').disabled = true; setStatus('Admitting…');
  try {
    const response = await fetch('/api/render', {method: 'POST', body: form}); const result = await response.json();
    if (!response.ok) { setStatus(result.error || 'Rejected', true); $('render').disabled = false; return; }
    activeRunId = result.run_id; $('preview').innerHTML = ''; $('benchmark').innerHTML = ''; $('log').textContent = 'Reserved run ' + result.run_id + '\n'; setStatus('Starting…');
    startStatusPolling();
    refreshHistory();
  } catch (error) { setStatus(String(error), true); $('render').disabled = false; }
}
async function boot() {
  config = await (await fetch('/api/config')).json();
  $('mode').innerHTML = config.modes.map(item => `<option value="${item.id}">${item.label}</option>`).join('');
  $('text-encoder').innerHTML = (config.text_encoders || []).map(item => `<option value="${item.id}" ${item.available ? '' : 'disabled'}>${item.label}</option>`).join('');
  $('text-encoder').value = config.defaults.text_encoder_id || 'canonical-qwen3-vl';
  $('turbo-preset').innerHTML = (config.turbo_presets || []).map(item => `<option value="${item.id}">${item.label}</option>`).join('');
  $('turbo-preset').value = config.defaults.turbo_preset_id || 'none';
  const bounds = config.resolution_contract || {min_dimension: 128, max_dimension: 1344, step: 32};
  for (const axis of ['width', 'height']) {
    $(`${axis}-range`).min = bounds.min_dimension;
    $(`${axis}-range`).max = bounds.max_dimension;
    $(`${axis}-range`).step = bounds.step;
    $(axis).min = bounds.min_dimension;
    $(axis).max = bounds.max_dimension;
    $(axis).value = config.defaults[axis];
    $(`${axis}-range`).value = config.defaults[axis];
    $(`${axis}-range`).oninput = () => syncDimensionFromSlider(axis);
    $(axis).oninput = () => syncDimensionFromNumber(axis);
  }
  $('steps').value = config.defaults.steps; $('duration').value = config.defaults.duration_seconds; $('seed').value = config.defaults.seed;
  additionalLoras = (config.defaults.additional_loras || []).map(row => ({path: row.path || '', scale: row.scale ?? '1.0'}));
  renderAdditionalLoraRows();
  $('conditioning-artifact').value = config.defaults.conditioning_artifact_path || '';
  $('output-root').value = config.defaults.output_root; $('output-name').value = config.defaults.output_name;
  $('mode').onchange = updateMode;
  $('text-encoder').onchange = updateTextEncoderControls;
  $('turbo-preset').onchange = updateTurboPresetControls;
  updateTurboPresetControls(); updateMode(); updateGeometry();
  $('runtime').textContent = `Checkpoint: ${config.runtime.checkpoint_root} · Transformer: ${config.runtime.transformer_path || 'from checkpoint'} · generator: ${config.runtime.generator}`;
  if (!config.runtime.checkpoint_exists || !config.runtime.transformer_exists) { $('runtime-warning').textContent = 'Configured checkpoint or transformer path is not currently readable. Set H3_CHECKPOINT_ROOT and H3_TRANSFORMER before rendering.'; $('runtime-warning').classList.remove('hidden'); }
  else if (config.runtime.transformer_mode !== config.runtime.transformer_required_mode) { $('runtime-warning').textContent = `Render Lab requires ${config.runtime.transformer_required_mode}; the configured transformer is not admitted.`; $('runtime-warning').classList.remove('hidden'); }
  startStatusPolling(); refreshHistory();
}
boot();
</script>
</body></html>"""


CONTROLLER = RenderController(REPO_ROOT)


def send_json(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _submission(handler: BaseHTTPRequestHandler) -> tuple[dict[str, object], list[UploadedImage]]:
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0"))
    if length > 100 * 1024 * 1024:
        raise RenderValidationError("Upload is too large")
    body = handler.rfile.read(length)
    if content_type.startswith("multipart/form-data"):
        # Parse the bounded browser body with the stdlib email MIME parser. This keeps the surface
        # working on Python 3.13+, where cgi.FieldStorage was removed, and leaves runner contracts
        # independent of the HTTP encoding.
        envelope = (
            f"Content-Type: {content_type}\r\n"
            f"MIME-Version: 1.0\r\n"
            f"Content-Length: {length}\r\n\r\n"
        ).encode("utf-8") + body
        parsed = BytesParser(policy=email_default).parsebytes(envelope)
        fields: dict[str, object] = {}
        uploads = []
        for item in parsed.iter_parts():
            name = item.get_param("name", header="content-disposition")
            if not name:
                continue
            filename = item.get_filename()
            payload = item.get_payload(decode=True) or b""
            if filename:
                uploads.append(UploadedImage(str(filename), payload))
            else:
                charset = item.get_content_charset() or "utf-8"
                fields[str(name)] = payload.decode(charset, errors="replace")
        return fields, uploads
    try:
        value = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderValidationError("Request body is not valid JSON or multipart form data") from exc
    if not isinstance(value, dict):
        raise RenderValidationError("Request body must be an object")
    return {str(key): item for key, item in value.items()}, []


def _render_request_from_fields(fields: dict[str, object]) -> RenderRequest:
    """Convert browser/JSON fields into the shared request model."""
    return RenderRequest(
        mode=fields.get("mode", ""),
        prompt=fields.get("prompt") or "",
        text_encoder_id=fields.get("text_encoder_id", "canonical-qwen3-vl"),
        conditioning_artifact_path=fields.get("conditioning_artifact_path") or None,
        resolution_id=fields.get("resolution_id") or None,
        steps=int(fields.get("steps", "")),
        duration_seconds=float(fields.get("duration_seconds", "")),
        seed=int(fields.get("seed", "")),
        output_root=fields.get("output_root", str(DEFAULT_OUTPUT_ROOT.relative_to(REPO_ROOT))),
        output_name=fields.get("output_name", "render.mp4"),
        width=fields.get("width") or None,
        height=fields.get("height") or None,
        lora_enabled=fields.get("lora_enabled", "false"),
        lora_path=fields.get("lora_path") or None,
        lora_scale=fields.get("lora_scale", "1.0"),
        additional_loras=parse_additional_loras_payload(fields.get("additional_loras")),
        turbo_enabled=fields.get("turbo_enabled", "false"),
        turbo_steps=fields.get("turbo_steps") or None,
        turbo_preset_id=fields.get("turbo_preset_id") or None,
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "H3RenderLab/1"

    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            if parsed.path == "/api/config":
                send_json(self, CONTROLLER.config_payload())
            elif parsed.path == "/api/status":
                send_json(self, CONTROLLER.snapshot())
            elif parsed.path == "/api/history":
                query = parse_qs(parsed.query)
                raw_root = query.get("root", [str(DEFAULT_OUTPUT_ROOT)])[0]
                root = resolve_output_root(unquote(raw_root), REPO_ROOT)
                send_json(self, history_rows(root))
            elif parsed.path == "/api/artifact":
                self._send_artifact(parse_qs(parsed.query))
            else:
                send_json(self, {"error": "Not found"}, 404)
        except RenderValidationError as exc:
            send_json(self, {"error": str(exc)}, 400)
        except FileNotFoundError:
            send_json(self, {"error": "Artifact not found"}, 404)

    def _send_artifact(self, query: dict[str, list[str]]) -> None:
        run_id = query.get("run", [""])[0]
        name = Path(query.get("file", [""])[0]).name
        if not run_id or not name or "/" in run_id or "\\" in run_id:
            raise RenderValidationError("Invalid artifact selector")
        root = resolve_output_root(unquote(query.get("root", [str(DEFAULT_OUTPUT_ROOT)])[0]), REPO_ROOT)
        run_dir = root / run_id
        if not run_dir.is_dir() or not (run_dir / "render-config.json").is_file():
            raise FileNotFoundError(run_dir)
        artifact = run_dir / name
        config = json.loads((run_dir / "render-config.json").read_text(encoding="utf-8"))
        allowed = Path(str(config.get("output_path", ""))).name
        if name != allowed or not artifact.is_file():
            raise FileNotFoundError(artifact)
        self._stream_file(artifact)

    def _stream_file(self, path: Path) -> None:
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            value = range_header[6:].split(",", 1)[0]
            left, _, right = value.partition("-")
            if left:
                start = int(left)
            if right:
                end = int(right)
            end = min(end, size - 1)
            if start > end:
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/render":
            send_json(self, {"error": "Not found"}, 404)
            return
        try:
            fields, uploads = _submission(self)
            request = _render_request_from_fields(fields)
            namespace = CONTROLLER.start(request, uploads=uploads)
            send_json(self, {"ok": True, "run_id": namespace.run_id, "run_directory": str(namespace.run_dir)}, 202)
        except RenderBusyError as exc:
            send_json(self, {"error": str(exc)}, 409)
        except (RenderValidationError, ValueError) as exc:
            send_json(self, {"error": str(exc)}, 400)
        except Exception as exc:
            send_json(self, {"error": f"Render Lab admission failed: {exc}"}, 500)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost", "::1"), help="loopback bind address")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the browser after binding")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"H3 Render Lab: {url}", flush=True)
    if args.open:
        subprocess.Popen(["open", url])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
