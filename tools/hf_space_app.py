#!/usr/bin/env python3
"""
HK Neural Tensor Framework - Hugging Face Space Model Transcoder & Inspector App.

Provides bidirectional conversion and interactive inspection across:
- HK Neural Tensor Format (.hk)
- GGUF (.gguf v1, v2, v3)
- Safetensors (.safetensors)

Features:
- Standalone zero-dependency HTTP server with modern dark-themed HTML5/JS dashboard.
- Optional Gradio UI if gradio is installed.
- Zero-copy bitstream transplant for matching quantizations (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K).
- REST API for automated pipeline integration (/api/inspect, /api/convert, /api/health).
"""

import sys
import os
import io
import json
import time
import shutil
import tempfile
import argparse
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List, Tuple

# Ensure python/ package is importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import hk
from hk.gguf_parser import GGUFReaderLight, convert_gguf_to_hk, export_hk_to_gguf
from hk.torch import HKFile, safe_open, save_file
from hk.native import is_native_available


# ---------------------------------------------------------------------------
# Core Inspector & Transcoder Engine
# ---------------------------------------------------------------------------

def inspect_model_file(file_path: str) -> Dict[str, Any]:
    """Inspects any .hk, .gguf, or .safetensors model file and returns detailed stats."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)

    # Detect format via magic bytes
    with open(file_path, "rb") as f:
        magic_bytes = f.read(8)

    format_detected = "Unknown"
    metadata = {}
    tensors = []

    if len(magic_bytes) >= 4 and magic_bytes[:4] == b"HK01":
        format_detected = "HK Container (v1.0)"
        try:
            reader = safe_open(file_path, framework="pt")
            metadata = reader.metadata()
            tensor_names = reader.keys()
            for name in tensor_names:
                t = reader.get_tensor(name)
                dtype_str = str(t.dtype).replace("torch.", "")
                element_size = t.element_size()
                num_el = t.numel()
                tensors.append({
                    "name": name,
                    "shape": list(t.shape),
                    "dtype": dtype_str,
                    "num_elements": num_el,
                    "byte_size": num_el * element_size,
                })
            del reader
        except Exception as ex:
            metadata = {"error": str(ex)}

    elif len(magic_bytes) >= 4 and magic_bytes[:4] == b"GGUF":
        reader = GGUFReaderLight(file_path)
        format_detected = f"GGUF (v{reader.version})"
        # Clean metadata for JSON serialization
        for k, v in reader.metadata.items():
            if isinstance(v, (int, float, str, bool)):
                metadata[k] = v
            elif isinstance(v, list) and len(v) <= 32:
                metadata[k] = v
            elif isinstance(v, list):
                metadata[k] = f"Array[{len(v)} items: {v[:3]}...]"
            else:
                metadata[k] = str(v)

        for name, desc in reader.tensors.items():
            tensors.append({
                "name": name,
                "shape": desc.shape,
                "dtype": f"GGML_TYPE_{desc.ggml_type}",
                "num_elements": desc.size,
                "byte_size": desc.size,
                "offset": desc.offset,
            })

    else:
        # Check Safetensors (first 8 bytes is header length in little endian uint64)
        try:
            import struct
            header_len = struct.unpack("<Q", magic_bytes)[0]
            if 0 < header_len < file_size:
                with open(file_path, "rb") as f:
                    f.seek(8)
                    header_json = json.loads(f.read(header_len).decode("utf-8"))
                format_detected = "Safetensors"
                for k, v in header_json.items():
                    if k == "__metadata__":
                        metadata = v
                    else:
                        tensors.append({
                            "name": k,
                            "shape": v.get("shape", []),
                            "dtype": v.get("dtype", ""),
                            "data_offsets": v.get("data_offsets", []),
                            "byte_size": v.get("data_offsets", [0, 0])[1] - v.get("data_offsets", [0, 0])[0] if "data_offsets" in v else 0,
                        })
        except Exception:
            format_detected = "Unknown / Binary"

    return {
        "filename": file_name,
        "format": format_detected,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "tensor_count": len(tensors),
        "metadata_count": len(metadata),
        "metadata": metadata,
        "tensors": tensors,
    }


def transcode_model(input_path: str, output_path: str, target_format: str) -> Dict[str, Any]:
    """Transcodes an input model file to target_format ('hk', 'gguf', or 'safetensors')."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    t0 = time.perf_counter()
    input_size = os.path.getsize(input_path)
    target_format = target_format.lower().strip()

    # Detect source format
    with open(input_path, "rb") as f:
        magic = f.read(4)

    is_gguf = (magic == b"GGUF")
    is_hk = (magic == b"HK01")

    if target_format == "hk":
        if is_gguf:
            convert_gguf_to_hk(input_path, output_path)
        elif is_hk:
            shutil.copy2(input_path, output_path)
        else:
            # Assume safetensors or PyTorch
            from safetensors.torch import load_file as st_load
            tensors = st_load(input_path)
            save_file(tensors, output_path)

    elif target_format == "gguf":
        if is_hk:
            export_hk_to_gguf(input_path, output_path)
        elif is_gguf:
            shutil.copy2(input_path, output_path)
        else:
            # Safetensors -> HK temp -> GGUF
            from safetensors.torch import load_file as st_load
            tensors = st_load(input_path)
            with tempfile.NamedTemporaryFile(suffix=".hk", delete=False) as tmp_hk:
                tmp_hk_path = tmp_hk.name
            try:
                save_file(tensors, tmp_hk_path)
                export_hk_to_gguf(tmp_hk_path, output_path)
            finally:
                if os.path.exists(tmp_hk_path):
                    os.remove(tmp_hk_path)

    elif target_format == "safetensors":
        from safetensors.torch import save_file as st_save
        if is_hk:
            reader = safe_open(input_path, framework="pt")
            tensors = {k: reader.get_tensor(k) for k in reader.keys()}
            st_save(tensors, output_path)
            del reader
        elif is_gguf:
            # GGUF -> HK temp -> Safetensors
            with tempfile.NamedTemporaryFile(suffix=".hk", delete=False) as tmp_hk:
                tmp_hk_path = tmp_hk.name
            try:
                convert_gguf_to_hk(input_path, tmp_hk_path)
                reader = safe_open(tmp_hk_path, framework="pt")
                tensors = {k: reader.get_tensor(k) for k in reader.keys()}
                st_save(tensors, output_path)
                del reader
            finally:
                if os.path.exists(tmp_hk_path):
                    os.remove(tmp_hk_path)
        else:
            shutil.copy2(input_path, output_path)
    else:
        raise ValueError(f"Unsupported target format: {target_format}. Supported: hk, gguf, safetensors")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    output_size = os.path.getsize(output_path)

    return {
        "success": True,
        "input_file": os.path.basename(input_path),
        "output_file": os.path.basename(output_path),
        "target_format": target_format.upper(),
        "input_size_mb": round(input_size / (1024 * 1024), 2),
        "output_size_mb": round(output_size / (1024 * 1024), 2),
        "elapsed_ms": round(elapsed_ms, 2),
        "ratio": round(output_size / (input_size if input_size > 0 else 1), 3),
        "native_engine": is_native_available(),
    }


# ---------------------------------------------------------------------------
# Embedded HTML5 Web Interface
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HK Neural Tensor Framework - Model Transcoder & Hub</title>
  <style>
    :root {
      --bg-color: #0d1117;
      --card-bg: #161b22;
      --border-color: #30363d;
      --text-main: #c9d1d9;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-glow: rgba(88, 166, 255, 0.15);
      --success: #3fb950;
      --warning: #d29922;
      --purple: #bc8cff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background: var(--bg-color); color: var(--text-main); min-height: 100vh; padding: 24px; }
    .container { max-width: 1100px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 32px; padding: 20px 0; border-bottom: 1px solid var(--border-color); }
    h1 { font-size: 2.2rem; color: #fff; margin-bottom: 8px; display: flex; align-items: center; justify-content: center; gap: 12px; }
    .badge { font-size: 0.8rem; background: var(--purple); color: #fff; padding: 4px 10px; border-radius: 12px; font-weight: 600; text-transform: uppercase; }
    .subtitle { color: var(--text-muted); font-size: 1.05rem; }
    
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
    @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    
    .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .card h2 { font-size: 1.25rem; color: #fff; margin-bottom: 16px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    
    .upload-zone { border: 2px dashed var(--border-color); border-radius: 6px; padding: 28px; text-align: center; cursor: pointer; transition: all 0.2s; background: rgba(255,255,255,0.01); }
    .upload-zone:hover { border-color: var(--accent); background: var(--accent-glow); }
    .upload-zone p { color: var(--text-muted); margin-top: 8px; font-size: 0.9rem; }
    
    .btn { background: var(--accent); color: #fff; border: none; padding: 10px 18px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: background 0.2s; display: inline-flex; align-items: center; gap: 6px; }
    .btn:hover { background: #4791eb; }
    .btn:disabled { background: var(--border-color); cursor: not-allowed; opacity: 0.6; }
    .btn-success { background: var(--success); }
    .btn-success:hover { background: #2ea043; }
    
    select, input[type="text"] { background: var(--bg-color); border: 1px solid var(--border-color); color: var(--text-main); padding: 8px 12px; border-radius: 6px; width: 100%; margin-bottom: 14px; font-size: 0.95rem; }
    label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 6px; font-weight: 500; }
    
    .stats-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.95rem; }
    .stats-label { color: var(--text-muted); }
    .stats-val { font-weight: 600; color: #fff; font-family: monospace; }
    
    .scroll-table { max-height: 280px; overflow-y: auto; margin-top: 14px; border: 1px solid var(--border-color); border-radius: 6px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left; }
    th { background: #21262d; color: var(--text-muted); padding: 8px 12px; position: sticky; top: 0; }
    td { padding: 8px 12px; border-bottom: 1px solid var(--border-color); font-family: monospace; }
    tr:hover td { background: rgba(255,255,255,0.02); }
    
    .log-box { background: #000; border: 1px solid var(--border-color); border-radius: 6px; padding: 12px; font-family: monospace; font-size: 0.85rem; color: #3fb950; min-height: 80px; white-space: pre-wrap; word-break: break-all; margin-top: 14px; }
    
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
    .tab { padding: 6px 14px; border-radius: 6px; cursor: pointer; background: transparent; border: 1px solid var(--border-color); color: var(--text-muted); font-size: 0.9rem; }
    .tab.active { background: #21262d; color: #fff; border-color: var(--accent); }
    
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>HK Neural Tensor Framework <span class="badge">HF Space v1.0</span></h1>
      <p class="subtitle">Universal Bidirectional Transcoder & Deep Container Inspector for GGUF, HK, and Safetensors</p>
    </header>

    <div class="grid">
      <!-- Ingestion & Conversion Card -->
      <div class="card">
        <h2>Transcode Model <span id="status-badge" style="font-size:0.75rem; color:var(--accent);">Ready</span></h2>
        
        <label>Input Model File (.gguf, .hk, .safetensors)</label>
        <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <input type="file" id="file-input" class="hidden" onchange="handleFileSelect(event)">
          <strong id="file-name-label">Drag & Drop Model or Click to Select</strong>
          <p>Zero-copy bitstream transplant for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K</p>
        </div>

        <div style="margin-top: 16px;">
          <label>Target Container Format</label>
          <select id="target-format">
            <option value="hk" selected>HK Container (.hk) - Superior Super-Set</option>
            <option value="gguf">GGUF Container (.gguf) - llama.cpp / ggml</option>
            <option value="safetensors">Safetensors (.safetensors) - Hugging Face Hub</option>
          </select>
        </div>

        <div style="display:flex; gap: 10px; margin-top: 8px;">
          <button class="btn btn-success" id="btn-convert" onclick="triggerConvert()" disabled>Transcode Model</button>
          <button class="btn" id="btn-inspect" onclick="triggerInspect()" disabled>Inspect Tensors</button>
        </div>

        <div class="log-box" id="transcode-log">Awaiting model input...</div>
        <div id="download-area" style="margin-top: 14px;" class="hidden">
          <a id="download-link" class="btn" style="background:var(--purple); text-decoration:none;" download>Download Converted Model</a>
        </div>
      </div>

      <!-- Overview Stats Card -->
      <div class="card">
        <h2>Container Audit</h2>
        <div id="stats-container">
          <div class="stats-row"><span class="stats-label">Container Format:</span><span class="stats-val" id="stat-format">-</span></div>
          <div class="stats-row"><span class="stats-label">File Size:</span><span class="stats-val" id="stat-size">-</span></div>
          <div class="stats-row"><span class="stats-label">Tensors Count:</span><span class="stats-val" id="stat-tensors">-</span></div>
          <div class="stats-row"><span class="stats-label">Metadata Keys:</span><span class="stats-val" id="stat-meta">-</span></div>
          <div class="stats-row"><span class="stats-label">Native Zig Acceleration:</span><span class="stats-val" id="stat-zig" style="color:var(--success)">Available</span></div>
        </div>

        <div style="margin-top: 20px;">
          <div class="tabs">
            <div class="tab active" onclick="switchTab('tab-tensors')">Tensors Table</div>
            <div class="tab" onclick="switchTab('tab-metadata')">Metadata JSON</div>
          </div>
          
          <div id="tab-tensors" class="scroll-table">
            <table>
              <thead>
                <tr><th>Tensor Name</th><th>Shape</th><th>Type</th><th>Size</th></tr>
              </thead>
              <tbody id="tensor-rows">
                <tr><td colspan="4" style="text-align:center; color:var(--text-muted);">No model loaded</td></tr>
              </tbody>
            </table>
          </div>

          <div id="tab-metadata" class="scroll-table hidden">
            <pre id="metadata-json" style="padding:10px; font-size:0.8rem; color:#fff; font-family:monospace;"></pre>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let selectedFile = null;
    let serverTempPath = null;

    function handleFileSelect(e) {
      const file = e.target.files[0];
      if (!file) return;
      selectedFile = file;
      document.getElementById('file-name-label').innerText = `${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)`;
      document.getElementById('btn-convert').disabled = false;
      document.getElementById('btn-inspect').disabled = false;
      log(`Selected file: ${file.name}`);
    }

    const dropZone = document.getElementById('drop-zone');
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#58a6ff'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#30363d'; });
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = '#30363d';
      if (e.dataTransfer.files.length) {
        document.getElementById('file-input').files = e.dataTransfer.files;
        handleFileSelect({ target: { files: e.dataTransfer.files } });
      }
    });

    function log(msg) {
      const el = document.getElementById('transcode-log');
      el.innerText += "\\n" + msg;
      el.scrollTop = el.scrollHeight;
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-tensors').classList.add('hidden');
      document.getElementById('tab-metadata').classList.add('hidden');
      document.getElementById(tabId).classList.remove('hidden');
    }

    async function triggerInspect() {
      if (!selectedFile) return;
      log("Uploading and inspecting container...");
      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const res = await fetch("/api/inspect", { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) { log("Error: " + data.error); return; }

        document.getElementById('stat-format').innerText = data.format;
        document.getElementById('stat-size').innerText = data.file_size_mb + " MB";
        document.getElementById('stat-tensors').innerText = data.tensor_count;
        document.getElementById('stat-meta').innerText = data.metadata_count;

        // Render Tensors
        const tbody = document.getElementById('tensor-rows');
        tbody.innerHTML = "";
        data.tensors.forEach(t => {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${t.name}</td><td>[${t.shape.join(', ')}]</td><td>${t.dtype}</td><td>${(t.byte_size/1024).toFixed(1)} KB</td>`;
          tbody.appendChild(tr);
        });

        // Render Metadata
        document.getElementById('metadata-json').innerText = JSON.stringify(data.metadata, null, 2);
        log(`Inspected ${data.tensor_count} tensors across ${data.format} container successfully.`);
      } catch (err) {
        log("Inspect failed: " + err);
      }
    }

    async function triggerConvert() {
      if (!selectedFile) return;
      const targetFmt = document.getElementById('target-format').value;
      log(`Starting transcode pipeline to [${targetFmt.toUpperCase()}]...`);
      document.getElementById('status-badge').innerText = "Transcoding...";
      document.getElementById('btn-convert').disabled = true;

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("target_format", targetFmt);

      try {
        const res = await fetch("/api/convert", { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) {
          log("Conversion error: " + data.error);
          document.getElementById('status-badge').innerText = "Error";
          return;
        }

        log(`[SUCCESS] Transcoded in ${data.elapsed_ms} ms!`);
        log(`Input: ${data.input_size_mb} MB -> Output: ${data.output_size_mb} MB (ratio: ${data.ratio})`);
        
        const dlArea = document.getElementById('download-area');
        const dlLink = document.getElementById('download-link');
        dlLink.href = `/api/download?file=${encodeURIComponent(data.output_file)}`;
        dlLink.download = data.output_file;
        dlLink.innerText = `Download ${data.output_file} (${data.output_size_mb} MB)`;
        dlArea.classList.remove('hidden');
        document.getElementById('status-badge').innerText = "Complete";
      } catch (err) {
        log("Conversion request failed: " + err);
        document.getElementById('status-badge').innerText = "Failed";
      } finally {
        document.getElementById('btn-convert').disabled = false;
      }
    }
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Zero-Dependency HTTP Request Handler
# ---------------------------------------------------------------------------

class HKSpaceHTTPHandler(BaseHTTPRequestHandler):
    temp_dir = tempfile.mkdtemp(prefix="hk_space_")

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._set_headers(200, "text/html; charset=utf-8")
            self.wfile.write(INDEX_HTML.encode("utf-8"))
        elif path == "/api/health":
            self._set_headers(200)
            status = {
                "status": "healthy",
                "framework": "HK Neural Tensor Framework",
                "native_engine": is_native_available(),
                "supported_formats": ["hk", "gguf", "safetensors"],
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
        elif path == "/api/download":
            query = urllib.parse.parse_qs(parsed.query)
            fname = query.get("file", [""])[0]
            # Prevent path traversal
            safe_name = os.path.basename(fname)
            full_path = os.path.join(self.temp_dir, safe_name)
            if safe_name and os.path.exists(full_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
                self.send_header("Content-Length", str(os.path.getsize(full_path)))
                self.end_headers()
                with open(full_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "File not found"}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")

        if path == "/api/inspect":
            self._handle_api_inspect(content_type, content_length)
        elif path == "/api/convert":
            self._handle_api_convert(content_type, content_length)
        else:
            self._set_headers(404)
            self.wfile.write(b"Not Found")

    def _parse_multipart_or_file(self, content_type: str, content_length: int) -> Tuple[str, Dict[str, str]]:
        """Parses multipart form data or json file payload into a saved file."""
        if "multipart/form-data" in content_type:
            boundary = content_type.split("boundary=")[1].strip()
            if boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]
            raw_body = self.rfile.read(content_length)
            
            boundary_bytes = ("--" + boundary).encode("latin-1")
            parts = raw_body.split(boundary_bytes)
            
            saved_file_path = ""
            form_fields = {}

            for part in parts:
                if not part or part == b"--\r\n" or part == b"--":
                    continue
                header_data, _, body_data = part.partition(b"\r\n\r\n")
                if not body_data:
                    continue
                if body_data.endswith(b"\r\n"):
                    body_data = body_data[:-2]
                
                header_text = header_data.decode("latin-1", errors="replace")
                if 'filename="' in header_text:
                    fn_start = header_text.find('filename="') + 10
                    fn_end = header_text.find('"', fn_start)
                    filename = os.path.basename(header_text[fn_start:fn_end])
                    saved_file_path = os.path.join(self.temp_dir, filename)
                    with open(saved_file_path, "wb") as out_f:
                        out_f.write(body_data)
                elif 'name="' in header_text:
                    name_start = header_text.find('name="') + 6
                    name_end = header_text.find('"', name_start)
                    name = header_text[name_start:name_end]
                    form_fields[name] = body_data.decode("utf-8", errors="replace").strip()

            return saved_file_path, form_fields
        elif "application/json" in content_type:
            raw_body = self.rfile.read(content_length)
            data = json.loads(raw_body.decode("utf-8"))
            local_path = data.get("file_path", "")
            return local_path, data
        else:
            raise ValueError("Unsupported Content-Type. Use multipart/form-data or application/json")

    def _handle_api_inspect(self, content_type: str, content_length: int):
        try:
            file_path, _ = self._parse_multipart_or_file(content_type, content_length)
            if not file_path or not os.path.exists(file_path):
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "No valid file uploaded or path specified"}).encode("utf-8"))
                return

            result = inspect_model_file(file_path)
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _handle_api_convert(self, content_type: str, content_length: int):
        try:
            file_path, fields = self._parse_multipart_or_file(content_type, content_length)
            if not file_path or not os.path.exists(file_path):
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "No valid file uploaded or path specified"}).encode("utf-8"))
                return

            target_fmt = fields.get("target_format", "hk").lower().strip()
            base_stem = Path(file_path).stem
            output_filename = f"{base_stem}_transcoded.{target_fmt}"
            output_path = os.path.join(self.temp_dir, output_filename)

            result = transcode_model(file_path, output_path, target_fmt)
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


# ---------------------------------------------------------------------------
# Optional Gradio Interface Support
# ---------------------------------------------------------------------------

def build_gradio_app():
    """Builds a Hugging Face Space Gradio interface if gradio is installed."""
    try:
        import gradio as gr
    except ImportError:
        return None

    def gradio_inspect(file_obj):
        if not file_obj:
            return "No file selected", "", []
        info = inspect_model_file(file_obj.name)
        overview = f"Format: {info['format']}\\nSize: {info['file_size_mb']} MB\\nTensors: {info['tensor_count']}\\nMetadata Keys: {info['metadata_count']}"
        meta_str = json.dumps(info['metadata'], indent=2)
        tensor_rows = [[t['name'], str(t['shape']), t['dtype'], f"{t['byte_size']/1024:.1f} KB"] for t in info['tensors']]
        return overview, meta_str, tensor_rows

    def gradio_convert(file_obj, target_fmt):
        if not file_obj:
            return None, "No file uploaded."
        out_dir = tempfile.mkdtemp(prefix="hk_gradio_")
        target_fmt = target_fmt.lower()
        out_name = f"{Path(file_obj.name).stem}_transcoded.{target_fmt}"
        out_path = os.path.join(out_dir, out_name)
        res = transcode_model(file_obj.name, out_path, target_fmt)
        summary = (
            f"Transcode Complete!\\n"
            f"Target: {res['target_format']}\\n"
            f"Elapsed: {res['elapsed_ms']} ms\\n"
            f"Original Size: {res['input_size_mb']} MB -> Transcoded: {res['output_size_mb']} MB\\n"
            f"Size Ratio: {res['ratio']}\\n"
            f"Native Engine: {res['native_engine']}"
        )
        return out_path, summary

    with gr.Blocks(title="HK Neural Tensor Hub & Transcoder", theme="default") as demo:
        gr.Markdown("# HK Neural Tensor Framework: Hub & Transcoder")
        gr.Markdown("Direct GGUF, HK, and Safetensors inspection and zero-copy bitstream transcoding.")

        with gr.Tab("Transcode Model"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Input Model (.gguf, .hk, .safetensors)")
                    format_choice = gr.Dropdown(choices=["hk", "gguf", "safetensors"], value="hk", label="Target Format")
                    convert_btn = gr.Button("Transcode Now", variant="primary")
                with gr.Column():
                    file_output = gr.File(label="Download Transcoded Container")
                    status_output = gr.Textbox(label="Transcoding Report", lines=6)
            convert_btn.click(gradio_convert, inputs=[file_input, format_choice], outputs=[file_output, status_output])

        with gr.Tab("Inspect Model"):
            with gr.Row():
                with gr.Column():
                    inspect_file = gr.File(label="Model File to Inspect")
                    inspect_btn = gr.Button("Inspect Tensors & Metadata")
                with gr.Column():
                    audit_box = gr.Textbox(label="Overview Audit", lines=4)
                    meta_box = gr.Textbox(label="Metadata JSON", lines=6)
            tensor_table = gr.Dataframe(headers=["Name", "Shape", "Dtype", "Size"], label="Tensor Directory")
            inspect_btn.click(gradio_inspect, inputs=[inspect_file], outputs=[audit_box, meta_box, tensor_table])

    return demo


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HK Neural Tensor Framework HF Space App")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    args = parser.parse_args()

    demo = build_gradio_app()
    if demo is not None:
        print(f"[HK Space] Gradio detected. Launching Gradio server on {args.host}:{args.port}...")
        demo.launch(server_name=args.host, server_port=args.port)
    else:
        print(f"[HK Space] Gradio not installed. Starting built-in HTTP server on {args.host}:{args.port}...")
        server = HTTPServer((args.host, args.port), HKSpaceHTTPHandler)
        print(f"[HK Space] Ready! Visit: http://127.0.0.1:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[HK Space] Shutting down.")
            server.server_close()


if __name__ == "__main__":
    main()
