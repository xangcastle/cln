
import argparse
import sys

import torch

sys.path.insert(0, ".")

from cln.chat import CLNChat

GPT2_VARIANTS = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}



def parse_args():
    p = argparse.ArgumentParser(
        description="Chat interactivo con CLN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", "-m", default="microsoft/Phi-3-mini-4k-instruct",
                   help="ID del modelo HuggingFace o variante GPT-2 (default: Phi-3-mini)")
    p.add_argument("--memory", "-M", default="cln_memory.pt",
                   help="Ruta para estado plástico (default: cln_memory.pt)")
    p.add_argument("--no-plastic", action="store_true",
                   help="Desactiva aprendizaje online")
    p.add_argument("--temp", "-t", type=float, default=0.7,
                   help="Temperatura de muestreo (default: 0.7)")
    p.add_argument("--tokens", "-T", type=int, default=300,
                   help="Máximo de tokens por respuesta (default: 300)")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "float32", "bfloat16"],
                   help="Dtype para cargar el modelo HF (default: float16)")
    p.add_argument("--device", default=None,
                   help="Dispositivo: cpu | mps | cuda (default: auto-detect)")
    p.add_argument("--web", action="store_true",
                   help="Lanza interfaz web en el navegador (requiere flask)")
    p.add_argument("--port", "-p", type=int, default=5001,
                   help="Puerto para la interfaz web (default: 5001)")
    return p.parse_args()



_HTML_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CLN Chat</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  height: 100vh;
  display: flex;
  overflow: hidden;
}

/* ── Sidebar ── */
.sidebar {
  width: 272px;
  min-width: 272px;
  background: #1e293b;
  border-right: 1px solid #334155;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}
.sidebar-section {
  padding: 16px;
  border-bottom: 1px solid #334155;
}
.sidebar-section:last-child { border-bottom: none; }

.logo { font-size: 22px; font-weight: 700; color: #38bdf8; letter-spacing: 0.05em; }
.logo-sub { font-size: 11px; color: #64748b; margin-top: 2px; }
.model-name { font-size: 12px; color: #94a3b8; margin-top: 6px; word-break: break-all; }

.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat-box {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 8px 10px;
}
.stat-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-val { font-size: 15px; font-weight: 600; color: #38bdf8; margin-top: 2px; }

.slider-row { margin-bottom: 10px; }
.slider-row:last-child { margin-bottom: 0; }
.slider-label { display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; margin-bottom: 5px; }
.slider-label span { color: #e2e8f0; font-weight: 500; }
input[type="range"] {
  width: 100%; -webkit-appearance: none;
  height: 4px; border-radius: 2px; background: #334155; outline: none; cursor: pointer;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 14px; height: 14px;
  border-radius: 50%; background: #38bdf8; cursor: pointer;
}

.btn {
  display: block; width: 100%; background: #334155; color: #e2e8f0;
  border: none; padding: 9px 12px; border-radius: 7px; cursor: pointer;
  font-size: 13px; text-align: left; transition: background 0.15s; margin-bottom: 6px;
}
.btn:last-child { margin-bottom: 0; }
.btn:hover { background: #475569; }
.btn:active { background: #64748b; }
.btn-danger { color: #fca5a5; }
.btn-danger:hover { background: #7f1d1d; color: #fecaca; }
.btn-toggle { color: #86efac; }
.btn-toggle.off { color: #fca5a5; }

/* ── Main ── */
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

.header {
  padding: 14px 24px; border-bottom: 1px solid #334155;
  display: flex; align-items: center; gap: 10px; flex-shrink: 0;
}
.header h1 { font-size: 17px; font-weight: 600; color: #f1f5f9; }
.badge {
  font-size: 11px; padding: 3px 9px; border-radius: 20px;
  background: #0e4429; color: #3fb950; font-weight: 500;
}
.badge.off { background: #3d1515; color: #f87171; }

.chat-area {
  flex: 1; overflow-y: auto; padding: 20px 24px;
  display: flex; flex-direction: column; gap: 14px;
}
.chat-area::-webkit-scrollbar { width: 6px; }
.chat-area::-webkit-scrollbar-track { background: transparent; }
.chat-area::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

.message {
  max-width: 78%; padding: 11px 15px; border-radius: 14px;
  line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; font-size: 14px;
}
.message.user {
  align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px;
}
.message.bot {
  align-self: flex-start; background: #1e293b; color: #e2e8f0;
  border: 1px solid #334155; border-bottom-left-radius: 4px;
}
.message.system {
  align-self: center; background: #0f172a; color: #64748b; border: 1px solid #1e293b;
  font-size: 12px; max-width: 90%; font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
}

.typing-dot {
  display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: #64748b; margin: 0 2px; animation: blink 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
  0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
  40% { opacity: 1; transform: scale(1); }
}

.input-wrapper { flex-shrink: 0; border-top: 1px solid #334155; padding: 14px 24px; }
.hint { font-size: 11px; color: #475569; margin-bottom: 8px; }
.input-row { display: flex; gap: 10px; align-items: flex-end; }
#msg {
  flex: 1; background: #1e293b; color: #e2e8f0; border: 1px solid #334155;
  border-radius: 12px; padding: 11px 15px; font-size: 14px; resize: none; outline: none;
  min-height: 46px; max-height: 180px; line-height: 1.5; font-family: inherit;
  transition: border-color 0.15s;
}
#msg:focus { border-color: #38bdf8; }
#msg::placeholder { color: #475569; }
#send-btn {
  background: #2563eb; color: #fff; border: none; padding: 11px 18px;
  border-radius: 12px; font-size: 14px; font-weight: 500; cursor: pointer;
  transition: background 0.15s; white-space: nowrap; min-height: 46px;
}
#send-btn:hover { background: #1d4ed8; }
#send-btn:disabled { background: #334155; color: #64748b; cursor: default; }
</style>
</head>
<body>

<div class="sidebar">
  <div class="sidebar-section">
    <div class="logo">CLN</div>
    <div class="logo-sub">Continuous Liquid Network</div>
    <div class="model-name" id="model-name">cargando...</div>
  </div>

  <div class="sidebar-section">
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-label">‖ΔW‖</div>
        <div class="stat-val" id="s-norm">—</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Turnos</div>
        <div class="stat-val" id="s-turns">0</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Capas</div>
        <div class="stat-val" id="s-layers">—</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Modo</div>
        <div class="stat-val" id="s-mode" style="font-size:12px">—</div>
      </div>
    </div>
    <div style="margin-top:10px">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#64748b;margin-bottom:4px">
        <span>Contexto</span>
        <span id="ctx-label">— tok</span>
      </div>
      <div style="background:#0f172a;border-radius:4px;height:6px;overflow:hidden;border:1px solid #334155">
        <div id="ctx-bar" style="height:100%;width:0%;background:#38bdf8;border-radius:4px;transition:width 0.3s"></div>
      </div>
    </div>
  </div>

  <div class="sidebar-section">
    <div class="slider-row">
      <div class="slider-label">Temperatura <span id="temp-val">0.7</span></div>
      <input type="range" id="temp-slider" min="0.05" max="2.0" step="0.05" value="0.7">
    </div>
    <div class="slider-row">
      <div class="slider-label">Max tokens <span id="tokens-val">300</span></div>
      <input type="range" id="tokens-slider" min="50" max="1000" step="50" value="300">
    </div>
  </div>

  <div class="sidebar-section">
    <button class="btn" onclick="runCmd('/stats')">📊 Stats por capa</button>
    <button class="btn" onclick="runCmd('/save')">💾 Guardar memoria</button>
    <button class="btn" onclick="runCmd('/load')">📂 Cargar memoria</button>
    <button class="btn" onclick="runCmd('/consolidate')">🧠 Consolidar EWC</button>
    <button class="btn" onclick="runCmd('/clear')">🗑 Limpiar historial</button>
    <button class="btn btn-danger" onclick="runCmd('/reset')">⚡ Reset ΔW</button>
    <button class="btn btn-toggle" id="plastic-btn" onclick="togglePlastic()">🔵 Plasticidad: ON</button>
    <hr style="border:0; border-bottom:1px solid #334155; margin: 10px 0;">
    <input type="file" id="file-upload" accept=".txt,.md,.py,.json" style="display: none">
    <button class="btn" style="background:#0284c7; color:#fff;" onclick="document.getElementById('file-upload').click()">📚 Enseñar documento</button>
  </div>
</div>

<div class="main">
  <div class="header">
    <h1 id="header-model">CLN Chat</h1>
    <span class="badge" id="learn-badge">diferido</span>
  </div>
  <div class="chat-area" id="chat"></div>
  <div class="input-wrapper">
    <div class="hint">Enter envía &nbsp;·&nbsp; Shift+Enter nueva línea</div>
    <div class="input-row">
      <textarea id="msg" placeholder="Escribe tu mensaje..." rows="1"></textarea>
      <button id="send-btn" onclick="sendMessage()">Enviar ↑</button>
    </div>
  </div>
</div>

<script>
const chatArea   = document.getElementById('chat');
const msgInput   = document.getElementById('msg');
const sendBtn    = document.getElementById('send-btn');
const plasticBtn = document.getElementById('plastic-btn');
const learnBadge = document.getElementById('learn-badge');

let isGenerating = false;
let plasticOn    = true;

function addMessage(text, cls, streaming = false) {
  const div = document.createElement('div');
  div.className = 'message ' + cls;
  if (streaming) {
    div.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  } else {
    div.textContent = text;
  }
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function updateStats(s) {
  if (!s) return;
  document.getElementById('s-norm').textContent =
    s.delta_w_norm < 0.0001 ? s.delta_w_norm.toExponential(2) : s.delta_w_norm.toFixed(5);
  document.getElementById('s-turns').textContent  = s.turns;
  document.getElementById('s-layers').textContent = s.plastic_layers;
  document.getElementById('s-mode').textContent   = s.mode;

  learnBadge.textContent = s.mode;
  learnBadge.className   = 'badge' + (s.mode === 'OFF' ? ' off' : '');

  if (typeof s.plastic === 'boolean') {
    plasticOn = s.plastic;
    plasticBtn.textContent = plasticOn ? '🔵 Plasticidad: ON' : '⭕ Plasticidad: OFF';
    plasticBtn.className   = 'btn btn-toggle' + (plasticOn ? '' : ' off');
  }
  if (typeof s.temp === 'number') {
    document.getElementById('temp-val').textContent  = s.temp.toFixed(2);
    document.getElementById('temp-slider').value     = s.temp;
  }
  if (typeof s.max_tokens === 'number') {
    document.getElementById('tokens-val').textContent = s.max_tokens;
    document.getElementById('tokens-slider').value    = s.max_tokens;
  }
  if (s.ctx_tokens != null && s.max_ctx_tokens) {
    const pct = Math.min(100, (s.ctx_tokens / s.max_ctx_tokens) * 100);
    document.getElementById('ctx-label').textContent = s.ctx_tokens + ' / ' + s.max_ctx_tokens + ' tok';
    const bar = document.getElementById('ctx-bar');
    bar.style.width      = pct + '%';
    bar.style.background = pct > 85 ? '#ef4444' : pct > 65 ? '#f59e0b' : '#38bdf8';
  }
}

function setGenerating(v) {
  isGenerating = v; sendBtn.disabled = v; msgInput.disabled = v;
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text || isGenerating) return;

  addMessage(text, 'user');
  msgInput.value = '';
  autoResize();
  setGenerating(true);

  const botDiv = addMessage('', 'bot', true);
  let responseText = '';

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch(e) { continue; }
        if (data.token !== undefined) {
          responseText += data.token;
          botDiv.textContent = responseText;
          chatArea.scrollTop = chatArea.scrollHeight;
        }
        if (data.done) {
          botDiv.textContent = responseText.trim() || '(sin respuesta)';
          chatArea.scrollTop = chatArea.scrollHeight;
          if (data.stats) updateStats(data.stats);
        }
        if (data.error) {
          botDiv.textContent = '⚠️ Error: ' + data.error;
          botDiv.style.color = '#fca5a5';
        }
      }
    }
  } catch(e) {
    botDiv.textContent = '⚠️ Error de conexión: ' + e.message;
    botDiv.style.color = '#fca5a5';
  } finally {
    setGenerating(false);
  }
}

async function runCmd(cmd) {
  if (isGenerating) return;
  const resp = await fetch('/command', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cmd}),
  });
  const data = await resp.json();
  if (data.text) addMessage(data.text, 'system');
  if (data.stats) updateStats(data.stats);
}

async function togglePlastic() {
  await runCmd('/plastic ' + (plasticOn ? 'off' : 'on'));
}

document.getElementById('file-upload').addEventListener('change', async function(e) {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = ''; // reset
  if (isGenerating) return;
  
  const botDiv = addMessage('Subiendo y estudiando ' + file.name + '...', 'system', false);
  setGenerating(true);
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let responseText = 'Estudiando...\\n';
    
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch(err) { continue; }
        if (data.token !== undefined) {
          responseText += data.token;
          const displayLines = responseText.split('\\r');
          botDiv.textContent = displayLines[displayLines.length - 1];
          chatArea.scrollTop = chatArea.scrollHeight;
        }
        if (data.done) {
          botDiv.textContent += '\\n' + (data.text || 'Completado.');
          if (data.stats) updateStats(data.stats);
        }
        if (data.error) {
          botDiv.textContent += '\\n⚠️ Error: ' + data.error;
          botDiv.style.color = '#fca5a5';
        }
      }
    }
  } catch (err) {
    botDiv.textContent += '\\nError: ' + err.message;
  } finally {
    setGenerating(false);
  }
});

const tempSlider   = document.getElementById('temp-slider');
const tokensSlider = document.getElementById('tokens-slider');

tempSlider.addEventListener('input', () => {
  document.getElementById('temp-val').textContent = parseFloat(tempSlider.value).toFixed(2);
});
tempSlider.addEventListener('change', async () => { await runCmd('/temp ' + tempSlider.value); });

tokensSlider.addEventListener('input', () => {
  document.getElementById('tokens-val').textContent = tokensSlider.value;
});
tokensSlider.addEventListener('change', async () => { await runCmd('/tokens ' + tokensSlider.value); });

function autoResize() {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 180) + 'px';
}
msgInput.addEventListener('input', autoResize);
msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

(async function init() {
  const [statsResp, infoResp] = await Promise.all([fetch('/stats'), fetch('/model_info')]);
  updateStats(await statsResp.json());
  const info = await infoResp.json();
  document.getElementById('model-name').textContent  = info.name;
  document.getElementById('header-model').textContent = info.short;
})();
</script>
</body>
</html>
"""


def _run_web(chat: CLNChat, model_label: str, port: int) -> None:
    try:
        import json
        import queue
        import threading
        from flask import Flask, Response, jsonify, request, stream_with_context
    except ImportError:
        print("Error: Flask no está instalado. Ejecuta:\n  pip install flask")
        sys.exit(1)

    from cln.core import LiquidLinear

    app = Flask(__name__)


    def _stats() -> dict:
        layers = [m for m in chat.model.modules() if isinstance(m, LiquidLinear)]
        total_norm = sum(m.delta_w.float().norm().item() for m in layers)
        if chat.plastic:
            mode = "diferido" if (getattr(chat, "deferred_learning", True) and chat._is_hf) else "en línea"
        else:
            mode = "OFF"
        return {
            "plastic_layers": len(layers),
            "delta_w_norm":   round(total_norm, 6),
            "turns":          len(chat.history) // 2,
            "mode":           mode,
            "plastic":        chat.plastic,
            "temp":           chat.temperature,
            "max_tokens":     chat.max_new_tokens,
            "ctx_tokens":     getattr(chat, "_last_context_tokens", None),
            "max_ctx_tokens": chat.max_context_tokens,
        }

    def _layer_stats_text() -> str:
        layers = [(n, m) for n, m in chat.model.named_modules() if isinstance(m, LiquidLinear)]
        s = _stats()
        lines = [
            f"Capas plásticas : {s['plastic_layers']}",
            f"‖ΔW‖ total     : {s['delta_w_norm']:.6f}",
            f"Turnos          : {s['turns']}",
            f"Aprendizaje     : {s['mode']}",
            "",
            "Top capas por ‖ΔW‖ / ‖W‖:",
        ]
        top = sorted(layers, key=lambda x: x[1].delta_w.float().norm().item(), reverse=True)[:10]
        for name, m in top:
            dn = m.delta_w.float().norm().item()
            bn = m.weight.data.float().norm().item()
            short = ("…" + name[-38:]) if len(name) > 40 else name
            lines.append(f"  {short:<40}  {dn:.5f}  ({dn/(bn+1e-8):.2%})")
        return "\n".join(lines)

    class _SSECapture:
        def __init__(self, q):
            self._q = q
        def write(self, s):
            self._q.put(("token", s))
        def flush(self):
            pass


    @app.route("/")
    def index():
        return _HTML_PAGE

    @app.route("/model_info")
    def model_info():
        parts = model_label.split("/")
        short = parts[-1] if "/" in model_label else model_label
        return jsonify({"name": model_label, "short": short + " × CLN"})

    @app.route("/stats")
    def stats_api():
        return jsonify(_stats())

    @app.route("/command", methods=["POST"])
    def command():
        cmd = request.get_json().get("cmd", "").strip()

        if cmd == "/stats":
            return jsonify({"text": _layer_stats_text(), "stats": _stats()})
        if cmd == "/save":
            chat._save_memory()
            return jsonify({"text": f"Guardado → {chat.memory_path}", "stats": _stats()})
        if cmd == "/load":
            chat._load_memory()
            return jsonify({"text": f"Cargado ← {chat.memory_path}", "stats": _stats()})
        if cmd == "/consolidate":
            if chat._is_hf:
                from cln.loader import consolidate_hf
                consolidate_hf(chat.model)
            else:
                chat.model.consolidate_all()
            return jsonify({"text": "Memoria consolidada — EWC actualizado.", "stats": _stats()})
        if cmd == "/reset":
            for m in chat.model.modules():
                if isinstance(m, LiquidLinear):
                    m.reset_plasticity()
            return jsonify({"text": "ΔW reiniciado. Toda la memoria online borrada.", "stats": _stats()})
        if cmd == "/clear":
            chat.history.clear()
            return jsonify({"text": "Historial limpiado (ΔW se conserva).", "stats": _stats()})
        if cmd.startswith("/plastic"):
            parts = cmd.split()
            if len(parts) > 1:
                chat.plastic = parts[1].lower() in ("on", "true", "1")
            state = "ACTIVADO" if chat.plastic else "DESACTIVADO"
            return jsonify({"text": f"Aprendizaje online: {state}", "stats": _stats()})
        if cmd.startswith("/temp "):
            try:
                chat.temperature = float(cmd.split()[1])
            except (IndexError, ValueError):
                pass
            return jsonify({"text": f"Temperatura → {chat.temperature}", "stats": _stats()})
        if cmd.startswith("/tokens "):
            try:
                chat.max_new_tokens = int(cmd.split()[1])
            except (IndexError, ValueError):
                pass
            return jsonify({"text": f"Max tokens → {chat.max_new_tokens}", "stats": _stats()})
        return jsonify({"text": f"Comando desconocido: {cmd}", "stats": _stats()})

    @app.route("/upload", methods=["POST"])
    def upload_file():
        if "file" not in request.files:
            return jsonify({"text": "Error: no se recibió ningún archivo.", "stats": _stats()})
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"text": "Error: archivo vacío.", "stats": _stats()})
            
        import tempfile
        import os
        from cln.learn import learn_file
        
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, 'wb') as f:
            f.write(file.read())
            
        q = queue.Queue()
        
        def run():
            old_out = sys.stdout
            old_err = sys.stderr
            sys.stdout = _SSECapture(q)
            sys.stderr = _SSECapture(q)
            try:
                tok = chat.tokenizer if chat._is_hf else None
                learn_file(
                    chat.model, path, tokenizer=tok,
                    epochs=3, eta_multiplier=10.0,
                    save_path=chat.memory_path,
                )
                q.put(("done", {"text": f"✅ Estudio completado: '{file.filename}'.", "stats": _stats()}))
            except Exception as e:
                q.put(("error", str(e)))
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
                if os.path.exists(path):
                    os.remove(path)
                    
        def generate():
            t = threading.Thread(target=run, daemon=True)
            t.start()
            while True:
                kind, payload = q.get()
                if kind == "token":
                    yield f"data: {json.dumps({'token': payload})}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'done': True, 'text': payload['text'], 'stats': payload['stats']})}\n\n"
                    break
                elif kind == "error":
                    yield f"data: {json.dumps({'error': payload, 'done': True, 'stats': _stats()})}\n\n"
                    break
                    
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/chat", methods=["POST"])
    def chat_stream():
        message = request.get_json().get("message", "").strip()
        if not message:
            return Response("", mimetype="text/event-stream")

        q = queue.Queue()

        def run():
            old = sys.stdout
            sys.stdout = _SSECapture(q)
            try:
                chat.chat(message)
            except Exception as e:
                q.put(("error", str(e)))
            finally:
                sys.stdout = old
                q.put(("done", _stats()))

        def generate():
            t = threading.Thread(target=run, daemon=True)
            t.start()
            while True:
                kind, payload = q.get()
                if kind == "token":
                    yield f"data: {json.dumps({'token': payload})}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'done': True, 'stats': payload})}\n\n"
                    break
                elif kind == "error":
                    yield f"data: {json.dumps({'error': payload, 'done': True, 'stats': _stats()})}\n\n"
                    break

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


    print(f"\n  CLN Chat Web listo → http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)



def main():
    args = parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.model in GPT2_VARIANTS:
        from cln import load_gpt2, verify_load
        model = load_gpt2(args.model, verbose=True)
        print()
        verify_load(model, args.model, verbose=True)
        tokenizer = None
    else:
        from cln.loader import load_hf
        model, tokenizer = load_hf(
            args.model, dtype=dtype, device=args.device, verbose=True,
        )

    chat = CLNChat(
        model          = model,
        tokenizer      = tokenizer,
        memory_path    = args.memory,
        max_new_tokens = args.tokens,
        temperature    = args.temp,
        top_k          = args.top_k,
        top_p          = args.top_p,
        plastic        = not args.no_plastic,
    )

    if args.web:
        _run_web(chat, args.model, args.port)
    else:
        chat.run()


if __name__ == "__main__":
    main()
