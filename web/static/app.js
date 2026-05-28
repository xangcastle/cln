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
      const lines = buffer.split('\n');
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
          botDiv.textContent = responseText.trim() || '(no response)';
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

function setButtonLoading(btn, loading) {
  if (loading) {
    btn._originalHTML = btn.innerHTML;
    btn.innerHTML = '<span class="spinner"></span> Working...';
    btn.disabled = true;
  } else {
    btn.innerHTML = btn._originalHTML;
    btn.disabled = false;
  }
}

async function runCmd(cmd, btn = null) {
  if (isGenerating) return;
  if (btn) setButtonLoading(btn, true);
  try {
    const resp = await fetch('/command', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cmd}),
    });
    const data = await resp.json();
    if (data.text) addMessage(data.text, 'system');
    if (data.stats) updateStats(data.stats);
  } finally {
    if (btn) setButtonLoading(btn, false);
  }
}

async function togglePlastic() {
  const btn = document.getElementById('plastic-btn');
  await runCmd('/plastic ' + (plasticOn ? 'off' : 'on'), btn);
}

document.getElementById('file-upload').addEventListener('change', async function(e) {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = '';
  if (isGenerating) return;

  const botDiv = addMessage('Uploading and studying ' + file.name + '...', 'system', false);
  setGenerating(true);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let responseText = 'Learning...\n';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch(err) { continue; }
        if (data.token !== undefined) {
          responseText += data.token;
          const displayLines = responseText.split('\r');
          botDiv.textContent = displayLines[displayLines.length - 1];
          chatArea.scrollTop = chatArea.scrollHeight;
        }
        if (data.done) {
          botDiv.textContent += '\n' + (data.text || 'Done.');
          if (data.stats) updateStats(data.stats);
        }
        if (data.error) {
          botDiv.textContent += '\n⚠️ Error: ' + data.error;
          botDiv.style.color = '#fca5a5';
        }
      }
    }
  } catch (err) {
    botDiv.textContent += '\nError: ' + err.message;
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
