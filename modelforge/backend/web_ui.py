"""Embedded Material-3 Web Studio served by ModelForge Backend at `/` using strictly relative URLs."""

WEB_STUDIO_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ModelForge AI — Лаборатория моделей, агентов и исследований</title>
<style>
  :root {
    --bg: #0f141c; --surface: #18202c; --surface2: #212c3d; --primary: #6dd5fa;
    --accent: #a78bfa; --ok: #34d399; --warn: #fbbf24; --err: #f87171;
    --text: #e2e8f0; --muted: #94a3b8; --border: #2d3b52;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); display: flex; min-height: 100vh; }
  nav { width: 240px; background: var(--surface); border-right: 1px solid var(--border); padding: 16px 12px; display: flex; flex-direction: column; gap: 6px; }
  nav h1 { font-size: 18px; margin: 0 8px 12px; color: var(--primary); letter-spacing: 0.3px; }
  nav button { background: transparent; border: none; color: var(--text); text-align: left; padding: 10px 12px; border-radius: 10px; cursor: pointer; font-size: 14px; transition: background 0.15s; }
  nav button:hover, nav button.active { background: var(--surface2); color: var(--primary); font-weight: 600; }
  main { flex: 1; padding: 24px; overflow-y: auto; max-width: 1280px; }
  .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; gap: 12px; flex-wrap: wrap; }
  .badge { padding: 5px 10px; border-radius: 999px; font-size: 12px; background: var(--surface2); border: 1px solid var(--border); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px; }
  .card h3 { margin-top: 0; font-size: 16px; color: var(--primary); }
  label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 4px; margin-top: 10px; }
  input, select, textarea { width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 9px 11px; font-size: 14px; }
  textarea { min-height: 80px; font-family: inherit; }
  .btn { margin-top: 12px; background: linear-gradient(135deg, #38bdf8, #818cf8); color: #0b101b; font-weight: 700; border: none; padding: 10px 16px; border-radius: 9px; cursor: pointer; }
  .btn.secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  pre { background: #0a0e14; border: 1px solid var(--border); border-radius: 10px; padding: 12px; overflow-x: auto; font-size: 12.5px; line-height: 1.45; max-height: 340px; }
  .tab { display: none; }
  .tab.active { display: block; }
  @media (max-width: 780px) { body { flex-direction: column; } nav { width: 100%; flex-direction: row; overflow-x: auto; border-right: none; border-bottom: 1px solid var(--border); } }
</style>
</head>
<body>
<nav aria-label="ModelForge Navigation">
  <h1>ModelForge AI</h1>
  <button class="active" onclick="showTab('dashboard', this)">Dashboard</button>
  <button onclick="showTab('chat', this)">AI Chat / Prompt</button>
  <button onclick="showTab('models', this)">Model Studio</button>
  <button onclick="showTab('training', this)">Training Center</button>
  <button onclick="showTab('agents', this)">Agent Studio</button>
  <button onclick="showTab('promptedit', this)">Project Explorer & Diff</button>
  <button onclick="showTab('research', this)">Research Lab</button>
  <button onclick="showTab('mcp', this)">MCP & Plugins</button>
  <button onclick="showTab('monitor', this)">System & Settings</button>
</nav>
<main>
  <div class="topbar">
    <div>
      <strong id="activeProjLabel">Проект: загрузка...</strong>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <span class="badge" id="modeBadge">Mode: local</span>
      <span class="badge" id="ramBadge">RAM: --</span>
      <span class="badge" id="wsBadge">Events: connecting</span>
    </div>
  </div>

  <section id="tab-dashboard" class="tab active">
    <div class="grid" id="statsGrid"></div>
    <div class="grid">
      <div class="card">
        <h3>Создать проект</h3>
        <label>Название проекта</label>
        <input id="projName" value="Bilingual RU-TJ Lab"/>
        <label>Описание</label>
        <input id="projDesc" value="Компактные русско-таджикские модели и исследовательские агенты"/>
        <button class="btn" onclick="createProject()">Создать проект</button>
      </div>
      <div class="card">
        <h3>Поток событий ядра (Live WebSocket)</h3>
        <pre id="eventStreamLog">Ожидание событий...</pre>
      </div>
    </div>
  </section>

  <section id="tab-chat" class="tab">
    <div class="card">
      <h3>AI Chat / Prompt Workspace</h3>
      <label>Сообщение или инструкция</label>
      <textarea id="chatInput">Салом! Привет! Расскажи о компактной модели.</textarea>
      <button class="btn" onclick="sendChat()">Отправить</button>
      <pre id="chatOut"></pre>
    </div>
  </section>

  <section id="tab-models" class="tab">
    <div class="grid">
      <div class="card">
        <h3>Model Builder (по промпту)</h3>
        <label>Описание модели обычным языком</label>
        <textarea id="modelPrompt">Создай компактную языковую модель для русского и таджикского текста с экспериментальной архитектурой</textarea>
        <button class="btn" onclick="createModelFromPrompt()">Создать модель и токенизатор</button>
        <pre id="modelCreateOut"></pre>
      </div>
      <div class="card">
        <h3>Генерация текста из чекпоинта</h3>
        <label>Model ID</label>
        <input id="inferModelId" placeholder="mdl-..."/>
        <label>Промпт</label>
        <input id="inferPrompt" value="Салом ҷаҳон! Привет"/>
        <button class="btn" onclick="runGenerate()">Сгенерировать</button>
        <button class="btn secondary" onclick="exportModel('gguf')">Экспорт GGUF v3</button>
        <pre id="inferOut"></pre>
      </div>
    </div>
  </section>

  <section id="tab-training" class="tab">
    <div class="card">
      <h3>Training Center — Реальное обучение и чекпоинты</h3>
      <div class="grid">
        <div>
          <label>Model ID</label>
          <input id="trainModelId" placeholder="mdl-..."/>
          <label>Режим</label>
          <select id="trainMode">
            <option value="scratch">Обучение с нуля (scratch)</option>
            <option value="finetune">Fine-tuning</option>
            <option value="transfer">Transfer learning (freeze prefix)</option>
          </select>
        </div>
        <div>
          <label>Эпохи</label>
          <input id="trainEpochs" type="number" value="4" min="1" max="50"/>
          <label>Learning rate</label>
          <input id="trainLr" type="number" step="0.0005" value="0.004"/>
        </div>
      </div>
      <button class="btn" onclick="startTraining()">Запустить реальное обучение</button>
      <pre id="trainOut"></pre>
    </div>
  </section>

  <section id="tab-agents" class="tab">
    <div class="grid">
      <div class="card">
        <h3>Agent Studio — Создание агента</h3>
        <label>Описание задачи агента</label>
        <textarea id="agentPrompt">Исследовательский агент, который изучает проект и запускает безопасные тесты маленьких моделей</textarea>
        <button class="btn" onclick="createAgent()">Создать агента</button>
        <pre id="agentCreateOut"></pre>
      </div>
      <div class="card">
        <h3>Запуск агента</h3>
        <label>Agent ID</label>
        <input id="runAgentId" placeholder="agt-..."/>
        <label>Задача</label>
        <input id="runAgentTask" value="Проверь гибридную архитектуру и файлы проекта"/>
        <button class="btn" onclick="executeAgent()">Выполнить задачу</button>
        <pre id="agentRunOut"></pre>
      </div>
    </div>
  </section>

  <section id="tab-promptedit" class="tab">
    <div class="card">
      <h3>Prompt-to-Project — Редактирование проекта с Diff и Rollback</h3>
      <label>Текстовая инструкция изменения проекта</label>
      <input id="editPromptInput" value="Добавь таджикский язык и снизь RAM"/>
      <button class="btn" onclick="applyPromptEdit()">Применить промпт и запустить тесты</button>
      <button class="btn secondary" onclick="rollbackLastEdit()">Откатить (Rollback)</button>
      <pre id="editDiffOut"></pre>
    </div>
  </section>

  <section id="tab-research" class="tab">
    <div class="card">
      <h3>Research Lab — Сравнительный эксперимент архитектур</h3>
      <label>Исследовательский вопрос</label>
      <input id="resQuestion" value="Сравнение decoder_transformer и hybrid_attention_ssm на русско-таджикском корпусе"/>
      <div class="grid">
        <div><label>Baseline</label><input id="resBase" value="decoder_transformer"/></div>
        <div><label>Candidate</label><input id="resCand" value="hybrid_attention_ssm"/></div>
      </div>
      <button class="btn" onclick="runResearch()">Запустить исследование</button>
      <pre id="resOut"></pre>
    </div>
  </section>

  <section id="tab-mcp" class="tab">
    <div class="card">
      <h3>MCP Servers & Plugins</h3>
      <button class="btn" onclick="probeBuiltinMcp()">Проверить встроенный MCP STDIO сервер</button>
      <button class="btn secondary" onclick="runTajikPlugin()">Запустить плагин Tajik Text Inspector</button>
      <pre id="mcpOut"></pre>
    </div>
  </section>

  <section id="tab-monitor" class="tab">
    <div class="card">
      <h3>System Monitor, Modes & Diagnostics (`doctor`)</h3>
      <label>Режим работы (Mode)</label>
      <select id="modeSelect">
        <option value="local">Local Mode</option>
        <option value="remote">Remote Mode</option>
        <option value="hybrid">Hybrid Mode</option>
        <option value="offline">Offline Mode (без сети)</option>
      </select>
      <button class="btn" onclick="switchMode()">Переключить режим</button>
      <button class="btn secondary" onclick="refreshDoctor()">Обновить диагностику</button>
      <pre id="doctorOut"></pre>
    </div>
  </section>
</main>

<script>
let currentProjectId = "";
let lastEditId = "";

function showTab(name, btn) {
  document.querySelectorAll('.tab').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: {'Content-Type': 'application/json'},
    ...opts
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}

async function refreshDashboard() {
  const doc = await api('api/v1/doctor');
  document.getElementById('modeBadge').textContent = 'Mode: ' + doc.mode + (doc.platform.low_memory_mode ? ' (Low-RAM)' : '');
  document.getElementById('ramBadge').textContent = 'RAM avail: ' + doc.resources.available_ram_mb + ' MB';
  document.getElementById('doctorOut').textContent = JSON.stringify(doc, null, 2);

  const projs = await api('api/v1/projects');
  if (projs.projects.length > 0 && !currentProjectId) {
    currentProjectId = projs.projects[0].id;
  }
  document.getElementById('activeProjLabel').textContent = currentProjectId
    ? ('Активный проект: ' + currentProjectId)
    : 'Проект ещё не создан — создайте на Dashboard';

  const c = doc.counts;
  document.getElementById('statsGrid').innerHTML = [
    ['Проекты', c.projects], ['Модели', c.models], ['Агенты', c.agents],
    ['Эксперименты', c.experiments], ['Датасеты', c.datasets], ['Архитектуры', c.architectures]
  ].map(([k, v]) => `<div class="card"><h3>${k}</h3><div style="font-size:28px;font-weight:700;">${v}</div></div>`).join('');

  const models = await api('api/v1/models');
  if (models.models.length > 0) {
    const mid = models.models[0].id;
    if (!document.getElementById('inferModelId').value) document.getElementById('inferModelId').value = mid;
    if (!document.getElementById('trainModelId').value) document.getElementById('trainModelId').value = mid;
  }
  const agents = await api('api/v1/agents');
  if (agents.agents.length > 0 && !document.getElementById('runAgentId').value) {
    document.getElementById('runAgentId').value = agents.agents[0].id;
  }
}

async function ensureProject() {
  if (currentProjectId) return currentProjectId;
  const created = await api('api/v1/projects', {
    method: 'POST',
    body: JSON.stringify({name: 'Default Lab Project', description: 'Auto-created workspace'})
  });
  currentProjectId = created.id;
  await refreshDashboard();
  return currentProjectId;
}

async function createProject() {
  const p = await api('api/v1/projects', {
    method: 'POST',
    body: JSON.stringify({
      name: document.getElementById('projName').value,
      description: document.getElementById('projDesc').value
    })
  });
  currentProjectId = p.id;
  await refreshDashboard();
}

async function sendChat() {
  const pid = await ensureProject();
  const out = await api('api/v1/chat', {
    method: 'POST',
    body: JSON.stringify({project_id: pid, message: document.getElementById('chatInput').value})
  });
  document.getElementById('chatOut').textContent = JSON.stringify(out, null, 2);
}

async function createModelFromPrompt() {
  const pid = await ensureProject();
  const out = await api('api/v1/models/from-prompt', {
    method: 'POST',
    body: JSON.stringify({project_id: pid, prompt: document.getElementById('modelPrompt').value})
  });
  document.getElementById('inferModelId').value = out.id;
  document.getElementById('trainModelId').value = out.id;
  document.getElementById('modelCreateOut').textContent = JSON.stringify(out, null, 2);
  await refreshDashboard();
}

async function startTraining() {
  const mid = document.getElementById('trainModelId').value.trim();
  document.getElementById('trainOut').textContent = 'Обучение запущено...';
  const out = await api('api/v1/training/run', {
    method: 'POST',
    body: JSON.stringify({
      model_id: mid,
      mode: document.getElementById('trainMode').value,
      epochs: parseInt(document.getElementById('trainEpochs').value, 10),
      learning_rate: parseFloat(document.getElementById('trainLr').value),
      wait: true
    })
  });
  document.getElementById('trainOut').textContent = JSON.stringify(out, null, 2);
  await refreshDashboard();
}

async function runGenerate() {
  const mid = document.getElementById('inferModelId').value.trim();
  const out = await api('api/v1/inference/generate', {
    method: 'POST',
    body: JSON.stringify({model_id: mid, prompt: document.getElementById('inferPrompt').value, max_tokens: 28})
  });
  document.getElementById('inferOut').textContent = JSON.stringify(out, null, 2);
}

async function exportModel(fmt) {
  const mid = document.getElementById('inferModelId').value.trim();
  const out = await api('api/v1/models/' + encodeURIComponent(mid) + '/export', {
    method: 'POST',
    body: JSON.stringify({format: fmt})
  });
  document.getElementById('inferOut').textContent = JSON.stringify(out, null, 2);
}

async function createAgent() {
  const pid = await ensureProject();
  const out = await api('api/v1/agents/from-prompt', {
    method: 'POST',
    body: JSON.stringify({project_id: pid, prompt: document.getElementById('agentPrompt').value})
  });
  document.getElementById('runAgentId').value = out.id;
  document.getElementById('agentCreateOut').textContent = JSON.stringify(out, null, 2);
  await refreshDashboard();
}

async function executeAgent() {
  const aid = document.getElementById('runAgentId').value.trim();
  const out = await api('api/v1/agents/' + encodeURIComponent(aid) + '/run', {
    method: 'POST',
    body: JSON.stringify({prompt: document.getElementById('runAgentTask').value})
  });
  document.getElementById('agentRunOut').textContent = JSON.stringify(out, null, 2);
}

async function applyPromptEdit() {
  const pid = await ensureProject();
  const out = await api('api/v1/projects/' + encodeURIComponent(pid) + '/prompt-edit', {
    method: 'POST',
    body: JSON.stringify({prompt: document.getElementById('editPromptInput').value})
  });
  lastEditId = out.id;
  document.getElementById('editDiffOut').textContent = (out.diff_text || '(нет текстовых изменений)') + '\\n\\n' + JSON.stringify(out.test_report, null, 2);
}

async function rollbackLastEdit() {
  if (!lastEditId) { alert('Сначала примените изменение.'); return; }
  const out = await api('api/v1/prompt-edits/' + encodeURIComponent(lastEditId) + '/rollback', {method: 'POST', body: '{}'});
  document.getElementById('editDiffOut').textContent = 'Откат выполнен:\\n' + JSON.stringify(out, null, 2);
}

async function runResearch() {
  const pid = await ensureProject();
  document.getElementById('resOut').textContent = 'Выполняется контролируемый эксперимент...';
  const out = await api('api/v1/research/study', {
    method: 'POST',
    body: JSON.stringify({
      project_id: pid,
      question: document.getElementById('resQuestion').value,
      baseline_arch: document.getElementById('resBase').value,
      candidate_arch: document.getElementById('resCand').value,
      epochs: 3
    })
  });
  document.getElementById('resOut').textContent = JSON.stringify(out, null, 2);
  await refreshDashboard();
}

async function probeBuiltinMcp() {
  const probe = await api('api/v1/mcp/mcp-builtin-local/probe', {method: 'POST', body: '{}'});
  const call = await api('api/v1/mcp/mcp-builtin-local/call', {
    method: 'POST',
    body: JSON.stringify({tool_name: 'analyze_corpus_stats', arguments: {text: 'Салом ҷаҳон! Привет мир!'}})
  });
  document.getElementById('mcpOut').textContent = JSON.stringify({probe, call}, null, 2);
}

async function runTajikPlugin() {
  const out = await api('api/v1/plugins/tajik_text_inspector/run', {
    method: 'POST',
    body: JSON.stringify({inputs: {text: 'Салом ҷаҳон! Забони тоҷикӣ ва русӣ.'}})
  });
  document.getElementById('mcpOut').textContent = JSON.stringify(out, null, 2);
}

async function switchMode() {
  const m = document.getElementById('modeSelect').value;
  const out = await api('api/v1/mode', {method: 'POST', body: JSON.stringify({mode: m})});
  document.getElementById('doctorOut').textContent = JSON.stringify(out, null, 2);
  await refreshDashboard();
}

async function refreshDoctor() {
  await refreshDashboard();
}

function connectWs() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = proto + '//' + window.location.host + window.location.pathname.replace(/\\/$/, '') + '/ws/events';
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => { document.getElementById('wsBadge').textContent = 'Events: connected'; };
  ws.onmessage = (ev) => {
    const log = document.getElementById('eventStreamLog');
    const parsed = JSON.parse(ev.data);
    log.textContent = '[' + parsed.topic + '] ' + JSON.stringify(parsed.payload) + '\\n' + log.textContent.slice(0, 3000);
  };
  ws.onclose = () => {
    document.getElementById('wsBadge').textContent = 'Events: reconnecting';
    setTimeout(connectWs, 3000);
  };
}

refreshDashboard().catch(err => console.error(err));
connectWs();
</script>
</body>
</html>
"""
