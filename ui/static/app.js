/* Editor UI client — vanilla JS, no dependencies. Pure helpers first (unit-tested under node), then the
   DOM application, which only runs in a browser. */
'use strict';

// ───────────────────────────── pure helpers ─────────────────────────────
function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** text + [[start, end, kind], …] (sorted, non-overlapping) → HTML with <span class="t-kind">. */
function renderSpans(text, spans) {
  let out = '', pos = 0;
  for (const [a, b, kind] of spans || []) {
    if (a < pos || b > text.length || a >= b) continue;          // tolerate a stale or odd span list
    out += escapeHtml(text.slice(pos, a)) + '<span class="t-' + kind + '">' + escapeHtml(text.slice(a, b)) + '</span>';
    pos = b;
  }
  return out + escapeHtml(text.slice(pos));
}

/** Line diff: [{t: ' '|'+'|'-', s: line}]. LCS on the part between common prefix and suffix. */
function lineDiff(oldText, newText) {
  const a = oldText.split('\n'), b = newText.split('\n');
  let s = 0;
  while (s < a.length && s < b.length && a[s] === b[s]) s++;
  let ea = a.length, eb = b.length;
  while (ea > s && eb > s && a[ea - 1] === b[eb - 1]) { ea--; eb--; }
  const A = a.slice(s, ea), B = b.slice(s, eb);
  const out = a.slice(0, s).map(l => ({t: ' ', s: l}));
  if (A.length * B.length > 4e6) {                               // too big for an LCS table: show as replaced
    A.forEach(l => out.push({t: '-', s: l})); B.forEach(l => out.push({t: '+', s: l}));
  } else {
    const n = A.length, m = B.length;
    const L = Array.from({length: n + 1}, () => new Uint32Array(m + 1));
    for (let i = n - 1; i >= 0; i--)
      for (let j = m - 1; j >= 0; j--)
        L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    let i = 0, j = 0;
    while (i < n && j < m) {
      if (A[i] === B[j]) { out.push({t: ' ', s: A[i]}); i++; j++; }
      else if (L[i + 1][j] >= L[i][j + 1]) out.push({t: '-', s: A[i++]});
      else out.push({t: '+', s: B[j++]});
    }
    while (i < n) out.push({t: '-', s: A[i++]});
    while (j < m) out.push({t: '+', s: B[j++]});
  }
  a.slice(ea).forEach(l => out.push({t: ' ', s: l}));
  return out;
}

function diffStats(d) {
  return {added: d.filter(x => x.t === '+').length, removed: d.filter(x => x.t === '-').length};
}

/** Fenced code blocks of a markdown reply: [{lang, code, hint}] — hint is a file path named just before. */
function extractCodeBlocks(md) {
  const out = [], lines = md.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(`{3,})\s*([\w+#.-]*)\s*$/);
    if (!m) continue;
    const fence = m[1];
    let j = i + 1;
    const body = [];
    while (j < lines.length && !lines[j].startsWith(fence)) body.push(lines[j++]);
    let hint = '';
    for (let k = i - 1; k >= Math.max(0, i - 2); k--) {
      const h = lines[k].match(/`([\w./@+-]+\.\w+)`/) || lines[k].match(/^\**([\w./@+-]+\.\w+)\**:?\s*$/);
      if (h) { hint = h[1]; break; }
    }
    if (!hint && body.length) {
      const c = body[0].match(/^(?:\/\/|#|--|\/\*)\s*(?:file(?:name)?:\s*)?([\w./@+-]+\.\w+)\s*(?:\*\/)?$/i);
      if (c) hint = c[1];
    }
    out.push({lang: m[2], code: body.join('\n'), hint});
    i = j;
  }
  return out;
}

/** Small, safe markdown → HTML (everything is escaped first). Code blocks carry their index. */
function renderMarkdown(md) {
  const blocks = extractCodeBlocks(md);
  let n = 0, html = '', inFence = null, para = [];
  const inline = s => escapeHtml(s)
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  const flush = () => { if (para.length) { html += '<p>' + para.map(inline).join('<br>') + '</p>'; para = []; } };
  let listOpen = false;
  const closeList = () => { if (listOpen) { html += '</ul>'; listOpen = false; } };
  let buf = [], lang = '';
  for (const line of md.split('\n')) {
    if (inFence) {
      if (line.startsWith(inFence)) {
        html += '<div class="code"><div class="code-head"><span>' + escapeHtml(lang || 'code') +
          '</span><span><button data-act="copy" data-i="' + n + '">Copy</button>' +
          '<button data-act="apply" data-i="' + n + '">Apply…</button></span></div><pre>' +
          escapeHtml(buf.join('\n')) + '</pre></div>';
        n++; inFence = null; buf = [];
      } else buf.push(line);
      continue;
    }
    const f = line.match(/^(`{3,})\s*([\w+#.-]*)\s*$/);
    if (f) { flush(); closeList(); inFence = f[1]; lang = f[2]; continue; }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { flush(); closeList(); html += '<h' + (h[1].length + 2) + '>' + inline(h[2]) + '</h' + (h[1].length + 2) + '>'; continue; }
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) { flush(); if (!listOpen) { html += '<ul>'; listOpen = true; } html += '<li>' + inline(li[1]) + '</li>'; continue; }
    if (!line.trim()) { flush(); closeList(); continue; }
    closeList();
    para.push(line);
  }
  if (inFence) {                                                 // a reply still streaming: show the open block
    html += '<div class="code"><div class="code-head"><span>' + escapeHtml(lang || 'code') +
      '</span></div><pre>' + escapeHtml(buf.join('\n')) + '</pre></div>';
  }
  flush(); closeList();
  return {html, blocks};
}

/** Incremental SSE parser: feed text chunks, get {content, reasoning, done}. */
function makeSSE() {
  let buf = '';
  return {
    feed(chunk) {
      buf += chunk;
      const out = {content: '', reasoning: '', done: false, error: null};
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') { out.done = true; continue; }
        try {
          const ev = JSON.parse(data);
          if (ev.error) { out.error = ev.error.message || 'error'; continue; }
          for (const c of ev.choices || []) {
            const d = c.delta || {};
            if (d.content) out.content += d.content;
            if (d.reasoning_content) out.reasoning += d.reasoning_content;
          }
        } catch (e) { /* a partial or foreign line */ }
      }
      return out;
    },
  };
}

function lineStart(text, line) {                                 // 1-based line → character offset
  let pos = 0;
  for (let i = 1; i < line; i++) {
    const nl = text.indexOf('\n', pos);
    if (nl < 0) return text.length;
    pos = nl + 1;
  }
  return pos;
}

function flattenOutline(symbols, depth = 0, out = []) {
  for (const s of symbols || []) { out.push({depth, ...s}); flattenOutline(s.children, depth + 1, out); }
  return out;
}

function symbolAtLine(flat, line) {                              // innermost symbol containing the line
  let best = null;
  for (const s of flat) if (s.line <= line && line <= s.end && (!best || s.depth >= best.depth)) best = s;
  return best;
}

function statusLabel(f) {
  return {modified: 'M', added: 'A', deleted: 'D', renamed: 'R', untracked: '?'}[f.status] || f.status;
}

function indentSelection(text, start, end, outdent) {
  const ls = text.lastIndexOf('\n', start - 1) + 1;
  const block = text.slice(ls, end);
  const lines = block.split('\n');
  const changed = lines.map(l => outdent ? l.replace(/^( {1,4}|\t)/, '') : '    ' + l);
  const out = text.slice(0, ls) + changed.join('\n') + text.slice(end);
  const delta = changed.join('\n').length - block.length;
  return {text: out, start: Math.max(ls, start + (changed[0].length - lines[0].length)), end: end + delta};
}

if (typeof module !== 'undefined') {
  module.exports = {escapeHtml, renderSpans, lineDiff, diffStats, extractCodeBlocks, renderMarkdown, makeSSE,
                    lineStart, flattenOutline, symbolAtLine, statusLabel, indentSelection};
}

// ───────────────────────────── application (browser only) ─────────────────────────────
if (typeof document !== 'undefined') {
  (function app() {
    const $ = s => document.querySelector(s);
    const params = new URLSearchParams(location.search);
    let TOKEN = params.get('token') || sessionStorage.getItem('uiToken') || '';
    if (params.get('token')) { sessionStorage.setItem('uiToken', TOKEN); history.replaceState(null, '', location.pathname); }

    const S = {files: new Map(), active: null, attach: new Set(), chat: [], git: null, expanded: new Set(['']),
               tree: new Map(), busy: false, analyzeTimer: null, think: false};

    async function api(path, opt = {}) {
      const r = await fetch(path, {method: opt.method || 'GET',
        headers: {'X-UI-Token': TOKEN, 'Content-Type': 'application/json'},
        body: opt.body === undefined ? undefined : JSON.stringify(opt.body)});
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || r.statusText);
      return data;
    }
    const toast = (msg, bad) => {
      const t = $('#toast'); t.textContent = msg; t.className = 'show' + (bad ? ' bad' : '');
      clearTimeout(toast.t); toast.t = setTimeout(() => t.className = '', bad ? 6000 : 2500);
    };
    const guard = fn => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message, true); } };

    // ---- modal
    function modal(title, bodyHtml, buttons) {
      return new Promise(resolve => {
        const m = $('#modal');
        m.innerHTML = '<div class="box"><h3>' + escapeHtml(title) + '</h3><div class="mbody">' + bodyHtml +
          '</div><div class="mbtns"></div></div>';
        const row = m.querySelector('.mbtns');
        for (const b of buttons) {
          const el = document.createElement('button');
          el.textContent = b.label; el.className = b.cls || '';
          el.onclick = () => { m.className = ''; resolve(b.value); };
          row.appendChild(el);
        }
        m.className = 'open';
      });
    }

    // ---- file tree
    async function loadDir(path) {
      const r = await api('/api/tree?path=' + encodeURIComponent(path));
      S.tree.set(path, r.entries);
    }
    function treeHtml(path, depth) {
      let h = '';
      for (const e of S.tree.get(path) || []) {
        const pad = 'style="padding-left:' + (8 + depth * 12) + 'px"';
        if (e.type === 'dir') {
          const open = S.expanded.has(e.path);
          h += '<div class="node dir" ' + pad + ' data-dir="' + escapeHtml(e.path) + '">' + (open ? '▾ ' : '▸ ') +
            escapeHtml(e.name) + '</div>' + (open ? treeHtml(e.path, depth + 1) : '');
        } else {
          const sel = S.active === e.path ? ' active' : '';
          h += '<div class="node file' + sel + '" ' + pad + ' data-file="' + escapeHtml(e.path) + '"><span>' +
            escapeHtml(e.name) + '</span><button class="mini" data-attach="' + escapeHtml(e.path) +
            '" title="Attach to the next message">＋</button></div>';
        }
      }
      return h;
    }
    function renderTree() { $('#tree').innerHTML = treeHtml('', 0); }

    // ---- editor
    const ta = $('#ta'), hl = $('#hl'), gutter = $('#gutter');
    function paint() {
      const f = S.files.get(S.active);
      if (!f) { hl.innerHTML = ''; gutter.textContent = ''; return; }
      hl.innerHTML = renderSpans(f.text, f.spans) + '\n';
      const n = f.text.split('\n').length;
      gutter.textContent = Array.from({length: n}, (_, i) => i + 1).join('\n');
      syncScroll();
    }
    function syncScroll() {
      hl.scrollTop = ta.scrollTop; hl.scrollLeft = ta.scrollLeft; gutter.scrollTop = ta.scrollTop;
    }
    function renderTabs() {
      $('#tabs').innerHTML = [...S.files.entries()].map(([p, f]) =>
        '<div class="tab' + (p === S.active ? ' active' : '') + '" data-tab="' + escapeHtml(p) + '">' +
        escapeHtml(p.split('/').pop()) + (f.text !== f.saved ? ' ●' : '') +
        '<span class="x" data-close="' + escapeHtml(p) + '">×</span></div>').join('');
    }
    function renderOutline() {
      const f = S.files.get(S.active);
      const flat = flattenOutline(f ? f.outline : []);
      $('#outline').innerHTML = flat.length ? flat.map(s =>
        '<div class="sym" style="padding-left:' + (8 + s.depth * 12) + 'px" data-line="' + s.line + '">' +
        '<span class="k k-' + escapeHtml(s.kind) + '">' + escapeHtml(s.kind.slice(0, 3)) + '</span>' +
        escapeHtml(s.name) + '<span class="ln">' + s.line + '</span></div>').join('')
        : '<div class="empty">' + (f ? 'No symbols' : 'Open a file') + '</div>';
    }
    function renderStatus() {
      const f = S.files.get(S.active);
      if (!f) { $('#status-file').textContent = ''; return; }
      const pos = ta.selectionStart, before = f.text.slice(0, pos);
      const line = before.split('\n').length, col = pos - before.lastIndexOf('\n');
      const sym = symbolAtLine(flattenOutline(f.outline), line);
      $('#status-file').textContent = f.language + '  ·  ' + line + ':' + col + (sym ? '  ·  ' + sym.name : '');
    }
    function showFile() {
      const f = S.files.get(S.active);
      ta.value = f ? f.text : ''; ta.disabled = !f; ta.style.visibility = f ? 'visible' : 'hidden';
      $('#empty-editor').style.display = f ? 'none' : 'flex';
      paint(); renderTabs(); renderOutline(); renderStatus(); renderTree();
    }
    const openFile = guard(async path => {
      if (!S.files.has(path)) {
        const r = await api('/api/file?path=' + encodeURIComponent(path));
        if (r.binary || r.too_large) { toast(path + (r.binary ? ' is a binary file' : ' is too large to open'), true); return; }
        S.files.set(path, {text: r.text, saved: r.text, mtime: r.mtime, language: r.language, outline: r.outline, spans: r.spans});
      }
      S.active = path; showFile();
    });
    async function reanalyze() {
      const f = S.files.get(S.active); if (!f) return;
      const r = await api('/api/analyze', {method: 'POST', body: {path: S.active, text: f.text}});
      f.spans = r.spans; f.outline = r.outline; f.language = r.language;
      if (S.files.get(S.active) === f) { paint(); renderOutline(); renderStatus(); }
    }
    function onEdit() {
      const f = S.files.get(S.active); if (!f) return;
      f.text = ta.value; paint(); renderTabs(); renderStatus();
      clearTimeout(S.analyzeTimer); S.analyzeTimer = setTimeout(guard(reanalyze), 150);
    }
    const save = guard(async () => {
      const f = S.files.get(S.active); if (!f) return;
      const r = await api('/api/file', {method: 'PUT', body: {path: S.active, text: f.text, expected_mtime: f.mtime}});
      f.saved = f.text; f.mtime = r.mtime; renderTabs(); toast('Saved ' + S.active); refreshGit();
    });
    function closeFile(path) {
      const f = S.files.get(path);
      if (f && f.text !== f.saved && !confirm('Discard unsaved changes in ' + path + '?')) return;
      S.files.delete(path);
      if (S.active === path) S.active = [...S.files.keys()].pop() || null;
      showFile();
    }
    function jumpToLine(line) {
      const f = S.files.get(S.active); if (!f) return;
      const off = lineStart(f.text, line);
      ta.focus(); ta.setSelectionRange(off, off);
      const lh = parseFloat(getComputedStyle(ta).lineHeight) || 18;
      ta.scrollTop = Math.max(0, (line - 4) * lh); syncScroll(); renderStatus();
    }

    // ---- git
    const refreshGit = guard(async () => {
      S.git = await api('/api/git/status'); renderGit();
    });
    function renderGit() {
      const g = S.git;
      if (!g || !g.repo) { $('#git').innerHTML = '<div class="empty">Not a git repository</div>'; $('#branch').textContent = ''; return; }
      $('#branch').textContent = '⎇ ' + g.branch + (g.ahead ? ' ↑' + g.ahead : '') + (g.behind ? ' ↓' + g.behind : '');
      $('#git').innerHTML =
        (g.files.length ? g.files.map(f => '<label class="gf"><input type="checkbox" data-gpath="' + escapeHtml(f.path) +
          '" checked><span class="gs gs-' + f.status + '">' + statusLabel(f) + '</span><a data-gdiff="' + escapeHtml(f.path) + '">' +
          escapeHtml(f.path) + '</a></label>').join('') : '<div class="empty">Working tree clean</div>') +
        '<textarea id="cmsg" placeholder="Commit message" rows="2"></textarea>' +
        '<div class="row"><button id="commit">Commit selected</button><button id="push" class="warn">Push ' +
        escapeHtml(g.branch) + '…</button></div>';
    }
    const showDiff = guard(async path => {
      const r = await api('/api/git/diff?path=' + encodeURIComponent(path));
      const body = r.diff.split('\n').map(l => '<div class="dl ' + (l[0] === '+' ? 'add' : l[0] === '-' ? 'del' : '') + '">' +
        escapeHtml(l) + '</div>').join('');
      await modal(path, '<div class="diff">' + (body || '<div class="empty">No changes</div>') + '</div>', [{label: 'Close', value: 1}]);
    });
    const commit = guard(async () => {
      const paths = [...document.querySelectorAll('[data-gpath]')].filter(c => c.checked).map(c => c.dataset.gpath);
      const message = $('#cmsg').value.trim();
      if (!message) { toast('Write a commit message first', true); return; }
      if (!paths.length) { toast('Select at least one file', true); return; }
      const r = await api('/api/git/commit', {method: 'POST', body: {message, paths}});
      S.git = r.git; renderGit(); toast('Committed: ' + r.commit);
    });
    const push = guard(async () => {
      const g = S.git;
      const ok = await modal('Push to GitHub?', '<p>This sends <b>' + (g.ahead || 'your') + ' commit(s)</b> of <code>' +
        escapeHtml(g.branch) + '</code> to <code>' + escapeHtml(g.remotes[0] || 'origin') + '</code> using your local git ' +
        'credentials. It cannot be undone from here.</p>',
        [{label: 'Cancel', value: false}, {label: 'Push', value: true, cls: 'warn'}]);
      if (!ok) return;
      toast('Pushing…');
      const r = await api('/api/git/push', {method: 'POST', body: {confirm: true}});
      S.git = r.git; renderGit(); toast('Pushed');
    });

    // ---- chat
    function renderAttach() {
      $('#chips').innerHTML = [...S.attach].map(p => '<span class="chip">' + escapeHtml(p.split('/').pop()) +
        '<span class="x" data-detach="' + escapeHtml(p) + '">×</span></span>').join('');
    }
    function renderChat() {
      $('#messages').innerHTML = S.chat.map((m, i) => {
        if (m.role === 'user') return '<div class="msg user">' + escapeHtml(m.content).replace(/\n/g, '<br>') +
          (m.attached && m.attached.length ? '<div class="att">📎 ' + m.attached.map(escapeHtml).join(', ') + '</div>' : '') + '</div>';
        const r = renderMarkdown(m.content);
        m.blocks = r.blocks;
        return '<div class="msg bot" data-i="' + i + '">' + (m.reasoning ? '<details><summary>reasoning</summary><div class="rs">' +
          escapeHtml(m.reasoning) + '</div></details>' : '') + (r.html || '<em>…</em>') + '</div>';
      }).join('');
      const box = $('#messages'); box.scrollTop = box.scrollHeight;
    }
    const send = guard(async () => {
      if (S.busy) return;
      const input = $('#prompt'), text = input.value.trim();
      if (!text) return;
      const attached = [...S.attach];
      const cur = S.files.get(S.active);
      if (cur && cur.text !== cur.saved && attached.includes(S.active)) await save();
      S.chat.push({role: 'user', content: text, attached});
      const bot = {role: 'assistant', content: '', reasoning: ''};
      S.chat.push(bot); input.value = ''; S.attach.clear(); renderAttach(); renderChat();
      S.busy = true; $('#send').disabled = true;
      try {
        const history = S.chat.slice(0, -1).map(m => ({role: m.role, content: m.content}));
        const r = await fetch('/api/chat', {method: 'POST', headers: {'X-UI-Token': TOKEN, 'Content-Type': 'application/json'},
          body: JSON.stringify({messages: history, attachments: attached, think: S.think})});
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
        const reader = r.body.getReader(), dec = new TextDecoder(), sse = makeSSE();
        let raf = 0;
        for (;;) {
          const {value, done} = await reader.read();
          if (done) break;
          const ev = sse.feed(dec.decode(value, {stream: true}));
          if (ev.error) throw new Error(ev.error);
          bot.content += ev.content; bot.reasoning += ev.reasoning;
          if (!raf) raf = requestAnimationFrame(() => { raf = 0; renderChat(); });
        }
      } catch (e) {
        bot.content += (bot.content ? '\n\n' : '') + '**Error:** ' + e.message;
      } finally {
        S.busy = false; $('#send').disabled = false; renderChat();
      }
    });
    const applyBlock = guard(async (msgIndex, blockIndex) => {
      const block = S.chat[msgIndex].blocks[blockIndex];
      let target = block.hint && S.files.has(block.hint) ? block.hint : S.active;
      if (!target) { toast('Open the file this code belongs to first', true); return; }
      const f = S.files.get(target);
      const d = lineDiff(f.text, block.code), st = diffStats(d);
      const body = '<p>Replace the whole content of <code>' + escapeHtml(target) + '</code> (' + st.added + ' added, ' +
        st.removed + ' removed). Nothing is written to disk until you save.</p><div class="diff">' +
        d.map(x => '<div class="dl ' + (x.t === '+' ? 'add' : x.t === '-' ? 'del' : '') + '">' + escapeHtml(x.t + ' ' + x.s) + '</div>').join('') + '</div>';
      const ok = await modal('Apply to ' + target + '?', body, [{label: 'Cancel', value: false}, {label: 'Apply to editor', value: true}]);
      if (!ok) return;
      S.active = target; f.text = block.code; showFile(); onEdit();
    });

    // ---- wiring
    document.addEventListener('click', e => {
      const t = e.target;
      if (t.dataset.attach) { S.attach.add(t.dataset.attach); renderAttach(); e.stopPropagation(); return; }
      if (t.dataset.detach) { S.attach.delete(t.dataset.detach); renderAttach(); return; }
      const dir = t.closest('[data-dir]');
      if (dir) {
        const p = dir.dataset.dir;
        if (S.expanded.has(p)) { S.expanded.delete(p); renderTree(); }
        else { S.expanded.add(p); guard(async () => { await loadDir(p); renderTree(); })(); }
        return;
      }
      const file = t.closest('[data-file]'); if (file) { openFile(file.dataset.file); return; }
      if (t.dataset.close) { closeFile(t.dataset.close); return; }
      const tab = t.closest('[data-tab]'); if (tab) { S.active = tab.dataset.tab; showFile(); return; }
      const sym = t.closest('[data-line]'); if (sym) { jumpToLine(parseInt(sym.dataset.line, 10)); return; }
      if (t.dataset.gdiff) { showDiff(t.dataset.gdiff); return; }
      if (t.id === 'commit') { commit(); return; }
      if (t.id === 'push') { push(); return; }
      const act = t.dataset.act;
      if (act) {
        const msg = t.closest('.msg'); const i = parseInt(msg.dataset.i, 10), b = parseInt(t.dataset.i, 10);
        if (act === 'copy') navigator.clipboard.writeText(S.chat[i].blocks[b].code).then(() => toast('Copied'));
        else applyBlock(i, b);
      }
    });
    ta.addEventListener('input', onEdit);
    ta.addEventListener('scroll', syncScroll);
    ta.addEventListener('keyup', renderStatus);
    ta.addEventListener('click', renderStatus);
    ta.addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); save(); return; }
      if (e.key === 'Tab') {
        e.preventDefault();
        const r = indentSelection(ta.value, ta.selectionStart, ta.selectionEnd, e.shiftKey);
        if (ta.selectionStart === ta.selectionEnd && !e.shiftKey) {
          document.execCommand('insertText', false, '    ');
        } else { ta.value = r.text; ta.setSelectionRange(r.start, r.end); onEdit(); }
      }
    });
    $('#prompt').addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') send(); });
    $('#send').onclick = send;
    $('#attach-current').onclick = () => { if (S.active) { S.attach.add(S.active); renderAttach(); } };
    $('#think').onchange = e => { S.think = e.target.checked; };
    $('#clear').onclick = () => { S.chat = []; renderChat(); };
    $('#save').onclick = save;
    document.querySelectorAll('[data-pane]').forEach(b => b.onclick = () => {
      document.querySelectorAll('[data-pane]').forEach(x => x.classList.toggle('on', x === b));
      $('#tree').style.display = b.dataset.pane === 'files' ? 'block' : 'none';
      $('#git').style.display = b.dataset.pane === 'git' ? 'block' : 'none';
      if (b.dataset.pane === 'git') refreshGit();
    });
    window.addEventListener('beforeunload', e => {
      if ([...S.files.values()].some(f => f.text !== f.saved)) { e.preventDefault(); e.returnValue = ''; }
    });

    (guard(async () => {
      const info = await api('/api/info');
      $('#project').textContent = info.name;
      $('#engine').textContent = info.engine ? (info.engine.model + (info.engine.state === 'sleeping' ? ' (sleeping)' : '')) : 'model not running';
      $('#engine').className = info.engine ? 'ok' : 'off';
      S.git = info.git; renderGit();
      await loadDir(''); renderTree(); showFile(); renderChat(); renderAttach();
      setInterval(guard(async () => {
        const i = await api('/api/info');
        $('#engine').textContent = i.engine ? (i.engine.model + (i.engine.state === 'sleeping' ? ' (sleeping)' : '')) : 'model not running';
        $('#engine').className = i.engine ? 'ok' : 'off';
      }), 10000);
    }))();
  })();
}
