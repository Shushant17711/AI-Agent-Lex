// Lex web UI. No framework: one websocket, a handful of fetches, plain DOM.

const $ = (sel, root = document) => root.querySelector(sel);
const params = new URLSearchParams(location.search);
// The index route itself is token-gated, so the token stays in the URL (reloads keep working).
const TOKEN = params.get("t") || "";

const LRM = "\u200e"; // keeps "/" at the start of right-aligned-ellipsis paths
const AGENT_LETTER = { planner: "P", coder: "C", reviewer: "R" };
const AGENT_NAME = { planner: "Planner", coder: "Coder", reviewer: "Reviewer" };
const STATUS_TEXT = {
  completed: "Completed", needs_attention: "Needs attention", failed: "Failed",
  cancelled: "Stopped", interrupted: "Interrupted",
};

const state = {
  ws: null,
  active: false,
  replay: false,       // true while showing a past run (read-only)
  runId: null,
  steps: [],
  changes: new Map(),  // path -> change
  tools: new Map(),    // call id -> element
  approvals: new Map(),
  settings: null,
  lastAgent: null,
  treeOpen: new Set(),
  treeEntries: [],
};

// ---------------------------------------------------------------- utilities

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function el(tag, attrs = {}, html = "") {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else node.setAttribute(k, v);
  }
  if (html) node.innerHTML = html;
  return node;
}

async function api(path) {
  const r = await fetch(path, { headers: { "x-lex-token": TOKEN } });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), 6000);
}

function fmtTokens(n) {
  if (!n) return "0";
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
}

// Small, safe markdown: escape first, then add structure.
function md(src) {
  const blocks = [];
  let text = esc(src || "").replace(/```[\w+-]*\n([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const line of text.split("\n")) {
    let m;
    if ((m = line.match(/^\u0000(\d+)\u0000$/))) { closeList(); out.push(blocks[+m[1]]); }
    else if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { closeList(); out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); }
    else if ((m = line.match(/^\s*[-*]\s+(.*)$/))) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if (!line.trim()) { closeList(); out.push(""); }
    else { closeList(); out.push(`<p>${inline(line)}</p>`); }
  }
  closeList();
  return out.join("\n").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[+i]);
}

function diffHtml(diff) {
  return (diff || "(no textual change)").split("\n").map((l) => {
    let cls = "";
    if (l.startsWith("+++") || l.startsWith("---")) cls = "meta";
    else if (l.startsWith("@@")) cls = "hunk";
    else if (l.startsWith("+")) cls = "add";
    else if (l.startsWith("-")) cls = "del";
    return `<span class="l ${cls}">${esc(l) || " "}</span>`;
  }).join("");
}

function argPreview(name, args) {
  if (!args) return "";
  if (name === "run_command") return args.command ?? "";
  if (name === "search") return `/${args.pattern ?? ""}/ in ${args.path ?? "."}`;
  if (args.path) return args.path;
  return Object.entries(args).map(([k, v]) => `${k}=${String(v).slice(0, 40)}`).join(" ");
}

// ---------------------------------------------------------------- timeline

const timeline = $("#timeline");

function stickToBottom(fn) {
  const near = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 140;
  fn();
  if (near) timeline.scrollTop = timeline.scrollHeight;
}

function add(node) {
  node.classList.add("item");
  stickToBottom(() => timeline.appendChild(node));
  return node;
}

function resetView() {
  timeline.innerHTML = "";
  state.steps = [];
  state.changes.clear();
  state.tools.clear();
  state.approvals.clear();
  state.lastAgent = null;
  renderSteps();
  $("#approach").textContent = "No plan yet.";
  renderChanges();
  setRail(null, false);
  $("#meter").textContent = "";
  closeViewer();
}

function setRail(agent, finished) {
  const order = ["planner", "coder", "reviewer"];
  const idx = order.indexOf(agent);
  for (const node of document.querySelectorAll(".rail .node")) {
    const i = order.indexOf(node.dataset.agent);
    node.classList.toggle("active", !finished && i === idx);
    node.classList.toggle("done", finished ? idx >= 0 && i <= idx : i < idx);
  }
}

function handle(e) {
  const d = e.data || {};
  const agent = e.agent;
  switch (e.type) {
    case "run_started": {
      resetView();
      state.runId = d.id;
      const head = el("div", { class: "task-head" });
      head.innerHTML = `<h1>${esc(d.task)}</h1><div class="m">${esc(d.provider)}/${esc(d.model)} · ${new Date(e.ts * 1000).toLocaleString()}</div>`;
      add(head);
      highlightRun();
      break;
    }
    case "phase": {
      state.lastAgent = agent;
      setRail(agent, false);
      const p = el("div", { class: "phase", dataset: { agent: agent || "none" } });
      p.innerHTML = `<h3>${esc(d.title)}</h3><span class="who">${esc(AGENT_NAME[agent] || "")}</span>`;
      add(p);
      break;
    }
    case "message": {
      if (!d.text) break;
      const m = el("div", { class: "msg", dataset: { agent: agent || "none" } });
      m.innerHTML = `<div class="avatar">${AGENT_LETTER[agent] || "L"}</div><div class="prose">${md(d.text)}</div>`;
      add(m);
      break;
    }
    case "tool_call": {
      if (["submit_plan", "submit_review", "finish", "update_step"].includes(d.name)) break;
      const t = el("details", { class: "tool pending", dataset: { agent: agent || "none" } });
      t.innerHTML = `<summary><span class="g"></span><span class="n">${esc(d.name)}</span><span class="a">${esc(argPreview(d.name, d.args))}</span></summary><pre></pre>`;
      state.tools.set(d.id, t);
      add(t);
      break;
    }
    case "tool_result": {
      const t = state.tools.get(d.id);
      if (!t) break;
      t.classList.remove("pending");
      t.classList.add(d.ok ? "ok" : "bad");
      $("pre", t).textContent = d.preview || "(no output)";
      break;
    }
    case "plan": {
      state.steps = (d.steps || []).map((s) => ({ ...s }));
      $("#approach").innerHTML = d.approach ? md(d.approach) : "";
      renderSteps();
      if (state.steps.length) {
        const c = el("div", { class: "card", dataset: { agent: "planner" } });
        c.innerHTML = `<div class="card-head"><span class="title">Plan</span><span class="sub">${state.steps.length} step${state.steps.length > 1 ? "s" : ""}</span></div>
          <div class="card-body prose">${d.approach ? `<p><em>${esc(d.approach)}</em></p>` : ""}<ol>${state.steps.map((s) => `<li>${esc(s.text)}</li>`).join("")}</ol></div>`;
        add(c);
      }
      break;
    }
    case "step": {
      const s = state.steps[d.index];
      if (s) { s.status = d.status; s.note = d.note || ""; renderSteps(); }
      break;
    }
    case "file_changed": {
      state.changes.set(d.path, d);
      renderChanges();
      if (!$("#tab-files").classList.contains("hidden")) loadTree();
      if ($("#viewer").dataset.path === d.path) showChange(d.path);
      break;
    }
    case "approval_request": approvalCard(d, agent); break;
    case "approval_resolved": resolveApproval(d.id, d.allowed); break;
    case "review": {
      const c = el("div", { class: `card review ${d.approved ? "ok" : "no"}`, dataset: { agent: "reviewer" } });
      const issues = (d.issues || []).map((i) => `<li>${esc(i)}</li>`).join("");
      c.innerHTML = `<div class="card-head"><span class="title">Review${d.round > 1 ? ` · round ${d.round}` : ""}</span>
        <span class="pill" style="color:var(${d.approved ? "--green" : "--amber"})">${d.approved ? "approved" : "changes requested"}</span></div>
        <div class="card-body prose"><p>${esc(d.summary)}</p>${issues ? `<ul>${issues}</ul>` : ""}</div>`;
      add(c);
      break;
    }
    case "usage":
      if (d.input_tokens || d.output_tokens) $("#meter").textContent = `${fmtTokens(d.input_tokens)} in · ${fmtTokens(d.output_tokens)} out`;
      break;
    case "error": {
      add(el("div", { class: "error-line" }, `! ${esc(d.message)}`));
      break;
    }
    case "run_finished": finishCard(d); break;
  }
}

function approvalCard(d, agent) {
  const c = el("div", { class: "card approval", dataset: { agent: agent || "none" } });
  const body = d.kind === "write"
    ? `<pre class="diff">${diffHtml(d.detail)}</pre>`
    : `<pre class="card-body cmd">${esc(d.detail)}</pre>`;
  c.innerHTML = `<div class="card-head"><span class="title">${esc(d.title)}</span>
      <span class="sub">${esc(AGENT_NAME[agent] || "")} is asking</span>
      ${d.reason ? `<span class="warn">⚠ ${esc(d.reason)}</span>` : ""}</div>
    ${body}
    <div class="card-foot">
      <button class="btn small primary" data-a="allow">Allow</button>
      ${d.reason ? "" : `<button class="btn small" data-a="always">Always for this run</button>`}
      <button class="btn small ghost danger" data-a="deny">Deny</button>
      <span class="sub muted" style="margin-left:auto;font-size:12px">${d.kind === "write" ? "File change" : "Shell command"}</span>
    </div>`;
  c.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    const a = b.dataset.a;
    send({ kind: "approve", id: d.id, allow: a !== "deny", always: a === "always", scope: d.kind });
    c.querySelectorAll("button").forEach((x) => (x.disabled = true));
  }));
  if (state.replay) c.querySelectorAll("button").forEach((x) => (x.disabled = true));
  state.approvals.set(d.id, c);
  add(c);
  if (!state.replay) { $("button[data-a=allow]", c).focus({ preventScroll: true }); document.title = "● Lex — approval needed"; }
}

function resolveApproval(id, allowed) {
  const c = state.approvals.get(id);
  if (!c) return;
  c.classList.add("resolved");
  const foot = $(".card-foot", c);
  foot.innerHTML = `<span style="color:var(${allowed ? "--green" : "--red"})">${allowed ? "✓ Allowed" : "✗ Denied"}</span>`;
  document.title = "Lex";
}

function finishCard(d) {
  setRail(state.lastAgent || "reviewer", true);
  const color = { completed: "--green", needs_attention: "--amber", failed: "--red" }[d.status] || "--ink-3";
  const files = (d.changes || []).map((ch) =>
    `<li data-path="${esc(ch.path)}"><span style="color:var(${ch.status === "added" ? "--green" : ch.status === "deleted" ? "--red" : "--amber"})">${ch.status[0].toUpperCase()}</span><span style="flex:1">${esc(ch.path)}</span><span class="plus">+${ch.added}</span><span class="minus">-${ch.removed}</span></li>`).join("");
  const c = el("div", { class: "card final" });
  c.innerHTML = `<div class="card-head"><span class="verdict" style="color:var(${color})">${esc(STATUS_TEXT[d.status] || d.status)}</span>
      ${d.input_tokens || d.output_tokens ? `<span class="tokens" style="margin-left:auto">${fmtTokens(d.input_tokens)} in · ${fmtTokens(d.output_tokens)} out</span>` : ""}</div>
    <div class="card-body prose">${md(d.summary || "No summary.")}${files ? `<ul class="files">${files}</ul>` : ""}</div>`;
  c.querySelectorAll(".files li").forEach((li) => li.addEventListener("click", () => showChange(li.dataset.path)));
  add(c);
  for (const ch of d.changes || []) state.changes.set(ch.path, ch);
  renderChanges();
  if (!state.replay) { setActive(false); loadRuns(); }
}

// ---------------------------------------------------------------- side panes

function renderSteps() {
  const ol = $("#steps");
  ol.innerHTML = "";
  for (const s of state.steps) {
    const li = el("li", { class: s.status || "pending" });
    li.innerHTML = `${esc(s.text)}${s.note ? `<span class="note">${esc(s.note)}</span>` : ""}`;
    ol.appendChild(li);
  }
}

function renderChanges() {
  const ul = $("#changes");
  $("#change-count").textContent = state.changes.size;
  ul.innerHTML = "";
  if (!state.changes.size) { ul.innerHTML = `<li class="muted">Nothing changed yet.</li>`; return; }
  for (const [path, ch] of state.changes) {
    const li = el("li", { dataset: { path } });
    li.innerHTML = `<span class="s ${ch.status}">${ch.status[0].toUpperCase()}</span><span class="p">${LRM}${esc(path)}${LRM}</span><span class="plus">+${ch.added}</span><span class="minus">-${ch.removed}</span>`;
    li.classList.toggle("sel", $("#viewer").dataset.path === path);
    li.addEventListener("click", () => showChange(path));
    ul.appendChild(li);
  }
}

function openViewer(title, html, path) {
  const v = $("#viewer");
  v.classList.remove("hidden");
  v.dataset.path = path || "";
  $("#viewer-title").textContent = title;
  const body = $("#viewer-body");
  body.innerHTML = html;
  body.scrollTop = 0;
  renderChanges();
}

function closeViewer() {
  const v = $("#viewer");
  v.classList.add("hidden");
  v.dataset.path = "";
}

function showChange(path) {
  const ch = state.changes.get(path);
  if (!ch) return showFile(path);
  openViewer(`${path} · diff`, `<div class="diff">${diffHtml(ch.diff)}</div>`, path);
}

async function showFile(path) {
  try {
    const r = await api(`/api/file?path=${encodeURIComponent(path)}`);
    if (r.error) return toast(r.error);
    const lines = r.content.split("\n").map((l) => `<span class="ln">${esc(l) || " "}</span>`).join("");
    openViewer(path, lines, path);
  } catch (err) { toast(String(err)); }
}

async function loadTree() {
  try {
    state.treeEntries = (await api("/api/tree")).entries;
    renderTree();
  } catch (err) { toast(`Could not list files: ${err}`); }
}

function renderTree() {
  const ul = $("#tree");
  ul.innerHTML = "";
  for (const { path, dir } of state.treeEntries) {
    const parts = path.split("/");
    // hidden unless every ancestor directory is open
    let visible = true;
    for (let i = 1; i < parts.length; i++) {
      if (!state.treeOpen.has(parts.slice(0, i).join("/"))) { visible = false; break; }
    }
    if (!visible) continue;
    const li = el("li", { class: dir ? `dir${state.treeOpen.has(path) ? " open" : ""}` : "" });
    li.style.paddingLeft = `${16 + (parts.length - 1) * 14}px`;
    li.textContent = parts[parts.length - 1];
    li.title = path;
    if (state.changes.has(path)) li.classList.add("changed");
    li.addEventListener("click", () => {
      if (dir) {
        state.treeOpen.has(path) ? state.treeOpen.delete(path) : state.treeOpen.add(path);
        renderTree();
      } else showFile(path);
    });
    ul.appendChild(li);
  }
  if (!ul.children.length) ul.innerHTML = `<li class="muted">Empty workspace.</li>`;
}

async function loadRuns() {
  try {
    const { runs } = await api("/api/runs");
    const ul = $("#runs");
    ul.innerHTML = "";
    if (!runs.length) { ul.innerHTML = `<li class="muted" style="cursor:default">No runs yet.</li>`; return; }
    for (const r of runs) {
      const li = el("li", { dataset: { id: r.id } });
      const when = r.started ? new Date(r.started * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
      li.innerHTML = `<span class="t">${esc(r.task)}</span><span class="m"><span class="st-${esc(r.status)}">${esc(STATUS_TEXT[r.status] || r.status)}</span><span>${esc(when)}</span><span>${r.changes} file${r.changes === 1 ? "" : "s"}</span></span>`;
      li.addEventListener("click", () => replay(r.id));
      ul.appendChild(li);
    }
    highlightRun();
  } catch (err) { toast(`Could not load history: ${err}`); }
}

function highlightRun() {
  document.querySelectorAll("#runs li").forEach((li) => li.classList.toggle("current", li.dataset.id === state.runId));
}

async function replay(id) {
  if (state.active) return toast("A run is in progress. Stop it first to browse history.");
  try {
    const { events } = await api(`/api/runs/${encodeURIComponent(id)}`);
    state.replay = true;
    for (const e of events) handle(e);
    if (!events.some((e) => e.type === "run_finished")) {
      setRail(state.lastAgent, true);
      add(el("div", { class: "error-line" }, "This run was interrupted before it finished."));
    }
  } catch (err) { toast(String(err)); }
  finally { state.replay = false; }
}

// ---------------------------------------------------------------- run control

function setActive(on) {
  state.active = on;
  $("#run-btn").disabled = on;
  $("#task").disabled = on;
  $("#stop-btn").classList.toggle("hidden", !on);
  if (!on) document.title = "Lex";
}

function send(msg) {
  if (state.ws?.readyState === WebSocket.OPEN) { state.ws.send(JSON.stringify(msg)); return true; }
  toast("Not connected to Lex. Is the server still running?");
  return false;
}

function startRun() {
  const task = $("#task").value.trim();
  if (!task) { $("#task").focus(); return; }
  if (state.settings && !state.settings.has_api_key && state.settings.provider !== "openai") {
    toast("No API key found. Set GEMINI_API_KEY and restart `lex ui`.");
    return;
  }
  const sent = send({
    kind: "start", task,
    plan: $("#opt-plan").checked, review: $("#opt-review").checked,
    approve_writes: $("#opt-writes").value, approve_commands: $("#opt-commands").value,
  });
  if (sent) setActive(true);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws?t=${encodeURIComponent(TOKEN)}`);
  state.ws = ws;
  const conn = $("#conn");
  ws.onopen = () => { conn.className = "conn on"; $("span", conn).textContent = "connected"; };
  ws.onclose = () => {
    conn.className = "conn off"; $("span", conn).textContent = "offline";
    setTimeout(connect, 2000);
  };
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.kind === "event") {
      if (msg.event.type === "run_started") setActive(true);
      handle(msg.event);
    } else if (msg.kind === "error") {
      toast(msg.message);
      setActive(false);
    }
  };
}

async function init() {
  if (!TOKEN) {
    timeline.innerHTML = `<div class="empty"><h1>Missing <em>access token</em></h1><p>Open Lex with the full link printed in your terminal by <code>lex ui</code>.</p></div>`;
    return;
  }
  try {
    const s = await api("/api/state");
    state.settings = s.settings;
    $("#workspace").textContent = LRM + s.workspace + LRM;
    $("#workspace").title = s.workspace;
    $("#model-chip").textContent = `${s.settings.provider} · ${s.settings.model}`;
    $("#opt-plan").checked = s.settings.plan;
    $("#opt-review").checked = s.settings.review;
    $("#opt-writes").value = s.settings.approve_writes;
    $("#opt-commands").value = s.settings.approve_commands;
    if (!s.settings.has_api_key) {
      const n = $("#key-notice");
      n.innerHTML = s.settings.provider === "gemini"
        ? "No Gemini key found. Put <code>GEMINI_API_KEY=…</code> in <code>.env</code> or your shell, then restart <code>lex ui</code>."
        : "No API key found. Local servers like Ollama don't need one; hosted ones need <code>OPENAI_API_KEY</code>.";
      n.classList.remove("hidden");
    }
    if (s.current) {
      for (const e of s.current.events) handle(e);
      setActive(s.current.active);
    }
  } catch (err) {
    toast(`Could not reach Lex: ${err}`);
  }
  loadRuns();
  connect();
}

// ---------------------------------------------------------------- wiring

$("#composer").addEventListener("submit", (e) => { e.preventDefault(); startRun(); });
$("#task").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); startRun(); }
});
$("#stop-btn").addEventListener("click", () => send({ kind: "cancel" }));
$("#refresh-runs").addEventListener("click", loadRuns);
$("#viewer-close").addEventListener("click", () => { closeViewer(); renderChanges(); });
$("#ideas").addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") return;
  $("#task").value = e.target.textContent;
  $("#task").focus();
});
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
  $("#tab-changes").classList.toggle("hidden", tab.dataset.tab !== "changes");
  $("#tab-files").classList.toggle("hidden", tab.dataset.tab !== "files");
  if (tab.dataset.tab === "files") loadTree();
}));

init();
