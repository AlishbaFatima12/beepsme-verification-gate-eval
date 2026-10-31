// Agent Report Explorer. Reads the repository's data files and recomputes
// everything in the browser; nothing on the page is typed by hand.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Minimal RFC 4180 CSV parser (quoted fields, doubled quotes, commas inside quotes).
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((v) => v !== "")) rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows;
  return body.map((r) => Object.fromEntries(head.map((h, j) => [h, r[j] ?? ""])));
}

async function load() {
  const csv = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(p); return r.text(); }).then(parseCSV);
  const [tasks, gt, gate, sample, second, results, manifest, spoken] = await Promise.all([
    csv("data/tasks.csv"), csv("data/ground_truth.csv"), csv("data/gate_outputs.csv"),
    csv("data/second_rater_sample.csv"), csv("data/second_rater_labels.csv"),
    fetch("results/results.json").then((r) => r.json()),
    fetch("evidence/manifest.json").then((r) => (r.ok ? r.json() : { screenshots: {} })),
    fetch("evidence/agent_output.json").then((r) => (r.ok ? r.json() : { outputs: {} })),
  ]);
  const byId = (rows) => Object.fromEntries(rows.map((r) => [r.task_id, r]));
  const G = byId(gt), O = byId(gate), S = byId(second);
  const sampled = new Set(sample.map((r) => r.task_id));
  const all = tasks.map((t) => {
    const g = G[t.task_id], o = O[t.task_id];
    const scored = Boolean(g && o);
    const row = { ...t, scored, sampled: sampled.has(t.task_id), second: S[t.task_id]?.world_state_success ?? "",
      shot: manifest.screenshots[t.task_id] ?? "", said: spoken.outputs[t.task_id] ?? null };
    if (scored) {
      Object.assign(row, {
        truth: +g.world_state_success, achievable: +g.achievable, how: g.how_verified, notes: g.notes,
        gate: +o.gate_decision, reason: o.gate_reason, claim: +o.logged_claim, conf: +o.proxy_a_confidence,
      });
      row.cell = row.gate ? (row.truth ? "TP" : "FP") : (row.truth ? "FN" : "TN");
    }
    return row;
  });
  return { all, scored: all.filter((r) => r.scored), results };
}

function confusion(rows, predict) {
  const c = { TP: 0, FP: 0, TN: 0, FN: 0 };
  for (const r of rows) {
    const p = predict(r) ? 1 : 0;
    c[p ? (r.truth ? "TP" : "FP") : (r.truth ? "FN" : "TN")]++;
  }
  return c;
}

const pct = (x) => `${Math.round(x * 100)}%`;

function tiles({ scored, results }) {
  const g = results.gate_decision, p = results.proxy_a, s = results.second_rater, d = results.data;
  const items = [
    [pct(g.accuracy), `of trials where the agent’s verdict was right (${g.TP + g.TN} of ${g.n})`],
    [`${g.FP} of ${d.world_state_failures}`, "failures the agent called “done”"],
    [`${g.FN} of ${d.world_state_successes}`, "successes the agent called “failed”"],
    [p.auc.toFixed(2), "ROC AUC of the agent’s self-confidence"],
    [s.cohen_kappa.toFixed(2), `Cohen’s kappa, LLM rater vs a blind human (${s.n_compared} trials)`],
    [`${scored.length} of ${d.tasks_preregistered}`, "pre-specified tasks scored"],
  ];
  $("tiles").innerHTML = items.map(([b, l]) =>
    `<div class="tile"><div class="big">${esc(b)}</div><div class="lab">${esc(l)}</div></div>`).join("");
}

// ---------------------------------------------------------------------------
// Decision-rule explorer and trade-off chart
// ---------------------------------------------------------------------------
function ruleExplorer({ scored }) {
  const nFail = scored.filter((r) => !r.truth).length;
  const nSucc = scored.length - nFail;
  const thresholds = [...new Set(scored.map((r) => r.conf))].sort((a, b) => a - b);
  const confPoint = (t) => ({ t, ...confusion(scored, (r) => r.conf >= t) });
  const points = thresholds.map(confPoint);
  const gate = confusion(scored, (r) => r.gate);
  const claim = confusion(scored, (r) => r.claim);

  const W = 440, H = 300, m = { l: 48, r: 16, t: 14, b: 42 };
  const x = (v) => m.l + (v / nFail) * (W - m.l - m.r);
  const y = (v) => H - m.b - (v / nSucc) * (H - m.t - m.b);
  const svg = $("tradeoff");
  const tip = $("tip");
  let html = '<g class="grid">';
  for (let v = 0; v <= nFail; v += 2) html += `<line x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${H - m.b}"/>`;
  for (let v = 0; v <= nSucc; v += 2) html += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`;
  html += "</g>";
  for (let v = 0; v <= nFail; v += 2) html += `<text x="${x(v)}" y="${H - m.b + 15}" text-anchor="middle">${v}</text>`;
  for (let v = 0; v <= nSucc; v += 2) html += `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  html += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">False “done” (of ${nFail} failures)</text>`;
  html += `<text transform="translate(13 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">False “failed” (of ${nSucc} successes)</text>`;
  html += `<circle cx="${x(0)}" cy="${y(0)}" r="5" fill="none" stroke="var(--ink-3)"/>`;
  html += `<text x="${x(0) + 9}" y="${y(0) - 7}">ideal</text>`;
  const sorted = [...points].sort((a, b) => a.FP - b.FP);
  html += `<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${sorted.map((p) => `${x(p.FP)},${y(p.FN)}`).join(" ")}"/>`;
  const dot = (px, py, color, shape, label, key) => {
    const ring = 'stroke="var(--card)" stroke-width="2"';
    const tag = shape === "sq"
      ? `<rect x="${px - 6}" y="${py - 6}" width="12" height="12" fill="${color}" ${ring}/>`
      : shape === "tri"
        ? `<path d="M${px},${py - 8} L${px + 7},${py + 5} L${px - 7},${py + 5} Z" fill="${color}" ${ring}/>`
        : `<circle cx="${px}" cy="${py}" r="5.5" fill="${color}" ${ring}/>`;
    return `<g class="pt" data-key="${key}" data-label="${esc(label)}">${tag}
      <circle cx="${px}" cy="${py}" r="14" fill="transparent"/></g>`;
  };
  points.forEach((p) => { html += dot(x(p.FP), y(p.FN), "var(--accent)", "c",
    `Self-confidence ≥ ${p.t}: ${p.FP} false done, ${p.FN} false failed`, `t${p.t}`); });
  html += dot(x(gate.FP), y(gate.FN), "var(--gate)", "sq", `Agent’s verdict: ${gate.FP} false done, ${gate.FN} false failed`, "gate");
  html += dot(x(claim.FP), y(claim.FN), "var(--claim)", "tri", `Logged claim: ${claim.FP} false done, ${claim.FN} false failed`, "claim");
  html += `<text class="lbl" x="${x(gate.FP) + 11}" y="${y(gate.FN) - 8}">agent’s verdict</text>`;
  html += `<text class="lbl" x="${x(claim.FP)}" y="${y(claim.FN) - 16}" text-anchor="end">logged claim</text>`;
  html += '<circle id="sel" r="11" fill="none" stroke="var(--ink)" stroke-width="2"/>';
  svg.innerHTML = html;

  svg.querySelectorAll(".pt").forEach((g) => {
    g.addEventListener("mousemove", (e) => {
      const box = svg.parentElement.getBoundingClientRect();
      tip.textContent = g.dataset.label;
      tip.hidden = false;
      tip.style.left = `${Math.min(e.clientX - box.left + 12, box.width - tip.offsetWidth - 4)}px`;
      tip.style.top = `${e.clientY - box.top - 34}px`;
    });
    g.addEventListener("mouseleave", () => { tip.hidden = true; });
  });

  const thr = $("thr");
  thr.max = Math.max(...thresholds) + 8;
  function update() {
    const rule = document.querySelector('input[name="rule"]:checked').value;
    thr.disabled = rule !== "conf";
    $("thrOut").textContent = thr.value;
    const t = +thr.value;
    const c = rule === "gate" ? gate : rule === "claim" ? claim : confusion(scored, (r) => r.conf >= t);
    for (const k of ["TP", "FP", "TN", "FN"]) $(`c${k}`).innerHTML = `${c[k]}<small>${k}</small>`;
    const n = scored.length;
    const name = rule === "gate" ? "The agent’s verdict" : rule === "claim" ? "The logged claim field"
      : `Accepting self-confidence ≥ ${t}`;
    $("readout").textContent = `${name} reports “done” on ${c.TP + c.FP} of ${n} trials and is right on ` +
      `${c.TP + c.TN} (${pct((c.TP + c.TN) / n)}). It makes ${c.FP} false “done” and ` +
      `${c.FN} false “failed” reports.`;
    const sel = $("sel");
    sel.setAttribute("cx", x(c.FP));
    sel.setAttribute("cy", y(c.FN));
  }
  document.querySelectorAll('input[name="rule"]').forEach((el) => el.addEventListener("change", update));
  thr.addEventListener("input", () => {
    document.querySelector('input[name="rule"][value="conf"]').checked = true;
    update();
  });
  update();
}

// ---------------------------------------------------------------------------
// Rater agreement
// ---------------------------------------------------------------------------
function raters({ all, results }) {
  const s = results.second_rater;
  $("raterSummary").textContent =
    `An LLM rater labelled all ${results.data.tasks_scored} trials. A human then labelled a seeded random ` +
    `${s.n_sampled} without seeing the agent’s report, its confidence or the first labels. ` +
    `${s.n_cannot_tell} could not be decided from the evidence; on the other ${s.n_compared} the raters agreed on ` +
    `${s.both_success + s.both_failure} (Cohen’s kappa ${s.cohen_kappa.toFixed(2)}).`;
  const lab = (v) => v === "1" || v === 1 ? '<span class="chip ok">success</span>'
    : v === "0" || v === 0 ? '<span class="chip err">failure</span>' : '<span class="chip muted">cannot tell</span>';
  const rows = all.filter((r) => r.sampled);
  $("raterTable").innerHTML = "<thead><tr><th>Task</th><th>Command</th><th>LLM rater</th><th>Human rater</th>" +
    "<th>Agree?</th></tr></thead><tbody>" + rows.map((r) => {
      const agree = r.second === "cannot_tell" ? "–" : (+r.second === r.truth ? "yes" : "<b>no</b>");
      return `<tr><td>${esc(r.task_id.replace("EVAL-", ""))}</td><td>${esc(r.description)}</td>` +
        `<td>${lab(r.truth)}</td><td>${lab(r.second === "cannot_tell" ? "?" : r.second)}</td><td>${agree}</td></tr>`;
    }).join("") + "</tbody>";
}

// ---------------------------------------------------------------------------
// Agent output: replies with translations, and the homework solution files
// ---------------------------------------------------------------------------
function saidBlock(o) {
  const urdu = [o.reply, ...(o.speech || [])].filter(Boolean);
  return (urdu.length ? urdu.map((u) => `<div class="orig" dir="auto" lang="ur">${md(u, true)}</div>`).join("")
    : '<div class="orig muted">(nothing logged)</div>') +
    `<div class="en"><span class="en-tag">English</span> ${esc(o.english)}</div>`;
}

// Small Markdown renderer for the agent's solution files: headings, bold,
// italics, numbered lists, code blocks and the &middot; entity. Escapes first.
function md(text, inline = false) {
  const fmt = (t) => esc(t).replace(/&amp;middot;/g, "&middot;")
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|\W)_(.+?)_(?=\W|$)/g, "$1<i>$2</i>");
  if (inline) return fmt(text);
  const out = [];
  let code = null, list = null;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("```")) {
      if (code) { out.push(`<pre>${esc(code.join("\n"))}</pre>`); code = null; } else code = [];
      continue;
    }
    if (code) { code.push(line); continue; }
    const li = line.match(/^\s*\d+\.\s+(.*)$/);
    if (li) { (list ??= []).push(`<li>${fmt(li[1])}</li>`); continue; }
    if (list) { out.push(`<ol>${list.join("")}</ol>`); list = null; }
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) out.push(`<h${h[1].length + 3}>${fmt(h[2])}</h${h[1].length + 3}>`);
    else if (line.trim()) out.push(`<p>${fmt(line)}</p>`);
  }
  if (list) out.push(`<ol>${list.join("")}</ol>`);
  return out.join("");
}

function evidence({ all }) {
  const outcome = (r) => `<span class="chip ${r.cell === "TP" || r.cell === "TN" ? "ok" : "err"}">${esc(r.cell)}</span>
    <span class="muted">task ${r.truth ? "succeeded" : "failed"}; agent said ${r.gate ? "done" : "failed"}</span>`;
  const head = (r) => `<b>${esc(r.task_id.replace("EVAL-", ""))}</b> “${esc(r.description)}”`;

  const order = ["homework", "read", "translate", "teach"];
  const shots = all.filter((r) => r.shot)
    .sort((a, b) => order.indexOf(a.skill_domain) - order.indexOf(b.skill_domain));
  $("nShots").textContent = `(${shots.length})`;
  $("gallery").innerHTML = shots.map((r) => {
    const src = `evidence/screenshots/${esc(r.task_id)}.png`;
    return `<figure><a href="${src}" target="_blank" rel="noopener">
      <img src="${src}" loading="lazy" alt="${esc(r.task_id)}: ${esc(r.shot)}"></a>
      <figcaption>${head(r)}<br>${esc(r.shot)}<br>${outcome(r)}</figcaption></figure>`;
  }).join("");

  const hw = all.filter((r) => r.scored && r.skill_domain === "homework");
  $("nMath").textContent = `(${hw.length})`;
  $("solutions").innerHTML = hw.map((r) => `<article class="sol" id="sol-${esc(r.task_id)}">
    <header>${head(r)} ${outcome(r)}</header><div class="sol-body">Loading…</div></article>`).join("");
  hw.forEach((r) => fetch(`data/artifacts/${r.task_id}.md`).then((x) => x.text()).then((t) => {
    document.querySelector(`#sol-${r.task_id} .sol-body`).innerHTML = md(t);
  }));

  const said = all.filter((r) => r.said && r.skill_domain !== "homework");
  $("nSaid").textContent = `(${said.length})`;
  $("said").innerHTML = said.map((r) => `<article class="said-card">
    <header>${head(r)} ${outcome(r)}</header>${saidBlock(r.said)}</article>`).join("");
}

// ---------------------------------------------------------------------------
// Trial browser
// ---------------------------------------------------------------------------
function trials({ all }) {
  const fill = (id, values) => { $(id).innerHTML += values.map((v) => `<option>${esc(v)}</option>`).join(""); };
  fill("fSkill", [...new Set(all.map((r) => r.skill_domain))]);
  fill("fRegime", [...new Set(all.map((r) => r.intended_regime))]);
  const cellChip = (r) => !r.scored ? `<span class="chip muted">${esc(r.study_status.replace("_", " "))}</span>`
    : `<span class="chip ${r.cell === "TP" || r.cell === "TN" ? "ok" : "err"}">${r.cell}</span>`;
  function render() {
    const fs = $("fSkill").value, fr = $("fRegime").value, fc = $("fCell").value;
    const rows = all.filter((r) => (!fs || r.skill_domain === fs) && (!fr || r.intended_regime === fr) &&
      (!fc || (fc === "unscored" ? !r.scored : r.cell === fc)));
    $("count").textContent = `${rows.length} of ${all.length} tasks`;
    $("trialTable").innerHTML = "<thead><tr><th>Task</th><th>Command</th><th>Designed to</th>" +
      "<th>Truth</th><th>Verdict</th><th>Confidence</th><th>Outcome</th><th>Screenshot</th></tr></thead><tbody>" +
      rows.map((r, i) => `<tr class="row" tabindex="0" data-i="${i}">` +
        `<td>${esc(r.task_id.replace("EVAL-", ""))}</td><td>${esc(r.description)}</td>` +
        `<td>${esc(r.intended_regime.replace(/_/g, " "))}</td>` +
        `<td>${r.scored ? (r.truth ? "success" : "failure") : ""}</td>` +
        `<td>${r.scored ? (r.gate ? "done" : "failed") : ""}</td>` +
        `<td>${r.scored ? r.conf : ""}</td><td>${cellChip(r)}</td>` +
        `<td>${r.shot ? '<span class="chip">view</span>' : ""}</td></tr>`).join("") + "</tbody>";
    $("trialTable").querySelectorAll("tr.row").forEach((tr) => {
      const toggle = () => {
        const next = tr.nextElementSibling;
        if (next && next.classList.contains("detail")) { next.remove(); return; }
        const r = rows[+tr.dataset.i];
        const det = document.createElement("tr");
        det.className = "detail";
        det.innerHTML = `<td colspan="8"><dl>
          <dt>Setup</dt><dd>${esc(r.precondition || "none")}</dd>
          <dt>Oracle (written in June)</dt><dd>${esc(r.oracle)}</dd>
          ${r.scored ? `<dt>What the rater checked</dt><dd>${esc(r.how)}</dd>
          <dt>Achievable at all?</dt><dd>${r.achievable ? "yes" : "no"}</dd>
          <dt>Agent’s verdict</dt><dd>${r.gate ? "done" : "failed"}${r.reason ? ` (reason: ${esc(r.reason)})` : ""}</dd>
          <dt>Self-confidence</dt><dd>${r.conf} / 100</dd>` : `<dt>Status</dt><dd>${esc(r.study_status.replace("_", " "))}</dd>`}
          ${r.said ? `<dt>What the agent said</dt><dd>${saidBlock(r.said)}</dd>` : ""}
          ${r.shot ? `<dt>Screenshot</dt><dd><figure class="shot">
            <a href="evidence/screenshots/${esc(r.task_id)}.png" target="_blank" rel="noopener">
            <img src="evidence/screenshots/${esc(r.task_id)}.png" loading="lazy"
                 alt="Screenshot at the end of ${esc(r.task_id)}: ${esc(r.shot)}"></a>
            <figcaption>${esc(r.shot)}</figcaption></figure></dd>`
            : r.scored ? `<dt>Screenshot</dt><dd class="muted">Not published for this trial
              (<a href="https://github.com/AlishbaFatima12/beepsme-verification-gate-eval/blob/main/evidence/README.md">why</a>).</dd>` : ""}
        </dl></td>`;
        tr.after(det);
      };
      tr.addEventListener("click", toggle);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
    });
  }
  ["fSkill", "fRegime", "fCell"].forEach((id) => $(id).addEventListener("change", render));
  render();
}

load().then((data) => { tiles(data); ruleExplorer(data); raters(data); evidence(data); trials(data); })
  .catch((err) => {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<section><p>Could not load the data files (${esc(err.message)}). Open this page through GitHub Pages ` +
      "or a local web server rather than as a file.</p></section>");
  });

// Brand nav: transparent over the dark hero, solid once the page scrolls.
(() => {
  const nav = $("nav"), hero = $("top");
  const update = () => nav.classList.toggle("scrolled", window.scrollY > hero.offsetHeight - 80);
  window.addEventListener("scroll", update, { passive: true });
  update();
})();
