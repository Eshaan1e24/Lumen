/* lumen-app.js : UI helpers for the newer backend features (sample picker, trust panel, finding evidence, answer checks,
   forecast accuracy, notices, printable report).  Loaded at the end of <body>, after the inline script in index.html, whose
   helpers ($, esc, fmt, full, cellv, hasPlotly, Plotly, C, lay, cfg) it reuses.  Every function tolerates missing or null
   optional fields: an older or partial payload must never throw.  ALL dynamic text goes through esc(): CSV-derived text is hostile. */
(function () {
  'use strict';
  const q = s => document.querySelector(s);
  const isNum = v => typeof v === 'number' && Number.isFinite(v);
  const arr = v => Array.isArray(v) ? v : [];
  const cap = s => { s = String(s ?? ''); return s.charAt(0).toUpperCase() + s.slice(1) };
  const LX = window.LX = { state: { session: null, fc: null } };

  /* ---------- number and date formatting ---------- */
  const trim = (v, d) => String(+v.toFixed(d));
  function tnum(n, signed) {                                     // friendly: 1,234 / 2.87M / 12.5 / 0.034 ; optional explicit sign
    if (n == null || n === '' || !Number.isFinite(+n)) return '–';
    n = +n; const a = Math.abs(n); let t;
    if (a >= 1e9) t = trim(a / 1e9, 2) + 'B';
    else if (a >= 1e6) t = trim(a / 1e6, 2) + 'M';
    else if (a >= 1000) t = Math.round(a).toLocaleString();
    else if (a >= 100) t = String(Math.round(a));
    else if (a >= 1) t = trim(a, 1);
    else if (a === 0) t = '0';
    else t = String(+a.toPrecision(2));
    return (n < 0 ? '−' : signed && n > 0 ? '+' : '') + t;
  }
  function pct(x, signed) {                                      // 0.786 -> 79% ; tiny -> <1% ; no decimals (no false precision)
    if (x == null || !Number.isFinite(+x)) return '–';
    const v = +x * 100, a = Math.abs(v), sign = v < 0 ? '−' : signed && v > 0 ? '+' : '';
    return sign + (a === 0 ? '0%' : a < 1 ? '<1%' : Math.round(a).toLocaleString() + '%');
  }
  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function dateLabel(s) {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(s ?? ''));
    return m && MON[+m[2] - 1] ? `${+m[3]} ${MON[+m[2] - 1]} ${m[1]}` : String(s ?? '');
  }

  /* ---------- tiny inline icons (shape differs per state, so meaning never rests on colour alone) ---------- */
  const SV = (body, cls) => `<svg class="ico ${cls || ''}" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
  const ICO = {
    ok: SV('<path d="M3 8.5l3.2 3.2L13 4.8"/>'),
    bad: SV('<path d="M4 4l8 8M12 4l-8 8"/>'),
    na: SV('<path d="M4 8h8"/>'),
    info: SV('<circle cx="8" cy="8" r="6.2"/><path d="M8 7.3v3.9"/><circle cx="8" cy="4.9" r=".6" fill="currentColor"/>'),
    warn: SV('<path d="M8 2.3l6.3 11H1.7z"/><path d="M8 6.7v3"/><circle cx="8" cy="11.8" r=".6" fill="currentColor"/>'),
    half: SV('<circle cx="8" cy="8" r="6.2"/><path d="M8 1.8a6.2 6.2 0 0 0 0 12.4z" fill="currentColor"/>')
  };

  /* ---------- hero: trust panel (exact, deliberately modest wording) ---------- */
  function trustInner() {
    return '<ol class="trust-steps">'
      + '<li><span class="n" aria-hidden="true">1</span><p><b>Findings are computed in code.</b> Trends, spikes, what changed and forecasts are plain statistics you can check. The AI does not invent numbers.</p></li>'
      + '<li><span class="n" aria-hidden="true">2</span><p><b>Questions are checked three ways.</b> The AI\'s SQL is parsed and must be a single read-only query. A second, differently written query must return the same values. Every number, and every product or customer name from your data, in the explanation must appear in, or be calculated from, the result. If a check fails you will see it.</p></li>'
      + '<li><span class="n" aria-hidden="true">3</span><p><b>Forecasts are tested on your own history first,</b> and Lumen tells you whether they beat a simple guess.</p></li></ol>'
      + '<p class="trust-priv"><b>Privacy.</b> Your file is held in memory on this server and is not stored by Lumen (large uploads may use a temporary file while being read). When AI is on, Gemini receives your column names, a few sample values, your question, the query and the results or findings needed to answer. On Google\'s free tier prompts may be used to improve their products, so do not upload sensitive data to a shared demo. Self-host and leave the API key unset to keep everything on your own machine.</p>'
      + '<p class="trust-foot">Free and open source (MIT).</p>';
  }
  (function initTrust() {
    const t = q('#trust'); if (t) t.innerHTML = '<h2 id="trust-h">How Lumen keeps answers honest</h2>' + trustInner();
    const c = q('#trust-compact-body'); if (c) c.innerHTML = '<h3>How Lumen keeps answers honest</h3>' + trustInner();
  })();

  /* ---------- hero: sample picker ---------- */
  (async function initSamples() {
    const grid = q('#sample-grid'); if (!grid) return;
    let list = null;
    try { const r = await fetch('/api/samples'); if (r.ok) list = await r.json() } catch (e) { }
    if (!Array.isArray(list) || !list.length) list = [{ id: 'shop', title: 'Try sample shop data', blurb: 'Two years of orders for a student-run shop.', questions: [] }];
    grid.innerHTML = list.filter(s => s && s.id).map(s => {
      const n = arr(s.questions).length;
      return `<button type="button" class="sample-card" data-sample="${esc(s.id)}"><span class="sc-title">${esc(s.title || s.id)}</span><span class="sc-blurb">${esc(s.blurb || '')}</span>`
        + `<span class="sc-meta">${n ? n + ' ready-made question' + (n === 1 ? '' : 's') + ' · ' : ''}Open dashboard</span></button>`;
    }).join('');
    grid.addEventListener('click', e => {
      const b = e.target.closest('.sample-card'); if (!b || b.disabled) return;
      if (typeof window.loadSample === 'function') window.loadSample(b.dataset.sample);
    });
  })();

  /* ---------- narrative + notices ---------- */
  LX.sourceLabel = src => src === 'ai' ? 'Summary written by AI from the computed findings' : src === 'template' ? 'Summary built directly from the findings (no AI used)' : '';
  LX.noticesHtml = list => arr(list).filter(n => typeof n === 'string' && n.trim()).map(n =>
    `<div class="notice" role="status">${ICO.info}<span class="notice-t">${esc(n)}</span><button type="button" class="notice-x" aria-label="Dismiss this notice">Dismiss</button></div>`).join('');
  document.addEventListener('click', e => { const b = e.target.closest && e.target.closest('.notice-x'); if (b) { const n = b.closest('.notice'); if (n) n.remove() } });

  /* ---------- insights ---------- */
  const KIND = { change: 'What changed', anomaly: 'Spike/Drop', trend: 'Trend', driver: 'Driver', quality: 'Data quality' };
  const SEV = { warn: 'Needs attention', good: 'Good news', info: 'For information' };

  function changeEvidence(e) {
    const rows = arr(e.contributions).filter(c => c && typeof c === 'object');
    if (!rows.length && !isNum(e.change)) return '';
    let h = '';
    if (e.window) h += `<p class="ev-win">${esc(cap(e.window))}${e.dimension ? ', split by ' + esc(e.dimension) : ''}.</p>`;
    if (rows.length) {
      const sumCh = rows.reduce((s, c) => s + (isNum(c.change) ? c.change : 0), 0);
      const sumSh = rows.every(c => isNum(c.share_of_change)) ? rows.reduce((s, c) => s + c.share_of_change, 0) : null;
      const adds = typeof e.contributions_sum_to_change === 'boolean' ? e.contributions_sum_to_change
        : isNum(e.change) ? Math.abs(sumCh - e.change) <= 1e-6 * Math.max(1, Math.abs(e.change)) : null;
      h += '<div class="ev-scroll" tabindex="0" role="region" aria-label="Contribution table"><table class="evt"><thead><tr><th scope="col">Segment</th><th scope="col" class="r">Previous</th><th scope="col" class="r">Current</th><th scope="col" class="r">Change</th><th scope="col" class="r">Share</th></tr></thead><tbody>'
        + rows.map(c => `<tr><th scope="row">${esc(c.segment ?? '(blank)')}</th><td class="r">${tnum(c.previous)}</td><td class="r">${tnum(c.current)}</td><td class="r">${tnum(c.change, true)}</td><td class="r">${pct(c.share_of_change)}</td></tr>`).join('')
        + `</tbody><tfoot><tr><th scope="row">Total</th><td class="r">${tnum(e.previous_total)}</td><td class="r">${tnum(e.current_total)}</td><td class="r">${tnum(isNum(e.change) ? e.change : sumCh, true)}</td><td class="r">${sumSh == null ? '' : pct(sumSh)}</td></tr></tfoot></table></div>`;
      if (adds === true) h += `<p class="ev-ok">${ICO.ok}<span>The segment changes add up exactly to the total change.</span></p>`;
      else if (adds === false) h += `<p class="ev-warn">${ICO.warn}<span>The segments listed do not add up to the whole change.</span></p>`;
    }
    const pv = e.price_volume;
    if (pv && (isNum(pv.volume_effect) || isNum(pv.price_effect))) {
      h += `<p class="ev-pv"><b>Volume vs price:</b> ${tnum(pv.volume_effect, true)} from selling more or fewer units, ${tnum(pv.price_effect, true)} from the average price per unit.</p>`;
    }
    return h;
  }
  function anomalyEvidence(e) {
    if (!isNum(e.observed) && !isNum(e.expected)) return '';
    const diff = isNum(e.observed) && isNum(e.expected) ? e.observed - e.expected : null;
    let h = '<dl class="ev-dl">';
    if (e.date) h += `<dt>Day</dt><dd>${esc(dateLabel(e.date))}</dd>`;
    if (isNum(e.observed)) h += `<dt>Observed</dt><dd>${tnum(e.observed)}</dd>`;
    if (isNum(e.expected)) h += `<dt>Expected</dt><dd>${tnum(e.expected)}</dd>`;
    if (diff != null) h += `<dt>Difference</dt><dd>${tnum(diff, true)}${e.expected ? ' (' + pct(diff / Math.abs(e.expected), true) + ')' : ''}</dd>`;
    const sg = e.segment;
    if (sg && typeof sg === 'object' && sg.value != null) {
      h += `<dt>Driven by</dt><dd>${esc(sg.value)}${sg.dimension ? ' (' + esc(sg.dimension) + ')' : ''}${isNum(sg.change) ? ': ' + tnum(sg.change, true) + ' of the difference' : ''}</dd>`;
    }
    if (isNum(e.robust_score)) h += `<dt>Outlier score</dt><dd>${trim(e.robust_score, 1)} <span class="muted">(higher means more unusual)</span></dd>`;
    return h + '</dl>';
  }
  function recordsEvidence(e) {
    if (!Array.isArray(e.records) || !e.records.length || !Array.isArray(e.record_columns)) return '';
    const cell = (v, c) => isNum(v) && c === e.column ? tnum(v) : esc(v ?? '');
    return '<p class="ev-pv"><b>The largest records behind this:</b> check each one against your source.</p><div class="ev-scroll" tabindex="0" role="region" aria-label="Largest records"><table class="evt"><thead><tr>'
      + e.record_columns.map(c => `<th scope="col"${c === e.column ? ' class="r"' : ''}>${esc(c)}</th>`).join('') + '</tr></thead><tbody>'
      + e.records.map(r => '<tr>' + r.map((v, k) => `<td${e.record_columns[k] === e.column ? ' class="r"' : ''}>${cell(v, e.record_columns[k])}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>';
  }
  function evidenceHtml(i) {
    const e = i && i.evidence; if (!e || typeof e !== 'object') return '';
    const inner = i.kind === 'change' ? changeEvidence(e) : i.kind === 'anomaly' ? anomalyEvidence(e) : i.kind === 'quality' ? recordsEvidence(e) : '';
    return inner ? `<details class="nums"><summary>Show the numbers</summary><div class="ev">${inner}</div></details>` : '';
  }
  LX.insightHtml = i => {
    i = i && typeof i === 'object' ? i : {};
    const sev = SEV[i.severity] ? i.severity : 'info', kind = KIND[i.kind];
    let ev = ''; try { ev = evidenceHtml(i) } catch (err) { ev = '' }
    return `<div class="ins ${sev}">${kind ? `<span class="kind">${esc(kind)}</span>` : ''}<b><span class="sr">${SEV[sev]}: </span>${esc(i.title)}</b><p>${esc(i.detail)}</p>${ev}</div>`;
  };

  /* ---------- ask: status badge, note, caveats, "how this was worked out" ---------- */
  const STATUS = {
    checked: ['ok', 'checked', 'Checked: all checks passed'],
    partly: ['half', 'partly', 'Partly checked'],
    disagree: ['warn', 'disagree', 'Check this one: two independent queries disagree'],
    demo: ['info', 'demo', 'Sample answer: the stored query was run live on the data (no AI call)']
  };
  const statusOf = r => STATUS[r && r.status] ? r.status : (r && r.status) ? 'partly' : (r && r.mode === 'demo') ? 'demo' : (r && r.verified) ? 'checked' : 'partly';
  LX.statusBadge = r => {
    r = r || {}; const k = statusOf(r), s = STATUS[k];
    let why = '';
    if (k === 'partly' || k === 'disagree') {
      const bad = arr(r.checks).filter(c => c && c.ok !== true && c.label).map(c => `<li>${esc(c.label)}</li>`).join('');
      if (bad) why = `<ul class="why">${bad}</ul>`;
    }
    return `<div class="status st-${s[1]}"><span class="st-badge">${ICO[s[0]]}<span>${esc(s[2])}</span></span></div>${why}`;
  };
  LX.noteHtml = r => r && typeof r.note === 'string' && r.note.trim() ? `<p class="ans-note">${ICO.info}<span>${esc(r.note)}</span></p>` : '';
  LX.caveatsHtml = c => {
    if (Array.isArray(c)) { c = c.filter(x => typeof x === 'string' && x.trim()); return c.length ? `<ul class="note caveats">${c.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : '' }
    return typeof c === 'string' && c.trim() ? `<p class="note">${esc(c)}</p>` : '';
  };
  LX.detailsHtml = (r, cols, rows, showAll) => {
    r = r || {}; const checks = arr(r.checks), k = statusOf(r);
    let h = `<details class="how"${showAll || k === 'disagree' ? ' open' : ''}><summary>${showAll ? 'Result table and how this was worked out' : 'How this was worked out'}</summary>`;
    if (checks.length) {
      h += '<h4>Checks</h4><ul class="checks">' + checks.map(c => {
        c = c || {}; const st = c.ok === true ? ['ok', 'ok', 'Passed'] : c.ok === false ? ['bad', 'bad', 'Failed'] : ['na', 'na', 'Not run'];
        return `<li class="ck ${st[1]}">${ICO[st[0]]}<span><span class="sr">${st[2]}: </span>${esc(c.label)}</span></li>`;
      }).join('') + '</ul>';
    }
    if (r.sql) h += `<h4>SQL that was run</h4><pre tabindex="0">${esc(r.sql)}</pre>`;
    if (typeof r.check_sql === 'string' && r.check_sql.trim()) h += `<h4>Second query used for the cross-check</h4><pre tabindex="0">${esc(r.check_sql)}</pre>`;
    h += `<h4>Result</h4><div class="tbl-scroll" tabindex="0" role="region" aria-label="Result table"><table><thead><tr>${cols.map(c => `<th scope="col">${esc(c)}</th>`).join('')}</tr></thead><tbody>`
      + rows.slice(0, 50).map(x => `<tr>${arr(x).map(v => `<td>${esc(typeof v === 'number' && (Math.abs(v) >= 10000 || !Number.isInteger(v)) ? v.toLocaleString(undefined, { maximumFractionDigits: Math.abs(v) < 1 ? 4 : 2 }) : cellv(v))}</td>`).join('')}</tr>`).join('') + '</tbody></table></div>'
      + (rows.length > 50 ? `<p class="note">Showing the first 50 of ${rows.length.toLocaleString()} rows.</p>` : '');
    return h + '</details>';
  };

  /* ---------- forecast ---------- */
  LX.fcClear = () => {
    try { const g = q('#fchart'); if (g) { if (g.classList.contains('js-plotly-plot') && typeof Plotly !== 'undefined') Plotly.purge(g); g.hidden = true } } catch (e) { }
    const l = q('#fleg'), a = q('#facc'), n = q('#fnote');
    if (l) { l.hidden = true; l.innerHTML = '' } if (a) { a.hidden = true; a.innerHTML = '' } if (n) n.textContent = '';
    LX.state.fc = null;
  };
  const lcfirst = s => { s = String(s ?? ''); return s.charAt(0).toLowerCase() + s.slice(1) };
  function accHtml(r) {
    const bt = r.backtest && typeof r.backtest === 'object' ? r.backtest : {};
    let h = '<h3>How this forecast was tested</h3>';
    if (r.status === 'limited') h += `<div class="warnbox" role="note">${ICO.warn}<div><b>Limited history.</b> ${esc(typeof r.reason === 'string' && r.reason ? r.reason : 'There is too little history to test this forecast properly, so treat it as a rough guide.')}</div></div>`;
    h += '<dl class="acc-grid">';
    if (r.model_name) h += `<div><dt>Method</dt><dd>${esc(r.model_name)}</dd></div>`;
    if (isNum(bt.n_folds)) h += `<div><dt>Test runs</dt><dd>${bt.n_folds > 0 ? 'Tested on ' + bt.n_folds + ' past period' + (bt.n_folds === 1 ? '' : 's') : 'Not tested (too little history)'}</dd></div>`;
    if (typeof r.baseline_comparison === 'string' && r.baseline_comparison.trim()) {
      const t = lcfirst(r.baseline_comparison.trim()).replace(/\.$/, '');
      h += `<div class="wide"><dt>Compared with a simple guess</dt><dd>In tests on your own past data, ${esc(t)}.`
        + (/naive/i.test(r.baseline_comparison) ? ' <span class="muted">(“Naive” repeats the last value; “seasonal-naive” repeats the same period from the previous cycle, such as the same month last year.)</span>' : '') + '</dd></div>';
    }
    h += '</dl>';
    const cav = arr(r.caveats).filter(c => typeof c === 'string' && c.trim());
    if (cav.length) {
      h += '<h4>Things to keep in mind</h4><ul class="caveats">' + cav.slice(0, 2).map(c => `<li>${esc(c)}</li>`).join('') + '</ul>';
      if (cav.length > 2) h += `<details class="more"><summary>Show ${cav.length - 2} more</summary><ul class="caveats">${cav.slice(2).map(c => `<li>${esc(c)}</li>`).join('')}</ul></details>`;
    }
    return h + '<p class="acc-fixed">Past accuracy does not guarantee future accuracy. The shaded bands are ranges built from this model\'s own past errors.</p>';
  }
  const finite = (a, i) => isNum(a && a[i]);
  LX.drawForecast = (r, metric) => {
    r = r || {}; const hist = r.history || {}, f = r.forecast || {}, hx = arr(hist.x), hy = arr(hist.y), fx = arr(f.x), fy = arr(f.y);
    LX.state.fc = r;
    if (!hx.length || !fx.length) { LX.fcClear(); q('#ferr').textContent = typeof r.note === 'string' && r.note ? r.note : 'No forecast could be made for these columns.'; return }
    const g = q('#fchart'); g.hidden = false;
    const last = hx.length - 1, isDate = /^\d{4}-\d{2}-\d{2}/.test(String(hx[0]));
    const fr = String(r.freq || ''), daily = /^[DB]/.test(fr), weekly = /^W/.test(fr);
    const tf = daily ? '%d %b %Y' : weekly ? '%d %b' : '%b %Y', ht = daily || weekly ? '%{x|%d %b %Y}' : '%{x|%b %Y}';
    const lo95 = f.lower95 || f.lower, hi95 = f.upper95 || f.upper, lo80 = f.lower80, hi80 = f.upper80;
    const band = (lo, hi, color, name) => {
      if (!Array.isArray(lo) || !Array.isArray(hi)) return null;
      const ix = fx.map((_, i) => i).filter(i => finite(lo, i) && finite(hi, i)); if (!ix.length) return null;
      return { x: [...ix.map(i => fx[i]), ...ix.map(i => fx[i]).reverse()], y: [...ix.map(i => hi[i]), ...ix.map(i => lo[i]).reverse()], mode: 'lines', fill: 'toself', fillcolor: color, line: { width: 0 }, name, hoverinfo: 'skip', showlegend: false };
    };
    const two = Array.isArray(lo80) && Array.isArray(hi80) && Array.isArray(lo95) && Array.isArray(hi95);
    const b95 = band(lo95, hi95, 'rgba(199,127,0,.15)', two ? 'Wider range (95%)' : 'Likely range');
    const b80 = two ? band(lo80, hi80, 'rgba(199,127,0,.34)', 'Likely range (80%)') : null;
    const ftext = fx.map((x, i) => {
      let t = isNum(fy[i]) ? full(fy[i]) : '–';
      if (b80 && finite(lo80, i) && finite(hi80, i)) t += `<br>Likely range: ${full(lo80[i])} to ${full(hi80[i])}`;
      if (b95 && finite(lo95, i) && finite(hi95, i)) t += `<br>${two ? 'Wider range' : 'Range'}: ${full(lo95[i])} to ${full(hi95[i])}`;
      return t;
    });
    const hxl = isDate ? ht : '%{x}';
    const traces = [{ x: hx, y: hy, text: hy.map(full), mode: 'lines', name: 'Actual', line: { color: C.teal, width: 2.5 }, hovertemplate: hxl + ': %{text}<extra>Actual</extra>' }];
    if (b95) traces.push(b95); if (b80) traces.push(b80);
    traces.push({ x: [hx[last], ...fx], y: [hy[last], ...fy], text: [full(hy[last]), ...ftext], mode: 'lines', name: 'Forecast', line: { color: C.amber, width: 2.5, dash: 'dot' }, hovertemplate: hxl + ': %{text}<extra>Forecast</extra>' });
    Plotly.newPlot(g, traces, lay({
      xaxis: isDate ? { tickformat: tf } : {}, rest: {
        showlegend: false, margin: { l: 56, r: 16, t: 12, b: 36 },
        shapes: [{ type: 'line', x0: hx[last], x1: hx[last], yref: 'paper', y0: 0, y1: 1, line: { color: '#9AA7B1', width: 1, dash: 'dash' } }]
      }
    }), cfg);
    g.setAttribute('role', 'img'); g.setAttribute('aria-label', `Forecast of ${metric || 'the measure'}: ${r.note || ''}`.slice(0, 400));
    const leg = q('#fleg');
    leg.innerHTML = '<li><span class="sw sw-actual"></span>Actual</li><li><span class="sw sw-fc"></span>Forecast</li>'
      + (b80 ? '<li><span class="sw sw-b80"></span>Likely range (80%)</li>' : '') + (b95 ? `<li><span class="sw sw-b95"></span>${two ? 'Wider range (95%)' : 'Likely range'}</li>` : '');
    leg.hidden = false;
    q('#fnote').textContent = String(r.note || '').replace(/\b1 (month|week|day)s\b/g, '1 $1');
    const acc = q('#facc'); acc.innerHTML = accHtml(r); acc.hidden = false;
  };

  /* ---------- printable one-page report (window.print + @media print) ---------- */
  function reportHtml(img) {
    const d = LX.state.session; if (!d) return '';
    const nar = d.narrative || {}, p = d.profile || {}, fc = LX.state.fc, ins = arr(d.insights);
    const when = new Date().toLocaleDateString(undefined, { dateStyle: 'long' });
    let h = `<h1>Lumen report</h1><p class="r-meta">${esc(when)} · ${esc((+p.rows || 0).toLocaleString())} rows, ${arr(p.columns).length} columns. Main measure: ${esc(p.metric || 'none found')}.</p>`;
    h += `<h2>What matters most</h2><p>${esc(nar.summary || '')}</p>`;
    const src = LX.sourceLabel(nar.source); if (src) h += `<p class="r-small">${esc(src)}.</p>`;
    const k = arr(p.kpis).filter(x => x && x.label);
    if (k.length) h += '<h2>Key numbers</h2><ul class="r-kpis">' + k.map(x => `<li><span>${esc(x.label)}</span> <b>${x.kind === 'int' ? esc(tnum(x.value)) : esc(fmt(x.value))}</b>${isNum(x.delta) ? ` <small>${esc(Math.round(x.delta) > 0 ? '+' : '')}${Math.round(x.delta)}% ${esc(x.vs || '')}</small>` : x.note ? ` <small>${esc(x.note)}</small>` : ''}</li>`).join('') + '</ul>';
    if (ins.length) {
      h += '<h2>Top findings</h2><ol class="r-ins">' + ins.slice(0, 6).map(i => {
        i = i || {}; const e = i.evidence || {}; let line = '';
        try {
          if (i.kind === 'change' && isNum(e.previous_total) && isNum(e.current_total)) {
            const top = arr(e.contributions)[0];
            line = `What changed: ${tnum(e.previous_total)} → ${tnum(e.current_total)} (${tnum(e.change, true)})` + (top && top.segment != null && isNum(top.share_of_change) ? `; biggest mover ${top.segment}, ${pct(top.share_of_change)} of the change` : '') + '.';
          } else if (i.kind === 'anomaly' && isNum(e.observed) && isNum(e.expected)) {
            line = `${e.date ? dateLabel(e.date) + ': ' : ''}observed ${tnum(e.observed)} against ${tnum(e.expected)} expected` + (e.segment && e.segment.value != null ? `; driven by ${e.segment.value}` : '') + '.';
          }
        } catch (err) { line = '' }
        return `<li><b>${esc(i.title)}</b>${KIND[i.kind] ? ` <small>(${esc(KIND[i.kind])})</small>` : ''}<br>${esc(i.detail)}${line ? `<br><i>${esc(line)}</i>` : ''}</li>`;
      }).join('') + '</ol>';
    }
    const recs = LX.recsPlain(nar.recommendations);
    if (recs.length) h += '<h2>What to do next</h2><ol>' + recs.map(x => `<li><b>${esc(x.title)}</b>${x.detail ? `<br>${esc(x.detail)}` : ''}</li>`).join('') + '</ol>';
    if (fc && arr(fc.forecast && fc.forecast.x).length) {
      const f = fc.forecast, n = Math.min(6, f.x.length);
      h += '<h2>Forecast</h2>' + (fc.note ? `<p>${esc(fc.note)}</p>` : '');
      h += `<p class="r-small">${fc.model_name ? 'Method: ' + esc(fc.model_name) + '. ' : ''}${fc.baseline_comparison ? 'In tests on your own past data, ' + esc(lcfirst(fc.baseline_comparison)) + '. ' : ''}${fc.status === 'limited' ? '<b>Limited history: treat as a rough guide.</b> ' : ''}${fc.backtest && isNum(fc.backtest.n_folds) ? 'Tested on ' + fc.backtest.n_folds + ' past period' + (fc.backtest.n_folds === 1 ? '' : 's') + '.' : ''}</p>`;
      if (img) h += `<img class="r-chart" alt="Forecast chart" src="${img}">`;
      const lo = f.lower80 || f.lower, hi = f.upper80 || f.upper;
      h += `<table class="r-tbl"><thead><tr><th>Period</th><th class="r">Forecast</th><th class="r">${f.lower80 ? 'Likely range (80%)' : 'Range'}</th></tr></thead><tbody>`
        + f.x.slice(0, n).map((x, i) => `<tr><td>${esc(dateLabel(x))}</td><td class="r">${tnum(f.y && f.y[i])}</td><td class="r">${isNum(lo && lo[i]) && isNum(hi && hi[i]) ? tnum(lo[i]) + ' to ' + tnum(hi[i]) : '–'}</td></tr>`).join('') + '</tbody></table>';
      const cav = arr(fc.caveats).filter(c => typeof c === 'string' && c.trim()).slice(0, 2);
      if (cav.length) h += '<ul class="r-small">' + cav.map(c => `<li>${esc(c)}</li>`).join('') + '</ul>';
    }
    return h + '<p class="r-foot">Findings are computed in code from the uploaded data. Forecasts are estimates, not promises. Generated by Lumen, free and open source (MIT).</p>';
  }
  async function printReport() {
    const box = q('#report'); if (!box || !LX.state.session) return;
    let img = '';
    try {
      const g = q('#fchart'); if (LX.state.fc && g && g.classList.contains('js-plotly-plot') && !g.hidden) img = await Plotly.toImage(g, { format: 'png', width: 760, height: 260, scale: 2 });
    } catch (e) { img = '' }
    box.innerHTML = reportHtml(img); window.print();
  }
  (function initReport() {
    if (!q('#report')) { const r = document.createElement('div'); r.id = 'report'; r.setAttribute('aria-hidden', 'true'); document.body.appendChild(r) }
    const b = q('#print-btn'); if (b) b.addEventListener('click', printReport);
    window.addEventListener('afterprint', () => { const r = q('#report'); if (r) r.innerHTML = '' });
  })();
  LX.reportHtml = reportHtml;


  /* ---------- data health: what was detected on upload (types, empty cells, unusual values, labels, duplicates) ---------- */
  function healthHtml(p) {
    if (!p || !arr(p.columns).filter(c => !c.synthetic).length) return '';
    const cols = arr(p.columns).filter(c => !c.synthetic), plural = (n, w) => n + ' ' + w + (n === 1 ? '' : 's');
    const pc = v => isNum(v) ? (v > 0 && v < 1 ? '<1%' : Math.round(v) + '%') : '–';
    const gaps = cols.filter(c => (c.missing_pct || 0) > 0), outs = cols.filter(c => (c.outliers || 0) > 0), labs = cols.filter(c => c.label_variant_groups);
    const dups = isNum(p.duplicates) ? p.duplicates : 0;
    const chip = (ok, t) => `<li class="hchip ${ok ? 'ok' : 'warn'}"><span aria-hidden="true">${ok ? '✓' : '!'}</span><span>${ok ? '' : '<b class="sr">Check: </b>'}${esc(t)}</span></li>`;
    const names = l => l.slice(0, 3).map(c => c.name).join(', ') + (l.length > 3 ? ' and more' : '');
    let h = '<h2 id="health-h">Data health</h2><p class="sub">What Lumen found when it read your file, column by column. Nothing is deleted: the Cleaned Output preview shows every change.</p><ul class="hchips">';
    h += chip(true, `${(+p.rows || 0).toLocaleString()} rows, ${cols.length} columns`);
    h += gaps.length ? chip(false, `${plural(gaps.length, 'column')} with empty cells: ${names(gaps)}`) : chip(true, 'No empty cells');
    h += outs.length ? chip(false, `Unusual values in ${names(outs)}`) : chip(true, 'No far-out values');
    if (labs.length) h += chip(false, `Inconsistent spelling in ${names(labs)}`);
    h += dups ? chip(false, `${dups.toLocaleString()} rows are identical to another row (normal if repeat sales have no order number)`) : chip(true, 'No repeated rows');
    h += '</ul><div class="preview-table"><table aria-label="Columns and what was detected in each"><thead><tr><th>Column</th><th>Type</th><th>Empty</th><th>Unusual values</th><th>What is in it</th></tr></thead><tbody>';
    h += cols.map(c => {
      const ex = arr(c.examples).slice(0, 3).map(String);
      let what = '';
      if (c.type === 'Number' || c.kind === 'numeric') what = isNum(c.min) && isNum(c.max) ? `${tnum(c.min)} to ${tnum(c.max)}, typical ${tnum(c.median)}` : '';
      else if (c.kind === 'date') what = arr(c.range).length === 2 ? `${c.range[0]} to ${c.range[1]}` : '';
      else what = ex.length ? `${ex.join(', ')} · ${(+c.unique || 0).toLocaleString()} different` : '';
      if (c.label_variant_groups && arr(c.label_variants)[0]) what += ` · spelled differently: ${arr(c.label_variants)[0].join(' / ')}`;
      const out = (c.outliers || 0) > 0 ? `${(+c.outliers).toLocaleString()} (largest ${arr(c.outlier_examples).map(x => tnum(x)).join(', ')})` : '–';
      return `<tr><td>${esc(c.name)}</td><td>${esc(c.type || cap(c.kind))}</td><td>${(c.missing_pct || 0) > 0 ? esc(pc(c.missing_pct)) : '–'}</td><td>${esc(out)}</td><td>${esc(what)}</td></tr>`;
    }).join('') + '</tbody></table></div><p class="note">Unusual means more than three times the typical spread away from the middle. It is not a verdict that a value is wrong.</p>';
    return h;
  }
  LX.healthHtml = healthHtml;

  /* ---------- recommendations: structured {title, detail, because}; plain strings from older payloads still work ---------- */
  const recObj = r => typeof r === 'string' ? { title: r, detail: '', because: '' } : (r && typeof r === 'object' ? { title: String(r.title || ''), detail: String(r.detail || ''), because: String(r.because || '') } : { title: '', detail: '', because: '' });
  LX.recsHtml = list => arr(list).map(recObj).filter(r => r.title).map(r =>
    `<li><b>${esc(r.title)}</b>${r.detail ? `<p>${esc(r.detail)}</p>` : ''}${r.because ? `<details class="why"><summary>Why we say this</summary><p>${esc(r.because)}</p></details>` : ''}</li>`).join('');
  LX.recsPlain = list => arr(list).map(recObj).filter(r => r.title);

  /* ---------- "what is Lumen analysing?": let the user correct the main measure and date column ---------- */
  LX.pickerHtml = p => {
    const ms = arr(p && p.metric_cols), ds = arr(p && p.date_cols);
    if (ms.length < 2 && ds.length < 2) return '';
    const opt = (v, sel, label) => `<option value="${esc(v)}"${v === sel ? ' selected' : ''}>${esc(label || v)}</option>`;
    return `<div class="pickbar" role="group" aria-label="What Lumen is analysing"><span class="pk-t">Analysing</span>`
      + (ms.length > 1 ? `<label>Main measure <select id="pm">${ms.map(m => opt(m, p.metric, m === 'records' ? 'Number of records' : m)).join('')}</select></label>` : '')
      + (ds.length > 1 ? `<label>over time by <select id="pdate">${ds.map(x => opt(x, p.date)).join('')}</select></label>` : '')
      + `<span class="pk-n">Not what you expected? Change it and everything below updates.</span></div>`;
  };

  LX.reset = () => { LX.fcClear(); LX.state.session = null; const r = q('#report'); if (r) r.innerHTML = '' };
  LX.tnum = tnum; LX.pct = pct;
})();
