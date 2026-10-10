/* lumen-ui.js : the look's behaviour. Three small things, all progressive: the hero lamp (reveals what matters in an unlit sheet),
   key numbers settling on arrival, and a cross-fade between the start screen and the dashboard. Nothing here touches data or the API. */
(function () {
  'use strict';
  const q = s => document.querySelector(s);
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  const LUI = window.LUI = {};

  /* ---------- the sheet ---------- */
  const COLS = ['date', 'product', 'region', 'units', 'amount'];
  const ROWS = [
    ['03 Mar', 'Notebook', 'North', 41, '1,230'], ['03 Mar', 'Hoodie', 'West', 22, '3,080'], ['04 Mar', 'Mug', 'North', 38, '1,140'],
    ['05 Mar', 'Tote', 'East', 17, '510'], ['06 Mar', 'Hoodie', 'North', 25, '3,500'], ['07 Mar', 'Sticker', 'West', 120, '600'],
    ['08 Mar', 'Notebook', 'East', 36, '1,080'], ['09 Mar', 'Mug', 'West', 44, '1,320'], ['10 Mar', 'Hoodie', 'North', 24, '3,360'],
    ['11 Mar', 'Tote', 'West', 19, '570'], ['12 Mar', 'Notebook', 'North', 40, '1,200'], ['13 Mar', 'Mug', 'East', 35, '1,050'],
    ['14 Mar', 'Hoodie', 'North', 187, '26,180'], ['15 Mar', 'Notebook', 'West', 12, '360'], ['16 Mar', 'Mug', 'North', 39, '1,170'],
    ['17 Mar', 'Tote', 'West', 21, '630'], ['18 Mar', 'Hoodie', 'West', 26, '3,640'], ['19 Mar', 'Sticker', 'East', 98, '490'],
  ];
  // [row, col, kind, label]: what the lamp points out
  const HOT = [[12, 4, 'hot', '8x a normal day'], [12, 3, 'hot', ''], [13, 4, 'cold', '-71% vs last Fri'], [13, 3, 'cold', '']];
  const hotAt = (r, c) => HOT.find(h => h[0] === r && h[1] === c);
  function buildGrid(lit) {
    let h = '<thead><tr>' + COLS.map((c, i) => `<th${i >= 3 ? ' class="r"' : ''}>${c}</th>`).join('') + '</tr></thead><tbody>';
    h += ROWS.map((r, ri) => '<tr>' + r.map((v, ci) => {
      const m = lit ? hotAt(ri, ci) : null;
      return `<td class="${ci >= 3 ? 'r' : ''}${m ? ' ' + m[2] : ''}">${v}${m && m[3] ? `<span class="tag">${m[3]}</span>` : ''}</td>`;
    }).join('') + '</tr>').join('') + '</tbody>';
    return h;
  }
  const body = q('#sheet-body');
  if (body) {
    q('#grid-dim').innerHTML = buildGrid(false);
    q('#grid-lit').innerHTML = buildGrid(true);
    const lit = q('#grid-lit');
    const sync = () => { lit.style.width = q('#grid-dim').offsetWidth + 'px' };
    sync(); window.addEventListener('resize', sync);

    let w = 1, h = 1, tx = 0, ty = 0, x = 0, y = 0, last = performance.now(), idleAt = 0, manual = false, raf = 0;
    const place = (px, py) => { body.style.setProperty('--lx', px + 'px'); body.style.setProperty('--ly', py + 'px') };
    const size = () => { const r = body.getBoundingClientRect(); w = r.width; h = r.height };
    size(); window.addEventListener('resize', size);
    // the lamp rests on the spike so the first frame already says something; reduced motion stays there
    const rest = () => { const row = lit.querySelectorAll('tbody tr')[12]; const t = row ? row.getBoundingClientRect() : null, b = body.getBoundingClientRect(); return t ? [w * .6, t.top - b.top + t.height / 2] : [w * .7, h * .6] };
    [x, y] = rest(); tx = x; ty = y; place(x, y);
    if (!reduce.matches) {
      const hero = q('#hero');
      const onMove = e => {
        const b = body.getBoundingClientRect();
        tx = Math.max(-40, Math.min(b.width + 40, e.clientX - b.left)); ty = Math.max(-40, Math.min(b.height + 40, e.clientY - b.top));
        manual = true; idleAt = performance.now() + 4500;
      };
      hero.addEventListener('pointermove', onMove);
      const tick = now => {
        if (document.hidden || q('#hero').style.display === 'none') { raf = requestAnimationFrame(tick); return }
        const dt = Math.min(48, now - last); last = now;
        if (manual && now > idleAt) manual = false;
        if (!manual) { const t = now / 1000; tx = w * (.5 + .36 * Math.sin(t * .31)); ty = h * (.5 + .38 * Math.sin(t * .23 + 1.2)) }
        const k = 1 - Math.exp(-dt / (manual ? 70 : 220));
        x += (tx - x) * k; y += (ty - y) * k; place(x, y);
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }
  }

  /* ---------- key numbers settle ---------- */
  function countUp(el) {
    const raw = el.textContent, m = /^([^\d-]*)(-?\d[\d,]*\.?\d*)(.*)$/.exec(raw.trim());
    if (!m || reduce.matches) return;
    const dec = (m[2].split('.')[1] || '').length, to = parseFloat(m[2].replace(/,/g, '')), comma = m[2].includes(',');
    if (!Number.isFinite(to) || Math.abs(to) < 10) return;
    const fmt = v => comma ? v.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec }) : v.toFixed(dec);
    const t0 = performance.now(), D = 900;
    const step = now => {
      const p = Math.min(1, (now - t0) / D), e = 1 - Math.pow(1 - p, 4);
      el.textContent = m[1] + fmt(to * e) + m[3];
      if (p < 1) requestAnimationFrame(step); else el.textContent = raw;
    };
    requestAnimationFrame(step);
  }

  /* ---------- hero <-> dashboard swap ---------- */
  function afterRender() {
    const app = q('#app'); if (!app) return;
    document.querySelectorAll('aside .ins').forEach((el, i) => el.style.setProperty('--i', Math.min(i, 8)));
    app.classList.remove('is-arriving'); void app.offsetWidth; app.classList.add('is-arriving');
    setTimeout(() => app.classList.remove('is-arriving'), 1800);
    document.querySelectorAll('#kpis .kpi strong').forEach(countUp);
  }
  LUI.swap = fn => {
    const go = () => { fn(); afterRender() };
    if (document.startViewTransition && !reduce.matches) { const t = document.startViewTransition(go); [t.ready, t.finished, t.updateCallbackDone].forEach(x => x && x.catch(() => { })) } else go();
    window.scrollTo(0, 0);
  };
  // re-analyse (measure picker) calls render directly; settle the numbers there too
  const kp = q('#kpis');
  if (kp) new MutationObserver(() => { if (!q('#app').classList.contains('is-arriving')) document.querySelectorAll('#kpis .kpi strong').forEach(countUp) }).observe(kp, { childList: true });
})();
