// Hand crop: the automatic outline as handles to drag, a loupe under the finger, a live preview.
// Points are in the scanner's photo pixels (the photo as sent to Python, ow x oh); the screen shows
// them through one transform (scale, ox, oy), which a pinch zooms and a finger pans (Razi, 22/11). 4 points = one plane, 6 = 4 corners + the 2 fold ends (scan_doc.from_points).
const $ = s => document.querySelector(s);
const RATIO = { card: 85.6 / 53.98, passport: 176 / 125, a4: 297 / 210 };
const PAGE = 88 / 125;           // passport page 125 x 88 mm, as scan_doc.PAGE
const HIT = 44;                  // touch radius around a handle, CSS px
const PAD = 26;                  // room around the photo so a handle on its edge can be grabbed
const LOUPE = 132, ZOOM = 4;     // loupe size (CSS px) and its magnification over the screen
const ZMAX = 8;                  // pinch zoom up to 8x the fitted photo
const TAP2 = 350;                // ms between the two taps of a double tap
const HINT = {
  1: 'گوشه را بکشید (ذره‌بین جای دقیقش را نشان می‌دهد). وسط هر لبه، کل لبه را جابه‌جا می‌کند. دو انگشت: بزرگ‌نمایی.',
  2: 'گوشه‌ها و دو سرِ خطِ تا (فیروزه‌ای) را بکشید؛ هر نیمه جدا صاف می‌شود. دو انگشت: بزرگ‌نمایی.',
  zoom: 'با یک انگشت عکس را جابه‌جا کنید. دو ضربه یا دکمهٔ ذره‌بین: دیدن کل عکس.',
  whole: 'کل عکس به‌کار می‌رود. برای برش، یکی از گوشه‌ها را بکشید.',
  bad: 'گوشه‌ها از روی هم رد شده‌اند؛ شکل را درست کنید.',
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const len = a => Math.hypot(a[0], a[1]);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const r1 = v => Math.round(v * 10) / 10;

// scan_doc.order: by angle around the centre, starting at the corner nearest the top-left
function orderIdx(P) {
  const idx = [0, 1, 2, 3];
  const c = [0, 1].map(k => idx.reduce((s, i) => s + P[i][k], 0) / 4);
  const ang = i => Math.atan2(P[i][1] - c[1], P[i][0] - c[0]);
  const by = idx.slice().sort((i, j) => ang(i) - ang(j));
  let k0 = 0;
  by.forEach((i, k) => { if (P[i][0] + P[i][1] < P[by[k0]][0] + P[by[k0]][1]) k0 = k; });
  return by.slice(k0).concat(by.slice(0, k0));
}

// scan_doc.fold_ends + panel_quads, as indices into P: corners, fold ends, outline, panels
export function geom(P) {
  const [tl, tr, br, bl] = orderIdx(P);
  if (P.length < 6) return { c: [tl, tr, br, bl], poly: [tl, tr, br, bl], panels: [[tl, tr, br, bl]] };
  const wd = add(sub(P[tr], P[tl]), sub(P[br], P[bl])), ht = add(sub(P[bl], P[tl]), sub(P[br], P[tr]));
  let a = 4, b = 5;
  const u = sub(P[b], P[a]);
  const across = Math.abs(dot(u, wd)) / len(wd) >= Math.abs(dot(u, ht)) / len(ht);
  if (dot(u, across ? wd : ht) < 0) [a, b] = [b, a];
  return across
    ? { c: [tl, tr, br, bl], across, fa: a, fb: b, poly: [tl, tr, b, br, bl, a], panels: [[tl, tr, b, a], [a, b, br, bl]] }
    : { c: [tl, tr, br, bl], across, fa: a, fb: b, poly: [tl, a, tr, br, b, bl], panels: [[tl, a, b, bl], [a, tr, br, b]] };
}

function convex(Q, minArea) {
  let s = 0, area = 0;
  for (let k = 0; k < 4; k++) {
    const a = Q[k], b = Q[(k + 1) % 4], c = Q[(k + 2) % 4];
    const z = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (!z || (s && Math.sign(z) !== s)) return false;
    s = Math.sign(z);
    area += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(area) / 2 >= minArea;
}

// the page's size as scan_doc.flatten / warp_panels give it ('doc' approximates the perspective
// estimate by the side lengths): output rects and the photo quads that fill them
function layout(P, g, kind, mm) {
  const Q = g.panels.map(pk => pk.map(i => P[i]));
  const fixed = mm ? Math.max(...mm) / Math.min(...mm) : RATIO[kind];
  if (Q.length === 1) {
    const [tl, tr, br, bl] = Q[0];
    const wd = Math.max(len(sub(tr, tl)), len(sub(br, bl))), ht = Math.max(len(sub(bl, tl)), len(sub(br, tr)));
    const r = fixed || Math.max(wd, ht) / Math.max(1, Math.min(wd, ht)), L = Math.max(wd, ht);
    const [W, H] = wd >= ht ? [L, L / r] : [L / r, L];
    return { W, H, rects: [[0, 0, W, H]], Q };
  }
  const A = g.across;
  const along = A ? Math.max(len(sub(Q[0][1], Q[0][0])), len(sub(Q[0][2], Q[0][3])), len(sub(Q[1][2], Q[1][3])))
    : Math.max(len(sub(Q[0][3], Q[0][0])), len(sub(Q[0][2], Q[0][1])), len(sub(Q[1][2], Q[1][1])));
  const est = Q.map(([tl, tr, br, bl]) => A ? (len(sub(bl, tl)) + len(sub(br, tr))) / 2 : (len(sub(tr, tl)) + len(sub(br, bl))) / 2);
  let d = est;
  if (kind === 'passport') d = [along * PAGE, along * PAGE];
  else if (fixed) d = est.map(e => along * fixed * e / (est[0] + est[1]));
  const rects = A ? [[0, 0, along, d[0]], [0, d[0], along, d[0] + d[1]]] : [[0, 0, d[0], along], [d[0], 0, d[0] + d[1], along]];
  return { W: rects[1][2], H: rects[1][3], rects, Q };
}

// 3x3 homography (row-major, last = 1) taking src[k] to dst[k]
function homography(src, dst) {
  const A = [], b = [];
  for (let k = 0; k < 4; k++) {
    const [x, y] = src[k], [u, v] = dst[k];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  for (let i = 0; i < 8; i++) {
    let p = i;
    for (let r = i + 1; r < 8; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    [A[i], A[p]] = [A[p], A[i]];
    [b[i], b[p]] = [b[p], b[i]];
    if (Math.abs(A[i][i]) < 1e-12) return null;
    for (let r = 0; r < 8; r++) {
      const f = r === i ? 0 : A[r][i] / A[i][i];
      if (!f) continue;
      for (let c = i; c < 8; c++) A[r][c] -= f * A[i][c];
      b[r] -= f * b[i];
    }
  }
  return [...b.map((v, i) => v / A[i][i]), 1];
}

const E = { open: false, ptrs: new Map() };     // ptrs: pointer id -> stage point of each finger down
let wired = false;

// photo pixels <-> stage CSS px: the only mapping (handles, hits, drags, loupe, photo)
const S = p => [E.ox + p[0] * E.scale, E.oy + p[1] * E.scale];
const toPhoto = (x, y) => [(x - E.ox) / E.scale, (y - E.oy) / E.scale];
const zoomed = () => E.scale > E.fit.scale * 1.001;
const corners = () => [[0, 0], [E.ow - 1, 0], [E.ow - 1, E.oh - 1], [0, E.oh - 1]];

function handles(g) {
  const out = g.c.map(i => ({ idx: [i], at: E.P[i] }));
  if (E.P.length === 6) out.push({ idx: [g.fa], at: E.P[g.fa], fold: true }, { idx: [g.fb], at: E.P[g.fb], fold: true });
  const n = g.poly.length;
  for (let k = 0; k < n; k++) {
    const i = g.poly[k], j = g.poly[(k + 1) % n];
    out.push({ idx: [i, j], at: mid(E.P[i], E.P[j]), mid: true });
  }
  if (E.P.length === 6) out.push({ idx: [g.fa, g.fb], at: mid(E.P[g.fa], E.P[g.fb]), mid: true, fold: true });
  return out;
}

function pick(x, y) {
  let best = null, bd = HIT;
  for (const h of handles(geom(E.P))) {
    const [sx, sy] = S(h.at);
    const d = Math.hypot(sx - x, sy - y) + (h.mid ? 12 : 0);       // a corner wins over a nearby midpoint
    if (d < bd) { bd = d; best = h; }
  }
  return best;
}

function valid(g) {
  return g.panels.every(pk => convex(pk.map(i => E.P[i]), 0.002 * E.ow * E.oh));
}

function foldAtMiddle() {
  const P = E.P.slice(0, 4), [tl, tr, br, bl] = orderIdx(P).map(i => P[i]);
  const tall = len(sub(bl, tl)) + len(sub(br, tr)) >= len(sub(tr, tl)) + len(sub(br, bl));
  E.P = P.concat(tall ? [mid(tl, bl), mid(tr, br)] : [mid(tl, tr), mid(bl, br)]);
}

function setPlanes(n) {
  if (n === 2 && E.P.length === 4) { E.whole = false; foldAtMiddle(); }
  if (n === 1) E.P = E.P.slice(0, 4);
  frame();
}

function layoutStage() {
  const st = $('#ed-stage').getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  E.W = st.width; E.H = st.height;
  const s = Math.min((E.W - 2 * PAD) / E.ow, (E.H - 2 * PAD) / E.oh);
  E.fit = { scale: s, ox: (E.W - E.ow * s) / 2, oy: (E.H - E.oh * s) / 2 };
  fitView();
  const cv = $('#ed-photo');
  cv.width = Math.round(E.W * dpr); cv.height = Math.round(E.H * dpr);
  cv.style.width = E.W + 'px'; cv.style.height = E.H + 'px';
  const svg = $('#ed-svg');
  svg.setAttribute('viewBox', `0 0 ${E.W} ${E.H}`);
  svg.setAttribute('width', E.W); svg.setAttribute('height', E.H);
  const L = $('#ed-loupe');
  L.width = L.height = Math.round(LOUPE * dpr);
}

function viewChanged() {
  E.vdirty = true;
  $('#ed-fit').hidden = !zoomed();
}

function fitView() {
  Object.assign(E, { scale: E.fit.scale, ox: E.fit.ox, oy: E.fit.oy });
  viewChanged();
}

// a zoomed photo always covers the stage (PAD spare at its edges); one that fits stays centred
function clampView() {
  for (const [o, n, W] of [['ox', E.ow, E.W], ['oy', E.oh, E.H]]) {
    const size = n * E.scale;
    E[o] = size + 2 * PAD <= W + 1e-6 ? (W - size) / 2 : clamp(E[o], W - PAD - size, PAD);
  }
  viewChanged();
}

// scale s (kept within fit..ZMAX x fit) with photo point p under stage point (x, y)
function zoomTo(s, p, x, y) {
  E.scale = clamp(s, E.fit.scale, E.fit.scale * ZMAX);
  E.ox = x - p[0] * E.scale;
  E.oy = y - p[1] * E.scale;
  clampView();
}

// only the part of the photo on the stage is drawn (a 3000 px photo at 8x runs far off the canvas)
function drawPhoto() {
  const cv = $('#ed-photo'), g = cv.getContext('2d'), dpr = cv.width / E.W;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, E.W, E.H);
  const kx = E.img.naturalWidth / E.ow, ky = E.img.naturalHeight / E.oh;
  const [x0, y0] = toPhoto(0, 0).map(v => Math.max(0, v)), [x1, y1] = toPhoto(E.W, E.H);
  const x2 = Math.min(E.ow, x1), y2 = Math.min(E.oh, y1);
  if (x2 <= x0 || y2 <= y0) return;
  g.imageSmoothingQuality = 'high';
  g.drawImage(E.img, x0 * kx, y0 * ky, (x2 - x0) * kx, (y2 - y0) * ky,
    E.ox + x0 * E.scale, E.oy + y0 * E.scale, (x2 - x0) * E.scale, (y2 - y0) * E.scale);
}

function drawOverlay(g, ok) {
  const pt = q => S(q).map(v => v.toFixed(1)).join(' ');
  const poly = g.poly.map(i => pt(E.P[i]));
  let h = `<path class="shade" d="M0 0H${E.W}V${E.H}H0Z M${poly.join('L')}Z"/>`;
  h += `<path class="edge${ok ? '' : ' bad'}" d="M${poly.join('L')}Z"/>`;
  if (E.P.length === 6) h += `<path class="fold" d="M${pt(E.P[g.fa])}L${pt(E.P[g.fb])}"/>`;
  const on = E.drag && E.drag.h;
  for (const x of handles(g)) {
    const [cx, cy] = S(x.at).map(v => v.toFixed(1));
    if (on && x.idx.join() === on.idx.join()) h += `<circle class="on" cx="${cx}" cy="${cy}" r="24"/>`;
    h += `<circle class="${x.mid ? 'm' : 'h'}${x.fold ? ' f' : ''}" cx="${cx}" cy="${cy}" r="${x.mid ? 7 : 11}"/>`;
  }
  $('#ed-svg').innerHTML = h;
}

function drawLoupe(g) {
  const L = $('#ed-loupe'), d = E.drag;
  if (!d) { L.hidden = true; return; }
  const at = d.h.idx.length === 1 ? E.P[d.h.idx[0]] : mid(E.P[d.h.idx[0]], E.P[d.h.idx[1]]);
  const [sx, sy] = S(at);
  const near = sx < LOUPE + 40 && sy < LOUPE + 40;               // never under the finger
  L.style.left = (near ? E.W - LOUPE - 16 : 10) + 'px';
  L.style.top = '10px';
  L.hidden = false;
  const dpr = L.width / LOUPE, z = ZOOM * E.scale, half = LOUPE / 2 / z;
  const c = L.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.fillStyle = '#000';
  c.fillRect(0, 0, LOUPE, LOUPE);
  const kx = E.img.naturalWidth / E.ow, ky = E.img.naturalHeight / E.oh;
  // source rect clipped to the photo (Safari draws nothing for a rect running off the image)
  const x0 = Math.max(0, at[0] - half), y0 = Math.max(0, at[1] - half);
  const x1 = Math.min(E.ow, at[0] + half), y1 = Math.min(E.oh, at[1] + half);
  if (x1 > x0 && y1 > y0) {
    c.imageSmoothingQuality = 'high';
    c.drawImage(E.img, x0 * kx, y0 * ky, (x1 - x0) * kx, (y1 - y0) * ky,
      LOUPE / 2 + (x0 - at[0]) * z, LOUPE / 2 + (y0 - at[1]) * z, (x1 - x0) * z, (y1 - y0) * z);
  }
  const T = p => [LOUPE / 2 + (p[0] - at[0]) * z, LOUPE / 2 + (p[1] - at[1]) * z];
  c.lineWidth = 1.5;
  c.strokeStyle = '#f59e0b';
  c.beginPath();
  g.poly.forEach((i, k) => { const [x, y] = T(E.P[i]); k ? c.lineTo(x, y) : c.moveTo(x, y); });
  c.closePath();
  c.stroke();
  if (E.P.length === 6) {
    c.strokeStyle = '#2dd4bf';
    c.beginPath(); c.moveTo(...T(E.P[g.fa])); c.lineTo(...T(E.P[g.fb])); c.stroke();
  }
  c.strokeStyle = 'rgba(255,255,255,.9)';                          // crosshair with a gap at the point
  c.lineWidth = 1;
  c.beginPath();
  for (const [a, b] of [[[0, 66], [58, 66]], [[74, 66], [132, 66]], [[66, 0], [66, 58]], [[66, 74], [66, 132]]]) { c.moveTo(...a); c.lineTo(...b); }
  c.stroke();
}

function drawPreview(g, ok) {
  const cv = $('#ed-prev'), box = cv.parentElement.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1), sm = E.small;
  const turned = E.rot % 180 !== 0;
  let W = E.ow, H = E.oh, lay = null;
  if (!E.whole && ok) { lay = layout(E.P, g, E.kind, E.mm); W = lay.W; H = lay.H; }
  const s = Math.min((box.width - 12) / (turned ? H : W), (box.height - 12) / (turned ? W : H));
  const rw = Math.max(1, Math.round(W * s * dpr)), rh = Math.max(1, Math.round(H * s * dpr));
  const tmp = E.tmp || (E.tmp = document.createElement('canvas'));
  tmp.width = rw; tmp.height = rh;
  const tg = tmp.getContext('2d');
  if (!lay) {
    tg.drawImage(sm.c, 0, 0, rw, rh);
    if (!E.whole) { tg.fillStyle = 'rgba(0,0,0,.6)'; tg.fillRect(0, 0, rw, rh); }
  } else {
    const out = tg.createImageData(rw, rh), o = out.data, src = sm.data, sw = sm.w, sh = sm.h;
    const kx = rw / W, ky = rh / H;
    lay.rects.forEach(([x0, y0, x1, y1], n) => {
      const R = [[x0 * kx, y0 * ky], [x1 * kx, y0 * ky], [x1 * kx, y1 * ky], [x0 * kx, y1 * ky]];
      const h = homography(R, lay.Q[n].map(p => [p[0] * sm.k, p[1] * sm.k]));
      if (!h) return;
      for (let y = Math.round(y0 * ky); y < Math.min(rh, Math.round(y1 * ky)); y++) {
        for (let x = 0; x < rw; x++) {
          if (x < Math.round(x0 * kx) || x >= Math.round(x1 * kx)) continue;
          const w = h[6] * (x + .5) + h[7] * (y + .5) + 1;
          const u = clamp((h[0] * (x + .5) + h[1] * (y + .5) + h[2]) / w - .5, 0, sw - 1.001);
          const v = clamp((h[3] * (x + .5) + h[4] * (y + .5) + h[5]) / w - .5, 0, sh - 1.001);
          const ix = u | 0, iy = v | 0, fx = u - ix, fy = v - iy, i = (iy * sw + ix) * 4, j = (y * rw + x) * 4;
          for (let c = 0; c < 3; c++) {
            const top = src[i + c] + (src[i + 4 + c] - src[i + c]) * fx;
            const bot = src[i + sw * 4 + c] + (src[i + sw * 4 + 4 + c] - src[i + sw * 4 + c]) * fx;
            o[j + c] = top + (bot - top) * fy;
          }
          o[j + 3] = 255;
        }
      }
    });
    tg.putImageData(out, 0, 0);
  }
  cv.width = turned ? rh : rw; cv.height = turned ? rw : rh;
  cv.style.width = cv.width / dpr + 'px'; cv.style.height = cv.height / dpr + 'px';
  const c = cv.getContext('2d');
  c.translate(cv.width / 2, cv.height / 2);
  c.rotate(E.rot * Math.PI / 180);
  c.drawImage(tmp, -rw / 2, -rh / 2);
}

function frame() {
  if (E.raf) return;
  E.raf = requestAnimationFrame(() => {
    E.raf = 0;
    if (!E.open || !E.img) return;
    const g = geom(E.P), ok = E.whole || valid(g);
    const two = E.P.length === 6;
    for (const b of $('#ed-planes').children) b.setAttribute('aria-pressed', String(+b.dataset.v === (two ? 2 : 1)));
    $('#ed-mid').hidden = !two;
    $('#ed-rot').style.gridColumn = two ? '' : '1 / -1';
    const hint = $('#ed-hint');
    hint.textContent = !ok ? HINT.bad : E.whole ? HINT.whole : zoomed() ? HINT.zoom : HINT[two ? 2 : 1];
    hint.className = 'ed-hint' + (ok ? '' : ' bad');
    $('#ed-ok').disabled = !ok;
    if (E.vdirty) { E.vdirty = false; drawPhoto(); }
    drawOverlay(g, ok);
    drawLoupe(g);
    drawPreview(g, ok);
  });
}

// fingers on the stage: one on a handle drags it; two pinch (zoom around their midpoint, pan with
// it); one elsewhere pans a zoomed photo; a double tap goes back to the whole photo (or zooms in)
const at = e => { const r = $('#ed-stage').getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

function startPinch() {
  const d = E.drag;
  if (d) {                    // the first finger only began the pinch: its handle goes back
    d.h.idx.forEach((i, k) => { E.P[i] = d.from[k].slice(); });
    E.whole = d.whole;
    E.drag = null;
  }
  E.pan = E.tap = null;
  const [a, b] = [...E.ptrs.values()];
  E.pinch = { d0: Math.max(1, len(sub(a, b))), s0: E.scale, p: toPhoto(...mid(a, b)) };
  frame();
}

function down(e) {
  if (!E.img || e.target.closest('#ed-fit')) return;
  e.preventDefault();
  const xy = at(e);
  E.ptrs.set(e.pointerId, xy);
  try { $('#ed-stage').setPointerCapture(e.pointerId); } catch (_) { /* the stage still gets the moves */ }
  if (E.ptrs.size === 2) return startPinch();
  if (E.ptrs.size > 2) return;
  const h = pick(...xy);
  if (h) {
    E.drag = { id: e.pointerId, h, x: e.clientX, y: e.clientY, from: h.idx.map(i => E.P[i].slice()), whole: E.whole };
    E.whole = false;
    return frame();
  }
  E.tap = { id: e.pointerId, x: xy[0], y: xy[1], t: performance.now() };
  if (zoomed()) E.pan = { id: e.pointerId, x: xy[0], y: xy[1], ox: E.ox, oy: E.oy };
}

function move(e) {
  if (!E.ptrs.has(e.pointerId)) return;
  e.preventDefault();
  const xy = at(e);
  E.ptrs.set(e.pointerId, xy);
  if (E.pinch) {
    const [a, b] = [...E.ptrs.values()];
    zoomTo(E.pinch.s0 * len(sub(a, b)) / E.pinch.d0, E.pinch.p, ...mid(a, b));
    return frame();
  }
  if (E.tap && e.pointerId === E.tap.id && Math.hypot(xy[0] - E.tap.x, xy[1] - E.tap.y) > 10) E.tap = null;
  const pn = E.pan;
  if (pn && e.pointerId === pn.id) {
    E.ox = pn.ox + xy[0] - pn.x;
    E.oy = pn.oy + xy[1] - pn.y;
    clampView();
    return frame();
  }
  const d = E.drag;
  if (!d || e.pointerId !== d.id) return;
  let dx = (e.clientX - d.x) / E.scale, dy = (e.clientY - d.y) / E.scale;
  for (const p of d.from) {                                      // a whole edge moves rigidly, on the photo
    dx = clamp(dx, -p[0], E.ow - 1 - p[0]);
    dy = clamp(dy, -p[1], E.oh - 1 - p[1]);
  }
  d.h.idx.forEach((i, k) => { E.P[i] = [d.from[k][0] + dx, d.from[k][1] + dy]; });
  frame();
}

function up(e) {
  if (!E.ptrs.delete(e.pointerId)) return;
  if (E.drag && e.pointerId === E.drag.id) E.drag = null;
  if (E.pan && e.pointerId === E.pan.id) E.pan = null;
  if (E.pinch && E.ptrs.size < 2) {          // the finger left on the glass pans on from here, no jump
    E.pinch = null;
    const [id, xy] = [...E.ptrs][0] || [];
    if (id != null && zoomed()) E.pan = { id, x: xy[0], y: xy[1], ox: E.ox, oy: E.oy };
  }
  const t = E.tap;
  if (t && e.pointerId === t.id) {
    E.tap = null;
    const last = E.lastTap;
    E.lastTap = null;
    if (e.type !== 'pointerup') return frame();
    if (last && t.t - last.t < TAP2 && Math.hypot(t.x - last.x, t.y - last.y) < 30) {
      if (zoomed()) fitView();
      else zoomTo(E.fit.scale * 3, toPhoto(t.x, t.y), t.x, t.y);
    } else E.lastTap = t;
  }
  frame();
}

function close() {
  E.open = false;
  Object.assign(E, { drag: null, pinch: null, pan: null, tap: null, lastTap: null });
  E.ptrs.clear();
  $('#ed').hidden = true;
  document.documentElement.classList.remove('lock');
  if (E.url) URL.revokeObjectURL(E.url);
  Object.assign(E, { url: null, img: null, small: null, tmp: null });
}

function wire() {
  wired = true;
  const st = $('#ed-stage');
  st.addEventListener('pointerdown', down);
  st.addEventListener('pointermove', move);
  st.addEventListener('pointerup', up);
  st.addEventListener('pointercancel', up);
  $('#ed').addEventListener('touchmove', e => e.preventDefault(), { passive: false });   // iOS: no page scroll
  for (const g of ['gesturestart', 'gesturechange']) $('#ed').addEventListener(g, e => e.preventDefault());   // iOS: the page never zooms
  $('#ed-fit').addEventListener('click', () => { fitView(); frame(); });
  $('#ed-planes').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setPlanes(+b.dataset.v); });
  $('#ed-mid').addEventListener('click', () => { foldAtMiddle(); frame(); });
  $('#ed-auto').addEventListener('click', () => {
    E.whole = !E.auto;
    E.P = E.auto ? E.auto.map(p => p.slice()) : corners();
    E.rot = E.autoRot;
    frame();
  });
  $('#ed-whole').addEventListener('click', () => { E.whole = true; E.P = corners(); frame(); });
  $('#ed-rot').addEventListener('click', () => { E.rot = (E.rot + 90) % 360; frame(); });
  $('#ed-cancel').addEventListener('click', close);
  $('#ed-ok').addEventListener('click', () => {
    const pts = E.whole ? null : E.P.map(p => [r1(p[0]), r1(p[1])]);
    const same = JSON.stringify([pts, E.rot]) === E.start;
    const done = E.done;
    close();
    if (!same) done(pts, E.rot);
  });
  window.addEventListener('resize', () => { if (E.open && E.img) { layoutStage(); frame(); } });   // back to the whole photo
}

// page: {ow, oh, src (the photo File), info: {pts, auto, auto_rot, rot, mm} or none for a refused photo}
// kind: the document kind; done(pts or null, rot) runs the scanner on the new crop
export async function openEditor(page, kind, done) {
  if (!wired) wire();
  const i = page.info || {};
  E.ptrs.clear();
  Object.assign(E, { open: true, ow: page.ow, oh: page.oh, kind, mm: i.mm || null, done, drag: null, pinch: null, pan: null, tap: null, lastTap: null,
    auto: i.auto || null, autoRot: i.auto_rot || 0, rot: i.rot || 0 });
  E.whole = !!page.info && !i.pts;
  E.P = i.pts ? i.pts.map(p => p.slice())
    : E.whole ? corners()
      : [[.1, .1], [.9, .1], [.9, .9], [.1, .9]].map(([x, y]) => [x * (E.ow - 1), y * (E.oh - 1)]);
  // what the page already shows: applying it again is a no-op (a refused photo shows nothing yet)
  E.start = page.info ? JSON.stringify([E.whole ? null : E.P.map(p => [r1(p[0]), r1(p[1])]), E.rot]) : null;
  $('#ed-planes').hidden = kind === 'card';
  $('#ed-hint').textContent = 'بارگذاری عکس…';
  $('#ed-svg').innerHTML = '';
  $('#ed-fit').hidden = true;
  $('#ed').hidden = false;
  document.documentElement.classList.add('lock');
  E.url = URL.createObjectURL(page.src);
  const img = new Image();
  await new Promise((ok, bad) => { img.onload = ok; img.onerror = () => bad(new Error('عکس باز نشد')); img.src = E.url; });
  if (!E.open) return;                                           // closed while loading
  const k = Math.min(1, 900 / Math.max(E.ow, E.oh));
  const c = document.createElement('canvas');
  c.width = Math.round(E.ow * k); c.height = Math.round(E.oh * k);
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0, c.width, c.height);
  E.small = { c, data: g.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height, k: c.width / E.ow };
  E.img = img;
  layoutStage();
  frame();
}

// for the self-test and the browser tools: the editor's state and its geometry
export const editorState = () => ({ open: E.open, P: E.P, whole: E.whole, rot: E.rot, scale: E.scale, ox: E.ox, oy: E.oy,
  fit: E.fit && { ...E.fit }, W: E.W, H: E.H, dragging: !!E.drag, pinching: !!E.pinch,
  loupe: !$('#ed-loupe').hidden, fitBtn: !$('#ed-fit').hidden, ok: !$('#ed-ok').disabled });
