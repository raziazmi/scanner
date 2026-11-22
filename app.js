// Page side: pick photos, show each result, build and share the PDF. The scanning itself is worker.js,
// the hand crop crop.js, the pages kept on the phone store.js.
import { openEditor, editorState, geom } from './crop.js';
import { ico, paintIcons } from './icons.js';
import { store } from './store.js';

const $ = s => document.querySelector(s);
const VER = '2e3036d825';        // build.py writes the build's version here and in sw.js
const MAX = 3000;                // long side sent to the scanner (iOS canvas limit; 256 dpi on A4)
const CAPS = { 1: 1e6, 2: 2e6, 5: 5e6, 0: null };    // MB -> bytes (10^6: fits a portal counting either way)
// kind: none until chosen (Razi, 22/11: no default, the photo step waits for it);
// sizeby: how the real size of a «سند دیگر» is known (none / typed mm / a bank card beside it); no
// default either: asked again at every start, new document and kind change (Razi, 22/11)
const S = { kind: null, sizeby: null, mode: 'color', out: 'print', cap: '2', paper: 'a3', pages: [], pdf: null, ready: false };
const SIZE_HELP = {
  ask: 'یکی از سه جواب را بزنید؛ بعد قدم عکس می‌آید.',
  none: 'کپی کل برگهٔ A4 را پر می‌کند. اگر سند از A4 بزرگ‌تر است، اندازه‌اش را بدهید تا کوچک چاپ نشود.',
  mm: 'لبه‌های سند را با خط‌کش اندازه بگیرید؛ کپی ۱۱۷٪ همین اندازه چاپ می‌شود.',
  ref: 'یک کارت بانکی یا کارت شناسایی را کنار سند بگذارید (نه رویش) تا هر چهار لبه‌اش در عکس باشد؛ اندازه از آن حساب می‌شود.',
};
const ERR = {
  nodoc: 'در این عکس سندی پیدا نشد. روی زمینهٔ تیره و ساده دوباره عکس بگیرید.',
  nocard: 'کارت مرجع پیدا نشد: کارت را کنار سند بگذارید (نه رویش) تا هر چهار لبه‌اش دیده شود.',
  crash: 'اسکن این عکس برنامه را بست (احتمالاً حافظهٔ گوشی کم آمد) و دوباره خودکار امتحان نمی‌شود. «دوباره بگیر» یا «حذف» را بزنید.',
};
// quality warnings from scan_doc.quality: chip text, then what to do
const QA = {
  blur: ['تار است', 'گوشی را ثابت نگه دارید و روی سند بزنید تا فوکوس کند'],
  glare: ['برق نور روی کارت', 'کارت را کمی کج کنید یا از زیر چراغ کنار بکشید'],
  dark: ['تاریک است', 'جای روشن‌تری عکس بگیرید'],
  lowres: ['وضوح کم', 'نزدیک‌تر بگیرید؛ از تلگرام عکس را «به‌صورت فایل» بفرستید'],
  cut: ['لبهٔ سند بیرون از عکس', 'کمی عقب‌تر بروید تا هر چهار لبه دیده شود'],
  edge: ['لبهٔ برگه واضح نیست', 'شاید چیز دیگری از روی میز بریده شده؛ کادر را نگاه کنید و اگر درست نیست «برش دستی» بزنید یا برگه را روی میز خلوت بگذارید'],
  shape: ['کادر شکل سند نیست', 'احتمالاً چیز دیگری بریده شده؛ «برش دستی» بزنید یا سند را روی میز ساده‌ای بگذارید'],
  whole: ['لبهٔ سند پیدا نشد', 'کل عکس به‌کار رفت؛ اگر عکس از قبل بریده نشده، «برش دستی» بزنید یا روی زمینهٔ ساده و تیره‌تر دوباره بگیرید'],
};
// a page's error in a word, for its tile (the full text is on the page itself)
const ERR_SHORT = { nodoc: 'سند پیدا نشد', nocard: 'کارت مرجع پیدا نشد', crash: 'برنامه بسته شد' };
// what the file will be, for the kind chosen (fotocopia_a4.build / .upload)
const PRINT_HELP = {
  card: 'هر کارت، رو و پشت، روی یک برگهٔ A4؛ ۱۱۷٪ اندازهٔ واقعی.',
  passport: 'هر عکس پاسپورت (دو صفحهٔ باز) روی یک برگهٔ A4؛ ۱۱۷٪ اندازهٔ واقعی.',
  a4: 'هر برگه روی یک A4، به اندازهٔ واقعی (۱۰۰٪).',
  doc: 'هر سند روی یک A4: ۱۱۷٪ اندازهٔ واقعی اگر معلوم باشد، وگرنه کل صفحه.',
};
const UPLOAD_HELP = 'برای پورتال و ایمیل: هر صفحه به اندازهٔ خود سند و حجم فایل زیر سقف.';
const DEBUG = new URLSearchParams(location.search).has('debug');     // the scanner's log on each page
const waiting = new Map();
let worker, seq = 0, pageSeq = 0, queue = Promise.resolve(), D = null, retakePid = null, PG = null;

const fa = n => Number(n).toLocaleString('fa-IR');
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const flagsOf = p => (p.status === 'ok' && p.info && p.info.quality && p.info.quality.flags) || [];

function setStatus(text, busy, err) {
  const s = $('#status');
  s.textContent = text;
  s.className = (busy ? 'busy' : '') + (err ? ' err' : '');
}

function call(cmd, args = {}, transfer = []) {
  return new Promise((res, rej) => {
    const id = ++seq;
    waiting.set(id, { res, rej });
    worker.postMessage({ id, cmd, ...args }, transfer);
  });
}

function startWorker() {
  worker = new Worker('worker.js', { type: 'module' });
  worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'status') { if (m.ready) S.ready = true; return setStatus(m.text, m.busy, m.error); }
    if (m.type === 'log') return console.log(m.text);
    if (m.heap) S.heap = m.heap;
    const w = waiting.get(m.id);
    if (!w) return;
    waiting.delete(m.id);
    m.type === 'error' ? w.rej(new Error(m.text)) : w.res(m);
  };
  worker.onerror = e => setStatus('اجرا نشد: ' + (e.message || 'خطای ناشناخته'), false, true);
}

// the photo as RGBA for the scanner, capped at MAX px; with thumb, also a small JPEG for the list
async function decode(file, thumb = true) {
  const src = URL.createObjectURL(file);
  try {
    const img = new Image();
    await new Promise((ok, bad) => {          // onload, not decode(): decode() waits for rendering (paused pane)
      img.onload = ok;
      img.onerror = () => bad(new Error('این فایل عکس خوانده نشد'));
      img.src = src;
    });
    const s = Math.min(1, MAX / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.round(img.naturalWidth * s), h = Math.round(img.naturalHeight * s);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0, w, h);
    const buf = g.getImageData(0, 0, w, h).data.buffer;
    let tb = null;
    if (thumb) {
      const t = Math.min(1, 480 / Math.max(w, h));
      const tc = document.createElement('canvas');
      tc.width = Math.round(w * t); tc.height = Math.round(h * t);
      tc.getContext('2d').drawImage(c, 0, 0, tc.width, tc.height);
      tb = await new Promise(r => tc.toBlob(r, 'image/jpeg', 0.8));
      tc.width = tc.height = 0;
    }
    c.width = c.height = 0;                    // iOS keeps canvas memory otherwise
    return { buf, w, h, tb };
  } finally {
    URL.revokeObjectURL(src);
  }
}

function opts() {
  const w = parseFloat($('#mmw').value), h = parseFloat($('#mmh').value), doc = S.kind === 'doc';
  const mm = doc && S.sizeby === 'mm' && w > 0 && h > 0 ? [w, h] : null;
  return { kind: S.kind, mode: S.mode, mm, ref: doc && S.sizeby === 'ref' };
}

function apply(p, m) {
  const i = m.info;
  if (i.error) {
    p.status = 'error';
    p.error = ERR[i.error] || i.error;
    p.errk = i.error;
    p.cropable = i.error in ERR;          // the photo stays in the scanner: it can be cropped by hand
    p.info = null;
  } else {
    p.cropable = true;
    if (p.url) URL.revokeObjectURL(p.url);
    p.res = new Blob([m.data], { type: i.path.endsWith('.png') ? 'image/png' : 'image/jpeg' });
    p.url = URL.createObjectURL(p.res);
    p.status = 'ok';
    p.error = null;
    p.info = i;
  }
  p.log = i.log && i.log.length ? i.log.join('\n') : p.log;
}

// ---- kept on the phone (store.js): photo once, the rest after every change that settled

function savePhoto(p) {
  return store.photo({ pid: p.pid, photo: p.src });
}

function save(p) {
  if (!S.pages.includes(p)) return Promise.resolve();
  let info = null;
  if (p.info) { info = { ...p.info }; delete info.log; delete info.path; }
  return store.page({ pid: p.pid, status: p.status === 'busy' ? (p.res ? 'ok' : 'busy') : p.status, error: p.error || null, errk: p.errk || null, tries: p.tries || 0,
    cropable: !!p.cropable, info, result: p.res || null, opts: p.opts, mode: p.mode || S.mode,
    thumb: p.tb || null, ow: p.ow || 0, oh: p.oh || 0 });
}

function saveMeta() {
  store.setMeta({ kind: S.kind, mode: S.mode, out: S.out, cap: S.cap, paper: S.paper, name: $('#name').value, copies: $('#copies').value,
    mmw: $('#mmw').value, mmh: $('#mmh').value, sizeby: S.sizeby, order: S.pages.map(p => p.pid) });
}

function forget(p) {
  for (const u of [p.url, p.thumb]) if (u) URL.revokeObjectURL(u);
  if (p.live) call('drop', { pid: p.pid }).catch(() => {});
  store.drop(p.pid);
}

// a kept page goes back into the scanner (Python lost it when iOS closed the app) before it is used
function ensure(p) {
  if (p.live) return Promise.resolve();
  return p.hyd || (p.hyd = (async () => {
    const d = await decode(p.src, false);
    const o = { ...p.opts, mode: p.mode || S.mode };
    if (p.info) Object.assign(o, { pts: p.info.pts, rot: p.info.rot });     // info: only a page with a result
    apply(p, await call('restore', { pid: p.pid, buf: d.buf, w: d.w, h: d.h, opts: JSON.stringify(o) }, [d.buf]));
    p.live = true;
  })().finally(() => { p.hyd = null; }));
}

// ---- the page list

// cards go in pairs, front then back, one card per A4 sheet (fotocopia_a4.build)
// (no "·" next to a Persian digit: it reads as ۰, «کارت ۰۲»)
const face = k => `${k % 2 ? 'پشت' : 'روی'} کارت ${fa(Math.floor(k / 2) + 1)}`;

// one tile of the grid: the result (the photo while it is scanned or when it failed), a word when
// something is wrong; a tap opens the page, the handle reorders (UX review 22/11: no second photo,
// no pixel sizes, the actions moved onto the page)
function tile(p, k) {
  const busy = p.status === 'busy', err = p.status === 'error', f = flagsOf(p);
  const img = p.url && !err ? p.url : p.thumb;
  const foot = err ? `<p class="tf bad">${ico('alert')}${ERR_SHORT[p.errk] || 'اسکن نشد'}</p>`
    : f.length ? `<p class="tf warn">${ico('alert')}${QA[f[0]][0]}${f.length > 1 ? ` و ${fa(f.length - 1)} مورد دیگر` : ''}</p>` : '';
  return `<article class="tile${busy ? ' busy' : ''}${err ? ' err' : ''}" data-pid="${p.pid}" role="button" aria-label="صفحهٔ ${fa(k + 1)}">`
    + `<div class="th"><b class="num">${fa(k + 1)}</b>${S.kind === 'card' ? `<span class="side">${face(k)}</span>` : ''}`
    + `<button class="grip" data-grip="${p.pid}" aria-label="جابه‌جا کردن صفحهٔ ${fa(k + 1)}">${ico('grip')}</button></div>`
    + `<div class="pic">${img ? `<img src="${img}" alt="" draggable="false">` : ''}${busy ? '<span class="spin"></span>' : ''}</div>${foot}</article>`;
}

// after the last page, so it reads as "the next page goes here" (Razi, 19/11); alone and wide at first
function addTile(n) {
  const next = S.kind === 'card' ? ` (${face(n)})` : '';
  const cam = `<label class="btn" for="cam">${ico('camera')}عکس بگیر</label>`, lib = `<label class="btn alt" for="lib">${ico('gallery')}از گالری</label>`;
  return n ? `<div class="tile add" id="add"><p class="add-lbl">${ico('plus')}<span id="addlbl">صفحهٔ ${fa(n + 1)}${next}</span></p>${cam}${lib}</div>`
    : `<div class="tile add first" id="add"><p class="add-lbl"><span id="addlbl">صفحهٔ ۱${next}: از سند عکس بگیرید یا از گالری انتخاب کنید</span></p>`
      + `<p class="add-tip">سند را روی زمینهٔ تیره و ساده بگذارید تا لبه‌اش پیدا شود.</p><div class="go">${cam}${lib}</div></div>`;
}

// the photo step waits for the kind and, for a «سند دیگر», for the size answer (with «می‌دانم»: both
// sides > 0); until then the box asks (Razi, 22/11)
function sizeOk() {
  if (S.kind !== 'doc') return true;
  return S.sizeby === 'mm' ? parseFloat($('#mmw').value) > 0 && parseFloat($('#mmh').value) > 0 : !!S.sizeby;
}
let gateOpen = null;

function render() {
  if (D) return;                                  // never rebuild the list under a dragging finger
  const n = S.pages.length, ask = !sizeOk();
  gateOpen = !!S.kind && !ask;
  $('#pages').innerHTML = S.pages.map(tile).join('') + (gateOpen ? addTile(n) : '');
  $('#kindbox').classList.toggle('need', !S.kind || ask);
  $('#opt-doc').classList.toggle('need', ask);
  $('#kindlbl').classList.toggle('need', !S.kind);
  $('#kindlbl').textContent = S.kind ? 'نوع مدرک' : 'اول نوع مدرک را انتخاب کنید';
  const ok = S.pages.filter(p => p.status === 'ok');
  $('#dock').hidden = !ok.length;
  $('#bar').hidden = !n;
  $('#count').textContent = `${fa(n)} صفحه`;
  const warn = ok.filter(p => flagsOf(p).length);
  $('#qasum').hidden = !warn.length;
  $('#qasum').dataset.pid = warn.length ? warn[0].pid : '';
  // one line above the buttons; what is wrong is on each tile, and a tap opens the first such page
  const nums = warn.map(p => fa(S.pages.indexOf(p) + 1));
  $('#qasum').innerHTML = warn.length ? `${ico('alert')}<span>${warn.length === 1
    ? `صفحهٔ ${nums[0]}: ${flagsOf(warn[0]).map(x => QA[x][0]).join('، ')}`
    : `هشدار کیفیت در صفحه‌های ${nums.slice(0, -1).join('، ')} و ${nums[nums.length - 1]}`}؛ بزنید تا ببینید</span>` : '';
  $('#pages').className = 'k-' + S.kind;
  $('#name').placeholder = defaultName();
  showBig();
  // the content ends above the bar at the bottom
  $('main').style.paddingBottom = $('#dock').hidden ? '' : `${$('#dock').offsetHeight + 24}px`;
  if (PG) drawPage();
}

// ---- one page, full screen: the result large, what is wrong with it and what to do, and its actions

function openPage(pid) {
  PG = pid;
  drawPage();
  $('#pg').hidden = false;
  document.body.style.overflow = 'hidden';
}

function closePage() {
  PG = null;
  $('#pg').hidden = true;
  $('#pg-body').innerHTML = '';
  if ($('#pv').hidden) document.body.style.overflow = '';
}

function drawPage() {
  const k = S.pages.findIndex(x => x.pid === PG);
  if (k < 0) return closePage();
  const p = S.pages[k], i = p.info, busy = p.status === 'busy', err = p.status === 'error';
  $('#pg').classList.toggle('busy', busy);
  $('#pg').classList.toggle('err', err);
  $('#pg-title').textContent = `صفحهٔ ${fa(k + 1)} از ${fa(S.pages.length)}${S.kind === 'card' ? `، ${face(k)}` : ''}`;
  const img = p.url && !err ? p.url : p.thumb;
  $('#pg-body').innerHTML = (img ? `<img src="${img}" alt="صفحهٔ ${fa(k + 1)}">` : '') + (busy ? '<span class="spin"></span>' : '');
  let msg = '';
  if (err) msg = `<p class="err">${esc(p.error)}${p.cropable ? ' با «برش» گوشه‌ها را خودتان بگذارید.' : ''}</p>`;
  else if (i && !busy) {
    msg = flagsOf(p).map(x => `<p><span class="chip">${ico('alert')}${QA[x][0]}</span><span class="tip">${QA[x][1]}</span></p>`).join('');
    const notes = [];
    if (i.more && i.more.length) notes.push(`${fa(i.more.length + 1)} کارت در این عکس بود؛ هر کدام صفحهٔ جدا شد`);
    else if (p.opts && p.opts.card_no) notes.push(`کارت ${fa(p.opts.card_no)} از همان عکس`);
    if (i.mm && S.kind === 'doc') notes.push(`اندازهٔ واقعی ${fa(Math.round(i.mm[0]))} × ${fa(Math.round(i.mm[1]))} میلی‌متر`);
    if (notes.length) msg += `<p>${notes.join('؛ ')}</p>`;
  }
  if (DEBUG && p.log) msg += `<details><summary>جزئیات</summary><pre>${esc(p.log)}</pre></details>`;
  $('#pg-msg').innerHTML = msg;
  const b = a => $(`#pg-tools [data-act=${a}]`);
  b('edit').disabled = !p.cropable || busy;
  b('rot').disabled = p.status !== 'ok';
  b('retake').disabled = b('del').disabled = busy;
}

function act(p, a) {
  if (a === 'rot') return rerun(p, 'rotate', {});
  if (a === 'edit') return edit(p);
  if (a === 'retake') { retakePid = p.pid; return $('#re').click(); }
  if (!confirm(`صفحهٔ ${fa(S.pages.indexOf(p) + 1)} حذف شود؟`)) return;
  S.pages = S.pages.filter(x => x !== p);
  forget(p);
  saveMeta();
  clearPdf();
  closePage();
  render();
}

// a sheet larger than A4 at 100%: the same test as fotocopia_a4.larger_than_a4 (4% for the card's
// measuring error), here for a size typed on the phone or measured by the card before the PDF
const bigSheet = mm => !!mm && (Math.max(...mm) > 297 * 1.04 || Math.min(...mm) > 210 * 1.04);

function bigMm() {
  if (S.kind !== 'doc' || S.out !== 'print') return null;
  const typed = opts().mm;
  if (typed) return bigSheet(typed) ? typed : null;
  const p = S.pages.find(x => x.status === 'ok' && x.info && bigSheet(x.info.mm));
  return p ? p.info.mm : null;
}

// Razi's rule: a copy may be larger than the original, never smaller. So before the PDF it says
// so and offers A3 at 117% (default: one sheet at any copy shop) or A4 tiles at 117% (home printer)
function showBig() {
  const mm = bigMm();
  $('#bigbox').hidden = !mm;
  if (!mm) return;
  const a = Math.min(...mm), b = Math.max(...mm), k = x => fa(Math.round(x * 1.17));
  const a3 = b * 1.17 <= 400 && a * 1.17 <= 277;
  $('#bigmsg').innerHTML = `${ico('alert')}<span>این برگه از A4 بزرگ‌تر است (${fa(Math.round(a))} × ${fa(Math.round(b))} میلی‌متر). کپی هیچ‌وقت کوچک‌تر از اصل نمی‌شود، پس روی A4 کوچکش نمی‌کنیم. روی چه کاغذی چاپ شود؟</span>`;
  $('#paperhelp').textContent = (S.paper === 'tiles'
    ? `چند برگ A4، هر تکه ۱۱۷٪ (کل ${k(a)} × ${k(b)} میلی‌متر)، هر تکه ۱ سانت روی تکهٔ بعد؛ برای چاپگر خانه: تکه‌ها را کنار هم بچسبانید.`
    : a3 ? `یک برگ A3 با ۱۱۷٪ (${k(a)} × ${k(b)} میلی‌متر)؛ هر کپی‌فروشی A3 چاپ می‌کند.`
      : 'یک برگ A3، بزرگ‌ترین اندازه‌ای که روی آن جا شود.')
    + ' برای پورتال یا ایمیل «برای ارسال و آپلود» را بزنید: یک صفحه به اندازهٔ واقعی خود سند.';
}

function clearPdf() {
  S.pdf = null;
  $('#pv-share').disabled = true;
  $('#pdfinfo').textContent = '';
  $('#pdfinfo').className = 'note';
}

// one page through the scanner, from its photo (new, retaken, or kept while it was being scanned)
async function process(p) {
  if (!S.pages.includes(p)) return;
  try {
    const d = await decode(p.src);
    if (p.thumb) URL.revokeObjectURL(p.thumb);
    Object.assign(p, { tb: d.tb, thumb: URL.createObjectURL(d.tb), ow: d.w, oh: d.h });
    render();
    if (S.ready) setStatus('در حال اسکن…', true);
    // on the phone before Python starts: a photo that closes the app (iOS memory) is not scanned
    // again at every launch, which would close it again (19/11: Razi's passport photo crashed it)
    p.tries = (p.tries || 0) + 1;
    await save(p);
    const t0 = performance.now();
    apply(p, await call('scan', { pid: p.pid, buf: d.buf, w: d.w, h: d.h, opts: JSON.stringify(p.opts) }, [d.buf]));
    p.live = true;
    p.mode = p.opts.mode;
    p.ms = Math.round(performance.now() - t0);
  } catch (err) {
    p.status = 'error';
    p.error = 'خطا: ' + err.message;
    p.errk = null;
  }
  render();
  save(p);
  // several cards in one photo (21/11: Razi's photo of four cards gave one): each other card
  // becomes its own page right after this one, from the same photo and the outline found for it
  const more = p.status === 'ok' && !p.opts.card_pts && p.info && p.info.more || [];
  if (more.length && S.pages.includes(p)) {
    const extra = more.map((pts, j) => Object.assign(newPage(p.src), { opts: { ...p.opts, card_pts: pts, card_no: j + 2 } }));
    S.pages.splice(S.pages.indexOf(p) + 1, 0, ...extra);
    for (const x of extra) { savePhoto(x); save(x); }
    saveMeta();
    render();
    for (const x of extra) await process(x);
  }
}

function enqueue(pages) {
  queue = queue.then(async () => {
    for (const p of pages) await process(p);
    if (S.ready) setStatus('آماده', false);
  });
  return queue.then(() => pages);
}

function newPage(file) {
  return { pid: 'p' + (++pageSeq), status: 'busy', src: file, opts: opts(), mode: S.mode };
}

function addFiles(files) {
  const made = files.map(newPage);
  S.pages.push(...made);
  for (const p of made) { savePhoto(p); save(p); }
  saveMeta();
  clearPdf();
  render();
  return enqueue(made);
}

// a new photo for one page, in its place
function retake(pid, file) {
  const k = S.pages.findIndex(x => x.pid === pid);
  if (k < 0) return Promise.resolve([]);
  forget(S.pages[k]);
  const p = newPage(file);
  S.pages[k] = p;
  if (PG === pid) PG = p.pid;                     // the open page follows its new photo
  savePhoto(p);
  save(p);
  saveMeta();
  clearPdf();
  render();
  return enqueue([p]);
}

async function rerun(p, cmd, args, mode) {
  p.status = 'busy';
  render();
  try {
    await ensure(p);
    apply(p, await call(cmd, { pid: p.pid, ...args }));
    if (mode) p.mode = mode;
  } catch (err) { p.status = 'error'; p.error = 'خطا: ' + err.message; p.errk = null; }
  clearPdf();
  render();
  save(p);
}

function edit(p) {
  return openEditor(p, S.kind, (pts, rot) => rerun(p, 'crop', { opts: JSON.stringify({ pts, rot, mode: S.mode }) }, S.mode))
    .catch(err => setStatus('برش دستی باز نشد: ' + err.message, false, true));
}

// a new document asks for its real size again: no answer, empty boxes
function resetSize() {
  S.sizeby = null;
  $('#mmw').value = $('#mmh').value = '';
  press('#sizeby', null);
}

// what a start keeps of the size answer: only with kept pages of a «سند دیگر», whose print size
// comes from it (the PDF takes the typed mm); otherwise it is asked again
function keptSize(m, kind, n) {
  if (!m || kind !== 'doc' || !n) return { sizeby: null, mmw: '', mmh: '' };
  // before 22/11 the size was a ticked card and two boxes
  return { sizeby: m.sizeby || (m.ref ? 'ref' : m.mmw && m.mmh ? 'mm' : 'none'), mmw: m.mmw || '', mmh: m.mmh || '' };
}

function clearAll() {
  for (const p of S.pages) forget(p);
  S.pages = [];
  resetSize();
  showOpts();
  store.clear();
  saveMeta();
  clearPdf();
  render();
}

// ---- drag to reorder: a large handle, the tile follows the finger in the grid, nothing else scrolls

function dragStart(e) {
  const h = e.target.closest('[data-grip]');
  if (!h || D || (e.pointerType === 'mouse' && e.button !== 0)) return;
  e.preventDefault();
  const art = h.closest('.tile'), r = $('#pages').getBoundingClientRect();
  // where the finger holds the tile; offsetLeft/Top are in #pages (position:relative) and ignore the transform
  D = { art, id: e.pointerId, gx: e.clientX - r.left - art.offsetLeft, gy: e.clientY - r.top - art.offsetTop, cx: e.clientX, cy: e.clientY, raf: 0 };
  try { h.setPointerCapture(e.pointerId); } catch (_) { /* synthetic pointers have no capture */ }
  art.classList.add('drag');
  document.documentElement.classList.add('dragging');
  D.raf = requestAnimationFrame(autoScroll);
}

// the tile under the dragged tile's centre gives up its place: before it when moving back, after it when forward
function place() {
  const { art } = D, list = $('#pages'), r = list.getBoundingClientRect();
  const x = D.cx - r.left - D.gx, y = D.cy - r.top - D.gy;
  const mx = x + art.offsetWidth / 2, my = y + art.offsetHeight / 2;
  const tiles = [...list.querySelectorAll('.tile[data-pid]')];
  const t = tiles.find(o => o !== art && mx >= o.offsetLeft && mx < o.offsetLeft + o.offsetWidth
    && my >= o.offsetTop && my < o.offsetTop + o.offsetHeight);
  if (t) list.insertBefore(art, tiles.indexOf(t) > tiles.indexOf(art) ? t.nextSibling : t);
  art.style.transform = `translate(${x - art.offsetLeft}px, ${y - art.offsetTop}px)`;
}

function autoScroll() {
  if (!D) return;
  const edge = 80, y = D.cy;
  const v = y < edge ? -(edge - y) / 3 : y > innerHeight - edge ? (y - innerHeight + edge) / 3 : 0;
  if (v) { scrollBy(0, v); place(); }
  D.raf = requestAnimationFrame(autoScroll);
}

function dragMove(e) {
  if (!D || e.pointerId !== D.id) return;
  e.preventDefault();
  D.cx = e.clientX;
  D.cy = e.clientY;
  place();
}

function dragEnd(e) {
  if (!D || e.pointerId !== D.id) return;
  cancelAnimationFrame(D.raf);
  D.art.style.transform = '';
  D.art.classList.remove('drag');
  document.documentElement.classList.remove('dragging');
  const order = [...$('#pages').querySelectorAll('.tile[data-pid]')].map(a => a.dataset.pid);
  const moved = order.join() !== S.pages.map(p => p.pid).join();
  S.pages.sort((a, b) => order.indexOf(a.pid) - order.indexOf(b.pid));
  D = null;
  if (moved) { clearPdf(); saveMeta(); }
  render();
}

// ---- the PDF

function stamp() {
  const d = new Date(), z = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${z(d.getMonth() + 1)}-${z(d.getDate())}_${z(d.getHours())}${z(d.getMinutes())}`;
}

const defaultName = () => `Fotocopia-${bigMm() && S.paper === 'a3' ? 'A3' : 'A4'}-${S.kind || 'doc'}-${stamp()}`;

function fileName() {
  const n = $('#name').value.trim().replace(/[\\/:*?"<>|\u0000-\u001f]+/g, '-').replace(/\.pdf$/i, '').slice(0, 90).trim();
  return (n || defaultName()) + '.pdf';
}

const size = b => b >= 1e6 ? `${fa((b / 1e6).toFixed(1))} مگابایت` : `${fa(Math.max(1, Math.round(b / 1e3)))} کیلوبایت`;

async function makePdf() {
  const pages = S.pages.filter(p => p.status === 'ok');
  if (!pages.length) return null;
  const bs = [$('#pdf'), $('#pdfup')];
  bs.forEach(b => { b.disabled = true; });
  $('#pv-share').disabled = true;
  try {
    if (pages.some(p => !p.live)) {
      setStatus('آماده‌سازی صفحه‌های نگه‌داشته…', true);
      for (const p of pages) await ensure(p);
      render();
    }
    setStatus('ساخت PDF…', true);
    const up = S.out === 'upload', cap = up ? CAPS[S.cap] : null;
    const m = await call('pdf', { opts: JSON.stringify({ ids: pages.map(p => p.pid), kind: S.kind, copies: +$('#copies').value || 1,
      mm: opts().mm, out: S.out, cap, paper: S.paper }) });
    const i = m.info, name = fileName();
    S.pdf = new File([m.data], name, { type: 'application/pdf' });
    // the size is in the preview's title bar, what each sheet holds in #outhelp: here only the count
    let t = up ? `${fa(i.pages)} صفحه`
      : `${fa(i.pages)} برگ ${[...new Set(i.mm.map(m => Math.max(...m) > 400 ? 'A3' : 'A4'))].join(' و ')}`;
    if (cap && i.fits) t += ` (زیر سقف ${fa(S.cap)} مگابایت${i.quality ? `؛ کیفیت عکس ${fa(i.quality)}٪${i.scale < 1 ? `، وضوح ${fa(Math.round(i.scale * 100))}٪` : ''}` : ''})`;
    const over = cap && !i.fits;
    $('#pdfinfo').className = 'note' + (over ? ' err' : '');
    $('#pdfinfo').textContent = `${t}.` + (over
      ? ` از سقف ${fa(S.cap)} مگابایت بزرگ‌تر ماند؛ صفحه‌های کمتری بگذارید یا سقف بالاتری انتخاب کنید.` : '');
    $('#pv-share').disabled = false;
    setStatus('آماده', false);
    return i;
  } catch (err) {
    setStatus('فایل ساخته نشد: ' + err.message, false, true);
    return null;
  } finally {
    bs.forEach(b => { b.disabled = false; });
  }
}

// the bar's two buttons: one tap from the pages to the finished file, shown before it is shared
async function build(out) {
  S.out = out;
  showOpts();
  saveMeta();
  const i = await makePdf();
  if (i) await openPreview();
  return i;
}

// a setting changed in the preview: the file is made again and shown; taps that come meanwhile
// make one more round, not one each
let rb = null, rbAgain = false;
async function rebuild() {
  clearPdf();
  if ($('#pv').hidden) return;
  if (rb) { rbAgain = true; return rb; }
  $('#pv').classList.add('busy');
  rb = (async () => {
    do { rbAgain = false; if (await makePdf()) await openPreview(); } while (rbAgain && !$('#pv').hidden);
  })();
  try { await rb; } finally { rb = null; $('#pv').classList.remove('busy'); }
}

async function setMode(v) {
  if (v === S.mode) return;
  S.mode = v;
  press('#mode', v);
  saveMeta();
  if (!$('#pv').hidden) { clearPdf(); $('#pv').classList.add('busy'); }
  for (const p of S.pages.filter(p => p.status === 'ok')) await rerun(p, 'restyle', { mode: v }, v);
  await rebuild();
}

function setCopies(d) {
  const c = $('#copies'), v = Math.min(10, Math.max(1, (+c.value || 1) + d));
  if (v === +c.value) return;
  c.value = v;
  saveMeta();
  return rebuild();
}

// the finished file before it is shared (Razi, 19/11): each page drawn from the PDF itself by
// PyMuPDF in the worker, so what is seen is what is sent; a tap on a page zooms it
let pvUrls = [];
async function openPreview() {
  if (!S.pdf) return false;
  setStatus('آماده‌سازی پیش‌نمایش…', true);
  try {
    const width = Math.min(1400, Math.round(Math.min(screen.width, 560) * (devicePixelRatio || 1) * 1.5));
    const m = await call('preview', { opts: JSON.stringify({ width }) });
    clearPv();
    const pages = m.info.pages;
    pvUrls = m.datas.map(d => URL.createObjectURL(new Blob([d], { type: 'image/jpeg' })));
    $('#pv-pages').innerHTML = pages.map((p, k) => `<figure><img src="${pvUrls[k]}" width="${p.w}" height="${p.h}" alt="صفحهٔ ${fa(k + 1)}">`
      + `<figcaption>صفحهٔ ${fa(k + 1)} از ${fa(pages.length)}، ${fa(p.mm[0])} × ${fa(p.mm[1])} میلی‌متر</figcaption></figure>`).join('');
    $('#pv-title').textContent = S.out === 'upload' ? 'فایل برای ارسال' : 'کپی برای چاپ';
    $('#pv-info').textContent = size(S.pdf.size);
    $('#pv').hidden = false;
    document.body.style.overflow = 'hidden';
    setStatus('آماده', false);
    return pages.length;
  } catch (err) {
    setStatus('پیش‌نمایش ساخته نشد: ' + err.message, false, true);
    return false;
  }
}

function closePreview() {
  $('#pv').hidden = true;
  if ($('#pg').hidden) document.body.style.overflow = '';
  clearPv();
}

function clearPv() {
  $('#pv-pages').innerHTML = '';
  $('#pv-pages').classList.remove('zoom');
  pvUrls.forEach(u => URL.revokeObjectURL(u));
  pvUrls = [];
}

// The app added to the Home Screen has no reload button (Razi, 19/11): a new build is looked for at
// launch, when the app comes back to the screen (at most every 10 minutes) and with «بررسی
// به‌روزرسانی»; the service worker takes it at once, and the page offers «به‌روزرسانی» at the top
// (a reload; a scan running finishes first, the pages are kept on the phone).
let swReg = null, swCtl = false, lastCheck = 0;
async function checkUpdate(manual) {
  const say = t => { if (manual) $('#updmsg').textContent = t; };
  if (!swReg) return say('به‌روزرسانی روی این مرورگر در دسترس نیست.');
  lastCheck = Date.now();
  say('در حال بررسی…');
  try { await swReg.update(); } catch (e) { return say('اینترنت در دسترس نیست؛ بعداً دوباره امتحان کنید.'); }
  const w = swReg.installing || swReg.waiting;
  if (!w) return say(`برنامه به‌روز است (نسخهٔ ${VER}).`);
  say('در حال دریافت نسخهٔ تازه…');
  const end = s => s === 'activated' || s === 'redundant';
  await new Promise(r => {
    if (end(w.state)) return r();
    w.addEventListener('statechange', () => end(w.state) && r());
    setTimeout(r, 60000);
  });
  if (w.state === 'activated') { $('#upd').hidden = false; say('نسخهٔ تازه آماده است؛ بالای صفحه «به‌روزرسانی» را بزنید.'); }
  else say('دریافت نسخهٔ تازه کامل نشد؛ دوباره امتحان کنید.');
}

async function applyUpdate() {
  $('#upd-go').disabled = true;
  if (S.pages.some(p => p.status === 'busy')) {
    setStatus('به‌روزرسانی بعد از پایان اسکن…', true);
    await queue;
  }
  location.reload();
}

// «بستن» (Razi, 22/11): a browser closes a window only where it allows it (not an app on the iPhone's
// Home Screen), so the scanner stops instead: a running scan finishes first and is kept, the Python
// worker ends and its memory (~0.5 GB of wasm heap) goes back to the phone, and the screen says so.
// When the app comes back to the screen it starts fresh, as if it had been closed.
async function quit(test = false) {
  if (S.quit) return;
  if (S.pages.some(p => p.status === 'busy')) { setStatus('بستن بعد از پایان اسکن…', true); await queue; }
  if (!test) {
    try { window.close(); } catch (_) { /* not allowed here */ }
    await sleep(300);                             // a window the browser let go is gone by now
  }
  S.quit = true;
  if (worker) worker.terminate();
  worker = null;
  for (const [, w] of waiting) w.rej(new Error('برنامه متوقف شد'));
  waiting.clear();
  for (const p of S.pages) p.live = false;        // Python forgot them; kept on the phone
  closePage();
  closePreview();
  $('#menu').hidden = true;
  $('#bye').hidden = false;
  setStatus('متوقف شد', false);
}

async function share() {
  const f = S.pdf;
  if (!f) return;
  if (navigator.canShare && navigator.canShare({ files: [f] })) {
    try { await navigator.share({ files: [f], title: f.name }); return; }
    catch (e) { if (e.name === 'AbortError') return; }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(f);
  a.download = f.name;
  document.body.append(a);
  a.click();
  a.remove();
}

// ---- settings

function showOpts() {
  $('#opt-card').hidden = S.kind !== 'card' || S.out !== 'print';
  $('#opt-doc').hidden = S.kind !== 'doc';
  $('#size-mm').hidden = S.sizeby !== 'mm';
  $('#sizehelp').textContent = SIZE_HELP[S.sizeby || 'ask'];
  $('#capbox').hidden = S.out !== 'upload';
  $('#outhelp').textContent = S.out === 'upload' ? UPLOAD_HELP : PRINT_HELP[S.kind] || '';
  showBig();
}

function press(group, v) {
  for (const x of $(group).children) x.setAttribute('aria-pressed', String(x.dataset.v === v));
}

function pick(group, fn) {
  $(group).addEventListener('click', e => {
    const b = e.target.closest('button');
    if (b && b.dataset.v) fn(b.dataset.v);
  });
}

function wire() {
  for (const id of ['#cam', '#lib']) {
    $(id).addEventListener('change', e => {
      const fs = [...e.target.files];
      e.target.value = '';
      if (fs.length) addFiles(fs);
    });
  }
  $('#re').addEventListener('change', e => {
    const f = e.target.files[0];
    e.target.value = '';
    if (f && retakePid) retake(retakePid, f);
    retakePid = null;
  });
  pick('#kind', v => {
    if (v === S.kind) return;
    if (S.pages.length && !confirm('با تغییر نوع مدرک، صفحه‌های اسکن‌شده پاک می‌شوند. ادامه بدهم؟')) return;
    S.kind = v;
    press('#kind', v);
    showOpts();
    clearAll();
  });
  // in the preview: each change makes the file again, so its effect is seen before sharing
  pick('#mode', setMode);
  pick('#cap', v => { S.cap = v; press('#cap', v); saveMeta(); rebuild(); });
  pick('#paper', v => { S.paper = v; press('#paper', v); showBig(); saveMeta(); $('#name').placeholder = defaultName(); rebuild(); });
  $('#cp-minus').addEventListener('click', () => setCopies(-1));
  $('#cp-plus').addEventListener('click', () => setCopies(1));
  $('#name').addEventListener('change', () => {          // only the name: the same file, renamed
    saveMeta();
    if (S.pdf) S.pdf = new File([S.pdf], fileName(), { type: 'application/pdf' });
  });
  $('#name').addEventListener('keydown', e => { if (e.key === 'Enter') e.target.blur(); });
  pick('#sizeby', v => { S.sizeby = v; press('#sizeby', v); showOpts(); render(); clearPdf(); saveMeta(); if (v === 'mm' && !$('#mmw').value) $('#mmw').focus(); });
  for (const id of ['#mmw', '#mmh']) {
    $(id).addEventListener('change', () => { clearPdf(); saveMeta(); showBig(); });
    $(id).addEventListener('input', () => { if (S.kind && sizeOk() !== gateOpen) render(); });   // the photo step once both are typed
  }
  $('#pages').addEventListener('click', e => {
    if (e.target.closest('[data-grip]')) return;
    const t = e.target.closest('.tile[data-pid]');
    if (t) openPage(t.dataset.pid);
  });
  $('#pg-tools').addEventListener('click', e => {
    const b = e.target.closest('[data-act]'), p = S.pages.find(x => x.pid === PG);
    if (b && !b.disabled && p && p.status !== 'busy') act(p, b.dataset.act);
  });
  $('#pg-x').addEventListener('click', closePage);
  $('#qasum').addEventListener('click', () => { if ($('#qasum').dataset.pid) openPage($('#qasum').dataset.pid); });
  $('#pages').addEventListener('pointerdown', dragStart);
  addEventListener('pointermove', dragMove, { passive: false });
  addEventListener('pointerup', dragEnd);
  addEventListener('pointercancel', dragEnd);
  document.addEventListener('touchmove', e => { if (D) e.preventDefault(); }, { passive: false });
  $('#pdf').addEventListener('click', () => build('print'));
  $('#pdfup').addEventListener('click', () => build('upload'));
  $('#pv-x').addEventListener('click', closePreview);
  $('#pv-share').addEventListener('click', share);
  $('#pv-pages').addEventListener('click', e => { if (e.target.tagName === 'IMG') $('#pv-pages').classList.toggle('zoom'); });
  const menu = open => { $('#menu').hidden = !open; };
  $('#menu-btn').addEventListener('click', () => menu(true));
  $('#menu-x').addEventListener('click', () => menu(false));
  $('#menu').addEventListener('click', e => { if (e.target === $('#menu')) menu(false); });
  $('#clear').addEventListener('click', () => {           // a new document starts with no kind again
    if (S.pages.length && !confirm(`همهٔ ${fa(S.pages.length)} صفحه از این گوشی پاک شوند؟ این کار برگشت ندارد.`)) return;
    S.kind = null;
    press('#kind', null);
    showOpts();
    clearAll();
    menu(false);
    scrollTo(0, 0);
  });
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape' || !$('#ed').hidden) return;
    if (!$('#menu').hidden) menu(false);
    else if (!$('#pv').hidden) closePreview();
    else if (!$('#pg').hidden) closePage();
  });
  $('#updchk').addEventListener('click', () => checkUpdate(true));
  $('#upd-go').addEventListener('click', applyUpdate);
  $('#quit').addEventListener('click', () => quit());
  $('#bye-go').addEventListener('click', () => location.reload());
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    if (S.quit) return location.reload();          // stopped with «بستن»: back on screen, a fresh start
    if (Date.now() - lastCheck > 600000) checkUpdate(false);
  });
}

// the pages and settings kept from last time; pages caught mid-scan are scanned again
async function restore() {
  const m = await store.getMeta();
  if (m) {
    for (const k of ['kind', 'mode', 'out', 'cap', 'paper']) if (m[k] != null) S[k] = String(m[k]);
    for (const k of ['name', 'copies']) if (m[k] != null) $('#' + k).value = m[k];
  }
  const recs = await store.all();
  const order = (m && m.order) || [];
  const at = pid => { const k = order.indexOf(pid); return k < 0 ? 1e9 + (+pid.slice(1) || 0) : k; };
  recs.sort((a, b) => at(a.pid) - at(b.pid));
  for (const r of recs) {
    if (!r.photo || !r.opts) { store.drop(r.pid); continue; }
    const p = { pid: r.pid, src: r.photo, tb: r.thumb, ow: r.ow, oh: r.oh, opts: r.opts, mode: r.mode, status: r.status || 'busy',
      error: r.error, errk: r.errk, cropable: r.cropable, info: r.info, live: false, tries: r.tries || 0 };
    if (r.thumb) p.thumb = URL.createObjectURL(r.thumb);
    if (r.result) { p.res = r.result; p.url = URL.createObjectURL(r.result); }
    if (p.status === 'ok' && !p.res) p.status = 'busy';
    if (p.status === 'busy' && p.tries) Object.assign(p, { status: 'error', error: ERR.crash, errk: 'crash', cropable: false });
    S.pages.push(p);
    pageSeq = Math.max(pageSeq, +r.pid.slice(1) || 0);
  }
  if (S.pages.length) $('#kept').textContent = `· ${fa(S.pages.length)} صفحه از دفعهٔ قبل نگه داشته شده بود`;
  else S.kind = null;                       // kept pages keep their kind; an empty start asks for it
  const ks = keptSize(m, S.kind, S.pages.length);
  S.sizeby = ks.sizeby;
  $('#mmw').value = ks.mmw;
  $('#mmh').value = ks.mmh;
  for (const [g, v] of [['#kind', S.kind], ['#sizeby', S.sizeby], ['#mode', S.mode], ['#cap', S.cap], ['#paper', S.paper]]) press(g, v);
  showOpts();
  render();
}

// the grid's reorder (22/11): three cards, the first one's handle dragged onto the third tile by a
// finger's pointer events; it lands after it, the numbers stay in place, and no page opens
async function dragTest() {
  clearAll();
  S.kind = 'card';
  const blob = await (await fetch('test/card.png', { cache: 'no-store' })).blob();
  await addFiles([0, 1, 2].map(k => new File([blob], `card${k}.png`, { type: 'image/png' })));
  scrollTo(0, 0);
  const before = S.pages.map(p => p.pid), tiles = [...document.querySelectorAll('#pages .tile[data-pid]')];
  const grip = tiles[0].querySelector('.grip'), g = grip.getBoundingClientRect(), t = tiles[2].getBoundingClientRect();
  const ev = (type, x, y, el = window) => el.dispatchEvent(new PointerEvent(type, { pointerId: 9, clientX: x, clientY: y,
    bubbles: true, button: 0, pointerType: 'touch', isPrimary: true }));
  const x0 = g.left + g.width / 2, y0 = g.top + g.height / 2, x1 = t.left + 20, y1 = t.top + 20;
  ev('pointerdown', x0, y0, grip);
  for (let k = 1; k <= 10; k++) {
    ev('pointermove', x0 + (x1 - x0) * k / 10, y0 + (y1 - y0) * k / 10);
    await new Promise(r => requestAnimationFrame(r));
  }
  ev('pointerup', x1, y1);
  grip.click();                                   // the click a real tap leaves on the handle
  const after = S.pages.map(p => p.pid), nums = [...document.querySelectorAll('#pages .tile[data-pid] .num')].map(n => n.textContent);
  const want = [before[1], before[2], before[0]];
  const kept = ((await store.getMeta()) || {}).order;
  const r = { before, after, nums, kept, opened: !$('#pg').hidden };
  r.pass = after.join() === want.join() && nums.join() === '۱,۲,۳' && !r.opened && (!store.ok || (kept || []).join() === want.join());
  closePage();
  return r;
}

// pinch zoom in the hand crop (22/11): fingers' pointer events on the stage of the last scanned page.
// Two fingers 60 -> 180 px apart zoom 3x and keep the photo point under them; one finger pans a zoomed
// photo; a double tap and the small button go back to fit; a handle dragged while zoomed lands on
// the photo point its screen motion means, and «اعمال» sends exactly that point (rounded to 0.1)
async function zoomTest() {
  const p = S.pages.find(x => x.status === 'ok');
  if (!p) return { pass: false, error: 'no page' };
  let sent = null;
  await openEditor(p, S.kind, pts => { sent = pts; });
  const st = $('#ed-stage'), R = st.getBoundingClientRect(), es = editorState;
  const ev = (type, id, x, y) => st.dispatchEvent(new PointerEvent(type, { pointerId: id, clientX: R.left + x, clientY: R.top + y,
    bubbles: true, cancelable: true, button: 0, pointerType: 'touch', isPrimary: id === 1 }));
  const raf = () => new Promise(r => requestAnimationFrame(r));
  const scr = (q, v = es()) => [v.ox + q[0] * v.scale, v.oy + q[1] * v.scale];
  const near = (a, b, e) => Math.abs(a[0] - b[0]) <= e && Math.abs(a[1] - b[1]) <= e;
  const P0 = es().P.map(q => q.slice()), f = es().fit;
  const atFit = () => { const v = es(); return Math.abs(v.scale - f.scale) < 1e-9 && Math.abs(v.ox - f.ox) < 1e-6 && Math.abs(v.oy - f.oy) < 1e-6; };
  const r = { fit: +f.scale.toFixed(4), stage: [Math.round(R.width), Math.round(R.height)], start_fit: atFit() };
  const pinch = async (cx, cy, from, to) => {
    ev('pointerdown', 1, cx - from, cy);
    ev('pointerdown', 2, cx + from, cy);
    for (let k = 1; k <= 6; k++) {
      const d = from + (to - from) * k / 6;
      ev('pointermove', 1, cx - d, cy);
      ev('pointermove', 2, cx + d, cy);
    }
    await raf();
    const pinching = es().pinching;
    ev('pointerup', 1, cx - to, cy);
    ev('pointerup', 2, cx + to, cy);
    return pinching;
  };
  // 1) pinch at the stage's centre
  const cx = R.width / 2, cy = R.height / 2, under = [(cx - f.ox) / f.scale, (cy - f.oy) / f.scale];
  const pinching = await pinch(cx, cy, 30, 90);
  let v = es();
  r.pinch = { pinching, x: +(v.scale / f.scale).toFixed(3), drift_px: +Math.hypot(...[0, 1].map(k => scr(under, v)[k] - [cx, cy][k])).toFixed(3),
    points_same: JSON.stringify(v.P) === JSON.stringify(P0) };
  r.pinch.pass = pinching && Math.abs(r.pinch.x - 3) < 0.01 && r.pinch.drift_px < 0.5 && r.pinch.points_same;
  // 2) one finger off the handles pans the zoomed photo, nothing else moves
  const o0 = [v.ox, v.oy];
  ev('pointerdown', 3, cx, cy);
  for (let k = 1; k <= 5; k++) ev('pointermove', 3, cx - 8 * k, cy - 6 * k);
  ev('pointerup', 3, cx - 40, cy - 30);
  v = es();
  r.pan = { d: [+(v.ox - o0[0]).toFixed(3), +(v.oy - o0[1]).toFixed(3)], points_same: JSON.stringify(v.P) === JSON.stringify(P0), scale_same: v.scale === f.scale * 3 || Math.abs(v.scale / f.scale - 3) < 0.01 };
  r.pan.pass = near(r.pan.d, [-40, -30], 1e-6) && r.pan.points_same && r.pan.scale_same;
  // 3) a double tap goes back to the whole photo
  for (let k = 0; k < 2; k++) { ev('pointerdown', 4, cx, cy); ev('pointerup', 4, cx, cy); }
  r.double_tap = { fit: atFit() };
  r.double_tap.pass = r.double_tap.fit;
  // 4) a pinch around the first corner (the first finger lands on its handle and starts a drag: the
  //    second makes it a pinch and the corner goes back), then that corner dragged while zoomed
  const i = geom(P0).c[0];
  const c0 = scr(P0[i]);
  const pinching2 = await pinch(c0[0] + 40, c0[1], 40, 120);
  v = es();
  let c1 = scr(P0[i], v);
  if (Math.hypot(c1[0] - cx, c1[1] - cy) > 60) {      // pan the zoomed corner towards the middle
    const d = [cx - c1[0], cy - c1[1]];
    ev('pointerdown', 6, cx, cy);
    for (let k = 1; k <= 5; k++) ev('pointermove', 6, cx + d[0] * k / 5, cy + d[1] * k / 5);
    ev('pointerup', 6, cx + d[0], cy + d[1]);
    v = es();
    c1 = scr(P0[i], v);
  }
  const inside = c1[0] > 10 && c1[1] > 10 && c1[0] < R.width - 10 && c1[1] < R.height - 10;
  r.corner = { pinching: pinching2, x: +(v.scale / f.scale).toFixed(3), points_same: JSON.stringify(v.P) === JSON.stringify(P0), on_stage: inside };
  const mv = [18, 12], want = [P0[i][0] + mv[0] / v.scale, P0[i][1] + mv[1] / v.scale];
  ev('pointerdown', 5, c1[0], c1[1]);
  for (let k = 1; k <= 6; k++) ev('pointermove', 5, c1[0] + mv[0] * k / 6, c1[1] + mv[1] * k / 6);
  await raf();
  const during = es();
  ev('pointerup', 5, c1[0] + mv[0], c1[1] + mv[1]);
  await raf();
  v = es();
  const got = v.P[i], err = Math.hypot(got[0] - want[0], got[1] - want[1]);
  const others = v.P.every((q, k) => k === i || (q[0] === P0[k][0] && q[1] === P0[k][1]));
  const onScreen = scr(got, v);
  Object.assign(r.corner, { dragging: during.dragging, loupe_on: during.loupe, loupe_off: !v.loupe, err_px: +err.toFixed(6),
    screen_err: +Math.hypot(onScreen[0] - c1[0] - mv[0], onScreen[1] - c1[1] - mv[1]).toFixed(6), others, fit_btn: v.fitBtn });
  r.corner.pass = r.corner.pinching && r.corner.x > 2.5 && r.corner.points_same && inside && during.dragging && during.loupe && !v.loupe
    && err < 1e-6 && r.corner.screen_err < 1e-6 && others && v.fitBtn;
  // 5) the small button goes back to fit, and «اعمال» sends the dragged point
  $('#ed-fit').click();
  await raf();
  r.button = { fit: atFit(), hidden: !es().fitBtn };
  r.button.pass = r.button.fit && r.button.hidden;
  $('#ed-ok').click();
  r.sent = { pts: !!sent, err: sent ? +Math.hypot(sent[i][0] - want[0], sent[i][1] - want[1]).toFixed(3) : null, closed: !es().open };
  r.sent.pass = !!sent && r.sent.err <= 0.071 && r.sent.closed && sent.every((q, k) => k === i || near(q, P0[k], 0.051));
  r.pass = r.start_fit && r.pinch.pass && r.pan.pass && r.double_tap.pass && r.corner.pass && r.button.pass && r.sent.pass;
  return r;
}

// ?test: scans the synthetic photos in test/ and compares with what the laptop produced (build.py)
async function selfTest() {
  const out = { cases: [] };
  const same = (a, b) => JSON.stringify(a || []) === JSON.stringify(b || []);
  try {
    // no kind at the start: the box asks for it and no photo step is offered; a tap on a kind brings
    // it; «سند دیگر» asks for its real size with no answer chosen, and the photo step waits for one
    // («می‌دانم»: both sides > 0); a new document and a kind change ask again (Razi, 22/11)
    render();
    const g = out.gate = { add_before: !!$('#add'), asks: $('#kindbox').classList.contains('need') };
    const add = () => !!$('#add'), sizeAsks = () => !$('#opt-doc').hidden && $('#opt-doc').classList.contains('need');
    const unset = () => S.sizeby === null && [...$('#sizeby').children].every(b => b.getAttribute('aria-pressed') === 'false');
    const typ = (id, v) => { $(id).value = v; $(id).dispatchEvent(new Event('input', { bubbles: true })); };
    $('#kind button[data-v="a4"]').click();
    g.a4 = { add: add(), size_q: !$('#opt-doc').hidden };            // other kinds: no size question
    $('#kind button[data-v="doc"]').click();
    g.doc = { add: add(), size_asks: sizeAsks(), unset: unset(), box_asks: $('#kindbox').classList.contains('need') };
    $('#sizeby button[data-v="mm"]').click();
    g.mm = { shown: !$('#size-mm').hidden, add_empty: add() };
    typ('#mmw', '102');
    g.mm.add_one = add();
    typ('#mmh', '330');
    Object.assign(g.mm, { add_both: add(), asks_after: sizeAsks() });
    typ('#mmh', '0');
    g.mm.add_zero = add();
    $('#sizeby button[data-v="none"]').click();
    g.none = { add: add(), mm_hidden: $('#size-mm').hidden, asks: sizeAsks() };
    $('#sizeby button[data-v="ref"]').click();
    g.ref = { add: add() };
    $('#clear').click();                              // «سند تازه» (no pages: no confirm)
    $('#kind button[data-v="doc"]').click();
    g.new_doc = { add: add(), unset: unset(), mm: $('#mmw').value + $('#mmh').value, asks: sizeAsks() };
    $('#sizeby button[data-v="none"]').click();
    $('#kind button[data-v="card"]').click();
    $('#kind button[data-v="doc"]').click();
    g.kind_change = { add: add(), unset: unset() };
    // a start keeps the answer only with kept pages of a «سند دیگر» (their print size comes from it)
    const m = { sizeby: 'mm', mmw: '102', mmh: '330' };
    g.kept = { doc_pages: keptSize(m, 'doc', 2), doc_empty: keptSize(m, 'doc', 0), card: keptSize(m, 'card', 3), old: keptSize({ ref: true }, 'doc', 1) };
    g.pass = !g.add_before && g.asks && !g.a4.size_q && g.a4.add && !g.doc.add && g.doc.size_asks && g.doc.unset && g.doc.box_asks
      && g.mm.shown && !g.mm.add_empty && !g.mm.add_one && g.mm.add_both && !g.mm.asks_after && !g.mm.add_zero
      && g.none.add && g.none.mm_hidden && !g.none.asks && g.ref.add
      && !g.new_doc.add && g.new_doc.unset && g.new_doc.mm === '' && g.new_doc.asks
      && !g.kind_change.add && g.kind_change.unset
      && JSON.stringify(g.kept.doc_pages) === JSON.stringify(m) && g.kept.doc_empty.sizeby === null && g.kept.card.sizeby === null
      && g.kept.card.mmw === '' && g.kept.old.sizeby === 'ref';
    clearAll();
    const exp = await (await fetch('test/expected.json', { cache: 'no-store' })).json();
    for (const c of exp.cases) {
      clearAll();
      S.kind = c.kind;
      S.sizeby = c.big ? 'mm' : 'none';
      S.mode = 'color';
      S.out = 'print';
      S.paper = 'a3';
      $('#mmw').value = c.big ? c.big.mm[0] : '';       // a typed real size (the bollettino case)
      $('#mmh').value = c.big ? c.big.mm[1] : '';
      const blob = await (await fetch('test/' + c.file, { cache: 'no-store' })).blob();
      const [p] = await addFiles([new File([blob], c.file, { type: blob.type || 'image/png' })]);
      const i = p.info || {};
      const a = (i.pts || []).flat(), b = (c.pts || []).flat();
      const dq = a.length === b.length && a.length ? Math.max(...a.map((v, k) => Math.abs(v - b[k]))) : (a.length || b.length ? 999 : 0);
      const near = (x, w) => Math.abs(x.w - w.w) <= 2 && Math.abs(x.h - w.h) <= 2 && Math.abs(x.mean - w.mean) <= 2;
      const flags = (i.quality || {}).flags;
      const r = { file: c.file, ms: p.ms, pts_px: +dq.toFixed(2), planes: i.planes, size: [i.w, i.h], want: [c.w, c.h],
        mean_diff: +((i.mean || 0) - c.mean).toFixed(2), flags, want_flags: c.flags, error: p.error };
      r.pass = !p.error && dq <= 2 && i.planes === c.planes && near(i, c) && same(flags, c.flags);
      // the grid has a tile per page; a tap opens the page with its result and its actions
      r.tiles = document.querySelectorAll('#pages .tile[data-pid]').length;
      $(`#pages .tile[data-pid="${p.pid}"] .pic`).click();
      const pg = $('#pg-body img');
      r.page_view = !$('#pg').hidden && !!pg && pg.getAttribute('src') === p.url && !$('#pg-tools [data-act=edit]').disabled;
      closePage();
      r.pass = r.pass && r.tiles === S.pages.length && r.page_view && $('#pg').hidden;
      r.heap_mb = Math.round((S.heap || 0) / 1e6);    // Pyodide's heap after this scan (it never shrinks)
      if (c.heap_mb_max) r.pass = r.pass && r.heap_mb > 0 && r.heap_mb <= c.heap_mb_max;
      const kept = (await store.all()).find(x => x.pid === p.pid);   // autosave: photo, result and crop on the phone
      r.stored = !!(kept && kept.photo && kept.result && kept.info);
      r.pass = r.pass && (r.stored || !store.ok);
      r.crops = [];
      for (const k of c.crops || []) {               // hand crops: same page size and tone as the laptop
        const t0 = performance.now();
        await rerun(p, 'crop', { opts: JSON.stringify({ pts: k.pts, rot: k.rot, mode: 'color' }) }, 'color');
        const x = p.info || {};
        const ok = !p.error && x.planes === k.planes && near(x, k);
        r.crops.push({ n: k.pts ? k.pts.length : 'whole', ms: Math.round(performance.now() - t0), size: [x.w, x.h], want: [k.w, k.h], mean_diff: +((x.mean || 0) - k.mean).toFixed(2), pass: ok });
        r.pass = r.pass && ok;
      }
      if (c.restore) {          // as after iOS closed the app: Python forgets the page, the kept one comes back
        const k = c.crops[c.restore.crop];
        await rerun(p, 'crop', { opts: JSON.stringify({ pts: k.pts, rot: k.rot, mode: 'color' }) }, 'color');
        await call('drop', { pid: p.pid });
        p.live = false;
        const t0 = performance.now();
        await ensure(p);
        const x = p.info || {};
        r.restore = { ms: Math.round(performance.now() - t0), size: [x.w, x.h], want: [k.w, k.h], pass: !p.error && near(x, k) };
        r.pass = r.pass && r.restore.pass;
      }
      if (c.pdf_pages) {
        const t0 = performance.now();
        r.pdf = await build('print');                // the bar's button: the file made, then shown from the file
        r.pdf_ms = Math.round(performance.now() - t0);
        r.pass = r.pass && !!r.pdf && r.pdf.pages === c.pdf_pages;
        const shown = async () => {
          const imgs = [...document.querySelectorAll('#pv-pages img')];
          await Promise.all(imgs.map(im => im.complete ? 0 : new Promise(ok => { im.onload = im.onerror = ok; setTimeout(ok, 5000); })));
          return imgs.filter(im => im.naturalWidth > 0).length;
        };
        const n = await shown();
        r.preview = { pages: n, open: !$('#pv').hidden, share: !$('#pv-share').disabled };
        // a colour changed in the preview: every page restyled, the file made again and shown again
        const first = S.pdf;
        await setMode('gray');
        r.preview.gray = { pages: await shown(), open: !$('#pv').hidden, remade: !!S.pdf && S.pdf !== first,
          pages_gray: S.pages.every(x => x.status !== 'ok' || x.mode === 'gray'), share: !$('#pv-share').disabled };
        await setMode('color');
        closePreview();
        const g = r.preview.gray;
        r.pass = r.pass && n === c.pdf_pages && r.preview.open && r.preview.share && $('#pv').hidden
          && g.pages === c.pdf_pages && g.open && g.remade && g.pages_gray && g.share && S.mode === 'color';
      }
      if (c.big) {              // larger than A4: said before the PDF, then A3 at 117% or A4 tiles
        showBig();
        const shown = !$('#bigbox').hidden && $('#bigmsg').textContent.includes('بزرگ‌تر');
        const a3 = await makePdf();
        S.paper = 'tiles';
        const tiles = await makePdf();
        S.paper = 'a3';
        const sameMm = (a, b) => !!a && JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
        r.big = { shown, a3_mm: a3 && a3.mm[0], tiles: tiles && tiles.pages,
          pass: shown && !!a3 && a3.pages === 1 && sameMm(a3.mm[0], c.big.a3_mm) && !!tiles && tiles.pages === c.big.tiles };
        r.pass = r.pass && r.big.pass;
        $('#mmw').value = $('#mmh').value = '';
      }
      if (c.upload) {           // the upload PDF: one page at the document's size, under the cap
        S.out = 'upload';
        S.cap = c.upload.cap ? String(c.upload.cap / 1e6) : '0';
        CAPS[S.cap] = c.upload.cap;
        const t0 = performance.now();
        const u = await makePdf();
        const mm = u && u.mm[0], w = c.upload.mm[0];
        r.upload = { ms: Math.round(performance.now() - t0), kb: u && u.kb, quality: u && u.quality, mm, want_mm: w,
          pass: !!u && u.pages === 1 && u.fits && (!c.upload.cap || u.bytes <= c.upload.cap)
            && Math.abs(mm[0] - w[0]) <= 0.6 && Math.abs(mm[1] - w[1]) <= 0.6 };
        r.pass = r.pass && r.upload.pass;
      }
      out.cases.push(r);
    }
    out.zoom = await zoomTest();
    out.drag = await dragTest();
    clearAll();
    // «بستن», last: the open views close, the worker ends, the stopped screen shows (no window.close here)
    $('#menu-btn').click();                       // open, so «بستن» has something to close
    await quit(true);
    out.quit = { bye: !$('#bye').hidden, worker_gone: worker === null, menu_closed: $('#menu').hidden, pg_closed: $('#pg').hidden, pv_closed: $('#pv').hidden };
    out.quit.pass = Object.values(out.quit).every(Boolean);
    $('#bye').hidden = true;                      // the results stay readable,
    S.quit = false;                               // and a hidden-then-shown pane does not reload them away
  } catch (e) {
    out.error = String(e);
  }
  out.pass = !out.error && out.cases.length > 0 && out.cases.every(c => c.pass) && !!out.drag && out.drag.pass
    && !!out.zoom && out.zoom.pass
    && !!out.gate && out.gate.pass && !!out.quit && out.quit.pass;
  const t = $('#test');
  t.hidden = false;
  t.textContent = JSON.stringify(out, null, 1);
  document.title = 'TEST ' + (out.pass ? 'PASS' : 'FAIL');
}

async function boot() {
  paintIcons();
  wire();
  const test = new URLSearchParams(location.search).has('test');
  window.scanApp = {                                // for testing in the browser tools
    pages: () => S.pages, state: S, editor: editorState, edit: k => edit(S.pages[k]), add: fs => addFiles(fs),
    retake: (k, f) => retake(S.pages[k].pid, f), pdf: makePdf, ensure: k => ensure(S.pages[k]), store,
    open: k => openPage(S.pages[k].pid), build, setMode,
  };
  if (test) { store.use('scanner-test'); await store.clear(); showOpts(); } else { store.persist(); await restore(); }
  if (!store.ok) $('#kept').textContent = '· ذخیرهٔ خودکار روی این مرورگر در دسترس نیست';
  $('#updmsg').textContent = `نسخهٔ ${VER}`;
  if ('serviceWorker' in navigator) {
    try {
      swReg = await navigator.serviceWorker.register('sw.js');
      if (!navigator.serviceWorker.controller) {   // first visit: let the worker's downloads go through the cache
        await Promise.race([new Promise(r => navigator.serviceWorker.addEventListener('controllerchange', r, { once: true })), sleep(4000)]);
      }
      swCtl = !!navigator.serviceWorker.controller;  // from now on a new controller is a new build
      navigator.serviceWorker.addEventListener('controllerchange', () => { if (swCtl) $('#upd').hidden = false; });
    } catch (e) {
      console.warn('service worker', e);
    }
  }
  startWorker();
  const left = S.pages.filter(p => p.status === 'busy');
  if (left.length) enqueue(left);
  if (test) selfTest();
  else checkUpdate(false);
}

boot();
