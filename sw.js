// Offline: the app's own files per build version, the Pyodide libraries once (their URLs never change).
const VERSION = '2e3036d825';
const SHELL = 'scan-shell-' + VERSION;
const LIB = 'scan-lib-pyodide-0.28.3';
const CDN = 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/';
const FILES = ['./', 'index.html', 'app.js', 'crop.js', 'icons.js', 'store.js', 'worker.js', 'driver.py', 'scan_doc.py', 'fotocopia_a4.py',
  'manifest.webmanifest', 'icon-192.png', 'icon-512.png', 'apple-touch-icon.png'];

self.addEventListener('install', e => {
  // cache: 'reload' so a new build never stores an older file from the HTTP cache (Pages: 10 min)
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(FILES.map(f => new Request(f, { cache: 'reload' }))))
    .then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(ks => Promise.all(ks.filter(k => k !== SHELL && k !== LIB).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const r = e.request;
  if (r.method !== 'GET') return;
  const lib = r.url.startsWith(CDN);
  const url = new URL(r.url);
  if (!lib && url.origin !== location.origin) return;
  if (url.pathname.includes('/test/')) return;                 // the self-test always reads fresh files
  e.respondWith((async () => {
    const c = await caches.open(lib ? LIB : SHELL);
    const hit = await c.match(r, { ignoreSearch: !lib });
    if (hit) return hit;
    const res = await fetch(r);
    if (res.ok) c.put(r, res.clone());
    return res;
  })());
});
