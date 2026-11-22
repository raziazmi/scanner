// Pages kept on the phone itself (IndexedDB), so iOS closing the app in the background loses
// nothing: 'photos' holds each photo once, 'pages' its crop, turn, colour and result, 'meta' the
// settings and the page order. Every access is wrapped: without storage (private browsing,
// blocked, full) the app still works, it only forgets on close, and `ok` turns false.
let NAME = 'scanner', dbp = null, ok = true;

function open() {
  return dbp || (dbp = new Promise((done, bad) => {
    const r = indexedDB.open(NAME, 1);
    r.onupgradeneeded = () => {
      const db = r.result;
      for (const s of ['photos', 'pages']) if (!db.objectStoreNames.contains(s)) db.createObjectStore(s, { keyPath: 'pid' });
      if (!db.objectStoreNames.contains('meta')) db.createObjectStore('meta');
    };
    r.onsuccess = () => done(r.result);
    r.onerror = () => bad(r.error);
    r.onblocked = () => bad(new Error('blocked'));
  }));
}

function run(stores, mode, fn) {
  return open().then(db => new Promise((done, bad) => {
    const t = db.transaction(stores, mode);
    const req = fn(t);
    t.oncomplete = () => done(req ? req.result : undefined);
    t.onerror = () => bad(t.error);
    t.onabort = () => bad(t.error || new Error('aborted'));
  }));
}

async function safe(f, fallback) {
  try { return await f(); } catch (e) { ok = false; console.warn('storage:', e && (e.message || e)); return fallback; }
}

const both = ['photos', 'pages'];
export const store = {
  get ok() { return ok; },
  use(name) { NAME = name; dbp = null; ok = true; },             // the self-test keeps its own database
  persist: () => safe(async () => !!(navigator.storage && navigator.storage.persist && await navigator.storage.persist()), false),
  photo: rec => safe(() => run('photos', 'readwrite', t => t.objectStore('photos').put(rec))),
  page: rec => safe(() => run('pages', 'readwrite', t => t.objectStore('pages').put(rec))),
  drop: pid => safe(() => run(both, 'readwrite', t => { t.objectStore('photos').delete(pid); t.objectStore('pages').delete(pid); })),
  clear: () => safe(() => run(both, 'readwrite', t => { t.objectStore('photos').clear(); t.objectStore('pages').clear(); })),
  getMeta: () => safe(() => run('meta', 'readonly', t => t.objectStore('meta').get('state')), null),
  setMeta: v => safe(() => run('meta', 'readwrite', t => t.objectStore('meta').put(v, 'state'))),
  // every kept page: its photo record merged with its page record (a page without a photo is gone)
  all: () => safe(async () => {
    const [ph, pg] = await Promise.all([run('photos', 'readonly', t => t.objectStore('photos').getAll()),
      run('pages', 'readonly', t => t.objectStore('pages').getAll())]);
    const by = new Map((pg || []).map(r => [r.pid, r]));
    return (ph || []).map(r => ({ ...(by.get(r.pid) || {}), ...r }));
  }, []),
};
