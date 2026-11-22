// Runs the Python scanner (scan_doc.py, fotocopia_a4.py, driver.py) in Pyodide, off the page's thread.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/pyodide.mjs';

const CDN = 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/';
const FILES = ['scan_doc.py', 'fotocopia_a4.py', 'driver.py'];
const post = (m, t) => self.postMessage(m, t || []);
let py, D, pdfLib;
// the wasm heap in bytes: it grows to the largest scan's peak and never shrinks, and iOS closes the
// app near 1-1.5 GB (19/11: a 3000 px passport photo reached that and closed it every time)
const heap = () => { try { return py._module.HEAPU8.length; } catch (e) { return 0; } };

const ready = (async () => {
  post({ type: 'status', text: 'بارگذاری پایتون…', busy: true });
  py = await loadPyodide({ indexURL: CDN, stdout: s => post({ type: 'log', text: s }), stderr: s => post({ type: 'log', text: s }) });
  post({ type: 'status', text: 'بارگذاری OpenCV…', busy: true });
  await py.loadPackage(['numpy', 'opencv-python']);
  for (const f of FILES) {
    const r = await fetch(f);
    if (!r.ok) throw new Error(f + ': ' + r.status);
    const code = await r.text();
    if (f !== 'fotocopia_a4.py') await py.loadPackagesFromImports(code);   // a new import in the scanner loads itself
    py.FS.writeFile('/home/pyodide/' + f, code);
  }
  py.runPython('import sys\nif "/home/pyodide" not in sys.path: sys.path.insert(0, "/home/pyodide")');
  D = py.pyimport('driver');
  post({ type: 'status', text: 'آماده', busy: false, ready: true });
  pdfLib = py.loadPackage(['pymupdf']);            // for the PDF step, in the background
})();
ready.catch(e => post({ type: 'status', text: 'بارگذاری نشد: ' + (e.message || e), busy: false, error: true }));

self.onmessage = async e => {
  const m = e.data;
  try {
    await ready;
    let info;
    if (m.cmd === 'scan') info = D.scan(m.pid, new Uint8Array(m.buf), m.w, m.h, m.opts);
    else if (m.cmd === 'crop') info = D.crop(m.pid, m.opts);
    else if (m.cmd === 'restore') info = D.restore(m.pid, new Uint8Array(m.buf), m.w, m.h, m.opts);
    else if (m.cmd === 'rotate') info = D.rotate(m.pid);
    else if (m.cmd === 'restyle') info = D.restyle(m.pid, m.mode);
    else if (m.cmd === 'drop') { D.drop(m.pid); return post({ id: m.id, type: 'ok', info: {} }); }
    else if (m.cmd === 'pdf') { await pdfLib; info = D.pdf(m.opts); }
    else if (m.cmd === 'preview') { await pdfLib; info = D.preview(m.opts); }
    else throw new Error('unknown command ' + m.cmd);
    const o = JSON.parse(info);
    const data = o.path ? py.FS.readFile(o.path) : null;
    const datas = o.paths ? o.paths.map(p => py.FS.readFile(p)) : null;
    post({ id: m.id, type: 'ok', info: o, data, datas, heap: heap() },
      [...(data ? [data.buffer] : []), ...(datas || []).map(d => d.buffer)]);
  } catch (err) {
    post({ id: m.id, type: 'error', heap: heap(), text: String((err && err.message) || err).split('\n').slice(-3).join(' ') });
  }
};
