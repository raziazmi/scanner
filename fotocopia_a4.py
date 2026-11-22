import argparse, os, sys
import fitz
MM = 72 / 25.4
K = 1.17
CARD = (85.6, 53.98)
BOX = (190, 277)
A4 = (210, 297)
TOL = 1.04
OVERLAP = 10
LABEL = (240, 90)

def faces(path):
    if not path.lower().endswith('.pdf'):
        pix = fitz.Pixmap(path)
        return [(open(path, 'rb').read(), pix.width, pix.height)]
    d, out = (fitz.open(path), [])
    for p in d:
        ims = [im for im in p.get_images() if (im[2], im[3]) != LABEL]
        if not ims:
            continue
        top = max((im[2] * im[3] for im in ims))
        ims = [im for im in ims if im[2] * im[3] >= top * 0.25]
        ims.sort(key=lambda im: p.get_image_rects(im[0])[0].y0)
        out += [(d.extract_image(im[0])['image'], im[2], im[3]) for im in ims]
    return out

def place(page, raw, w, h, real, y):
    s = K * max(real[0] / w, real[1] / h)
    x = (210 - w * s) / 2
    page.insert_image(fitz.Rect(x * MM, y * MM, (x + w * s) * MM, (y + h * s) * MM), stream=raw)
    print('  %.1f x %.1f mm (after a 94%% fit %.1f x %.1f; real %.1f x %.1f)' % (w * s, h * s, w * s * 0.94, h * s * 0.94, *real))
    return h * s

def per_face(mm, n):
    if isinstance(mm, list) and mm and (not isinstance(mm[0], (int, float))):
        return list(mm)
    return [tuple(mm) if mm else None] * n

def larger_than_a4(mm):
    return bool(mm) and (max(mm) > A4[1] * TOL or min(mm) > A4[0] * TOL)

def big_sheet(out, raw, w, h, px, real, paper):
    land = w > h
    if paper == 'tiles':
        W, H, bw, bh = (297, 210, 277, 190) if land else (210, 297, 190, 277)
        s = K * px
        iw, ih = (w * s, h * s)
        src = fitz.open()
        sp = src.new_page(width=iw * MM, height=ih * MM)
        sp.insert_image(sp.rect, stream=raw)
        xs = [min(i * (bw - OVERLAP), max(0, iw - bw)) for i in range(max(1, -int(-(iw - OVERLAP) // (bw - OVERLAP))))]
        ys = [min(j * (bh - OVERLAP), max(0, ih - bh)) for j in range(max(1, -int(-(ih - OVERLAP) // (bh - OVERLAP))))]
        n = len(xs) * len(ys)
        for j, y0 in enumerate(ys):
            for i, x0 in enumerate(xs):
                cw, ch = (min(bw, iw - x0), min(bh, ih - y0))
                pg = out.new_page(width=W * MM, height=H * MM)
                pg.show_pdf_page(fitz.Rect(10 * MM, 10 * MM, (10 + cw) * MM, (10 + ch) * MM), src, 0, clip=fitz.Rect(x0 * MM, y0 * MM, (x0 + cw) * MM, (y0 + ch) * MM))
                pg.insert_text(((10 + 1) * MM, (H - 4) * MM), '%d / %d  (%d mm overlap)' % (j * len(xs) + i + 1, n, OVERLAP), fontsize=7, color=(0.45, 0.45, 0.45))
        print('  larger than A4: %.1f x %.1f mm (117%%) on %d A4 pages, %d mm overlap (real %.1f x %.1f)' % (iw, ih, n, OVERLAP, *real))
        return
    W, H, bw, bh = (420, 297, 400, 277) if land else (297, 420, 277, 400)
    s = K * px
    if w * s > bw + 0.01 or h * s > bh + 0.01:
        s = min(bw / w, bh / h)
        if s < px:
            s = min(W / w, H / h)
        print('  [!] 117%% does not fit on A3: %.0f%% of real size' % (100 * s / px))
    x, y = ((W - w * s) / 2, 10 if h * s <= bh + 0.01 else (H - h * s) / 2)
    pg = out.new_page(width=W * MM, height=H * MM)
    pg.insert_image(fitz.Rect(x * MM, y * MM, (x + w * s) * MM, (y + h * s) * MM), stream=raw)
    print('  larger than A4: on A3, %.1f x %.1f mm (after a 94%% fit %.1f x %.1f; real %.1f x %.1f)' % (w * s, h * s, w * s * 0.94, h * s * 0.94, *real))

def build(kind, srcs, copies, out, mm=None, paper='a3'):
    fs = [f for s in srcs for f in faces(s)]
    if kind == 'card':
        for i in range(0, len(fs), 2):
            for _ in range(copies):
                pg, y = (out.new_page(width=210 * MM, height=297 * MM), 30.0)
                for raw, w, h in fs[i:i + 2]:
                    y += place(pg, raw, w, h, CARD, y) + 20
    elif kind == 'passport':
        for raw, w, h in fs:
            pg = out.new_page(width=210 * MM, height=297 * MM)
            place(pg, raw, w, h, (125, 176) if h > w else (176, 125), 20.0)
    elif kind == 'doc':
        for (raw, w, h), m in zip(fs, per_face(mm, len(fs))):
            if not m:
                pg, s = (out.new_page(width=210 * MM, height=297 * MM), min(BOX[0] / w, BOX[1] / h))
                x = (210 - w * s) / 2
                pg.insert_image(fitz.Rect(x * MM, 10 * MM, (x + w * s) * MM, (10 + h * s) * MM), stream=raw)
                print('  %.1f x %.1f mm: real size unknown (--mm W H), so the page is filled' % (w * s, h * s))
                continue
            real = (max(m), min(m)) if w > h else (min(m), max(m))
            px = max(real[0] / w, real[1] / h)
            s = min(K * px, BOX[0] / w, BOX[1] / h)
            if s >= px - 1e-09:
                if s < K * px - 1e-09:
                    print('  [!] %.0f%% of real size: 117%% does not fit on A4' % (100 * s / px))
                pg = out.new_page(width=210 * MM, height=297 * MM)
                x = (210 - w * s) / 2
                pg.insert_image(fitz.Rect(x * MM, 10 * MM, (x + w * s) * MM, (10 + h * s) * MM), stream=raw)
                print('  %.1f x %.1f mm (after a 94%% fit %.1f x %.1f; real %.1f x %.1f)' % (w * s, h * s, w * s * 0.94, h * s * 0.94, *real))
            elif not larger_than_a4(m):
                pg, s = (out.new_page(width=210 * MM, height=297 * MM), min(210 / w, 297 / h))
                x, y = ((210 - w * s) / 2, (297 - h * s) / 2)
                pg.insert_image(fitz.Rect(x * MM, y * MM, (x + w * s) * MM, (y + h * s) * MM), stream=raw)
                print('  A4-sized: the whole page, %.0f%% of the measured %.0f x %.0f mm (choose kind a4)' % (100 * s / px, *real))
            else:
                big_sheet(out, raw, w, h, px, real, paper)
    else:
        for src in srcs:
            if not src.lower().endswith('.pdf'):
                (raw, w, h), = faces(src)
                page = out.new_page(width=210 * MM, height=297 * MM)
                s = min(210 / w, 297 / h)
                x, y = ((210 - w * s) / 2, (297 - h * s) / 2)
                page.insert_image(fitz.Rect(x * MM, y * MM, (x + w * s) * MM, (y + h * s) * MM), stream=raw)
                continue
            a4_pdf(src, out)

def a4_pdf(src, out):
    d = fitz.open(src)
    for p in d:
        page = out.new_page(width=210 * MM, height=297 * MM)
        if len(p.get_text().strip()) > 50:
            page.show_pdf_page(page.rect, d, p.number)
            continue
        raw, w, h = faces_of_page(d, p)
        s = min(210 / w, 297 / h)
        x, y = ((210 - w * s) / 2, (297 - h * s) / 2)
        page.insert_image(fitz.Rect(x * MM, y * MM, (x + w * s) * MM, (y + h * s) * MM), stream=raw)

def faces_of_page(d, p):
    best = max(p.get_images(), key=lambda im: im[2] * im[3])
    return (d.extract_image(best[0])['image'], best[2], best[3])
REAL = {'card': CARD, 'passport': (125, 176), 'a4': (210, 297)}
QUALITY = (85, 70, 55, 45)

def page_mm(kind, w, h, mm=None):
    real = mm or REAL.get(kind)
    if not real:
        return (210.0, 210.0 * h / w)
    long_mm = float(max(real))
    return (long_mm, long_mm * h / w) if w >= h else (long_mm * w / h, long_mm)

def upload(kind, srcs, mm=None, cap=None):
    fs = [f for s in srcs for f in faces(s)]
    sizes = [page_mm(kind, w, h, m) for (_, w, h), m in zip(fs, per_face(mm, len(fs)))]

    def pdf(raws):
        d = fitz.open()
        for raw, (W, H) in zip(raws, sizes):
            pg = d.new_page(width=W * MM, height=H * MM)
            pg.insert_image(pg.rect, stream=raw)
        d.set_metadata({'title': 'Scan'})
        return d.tobytes(garbage=3, deflate=True)

    def jpeg(q, scale):
        out = []
        for raw, w, h in fs:
            pix = fitz.Pixmap(raw)
            if pix.alpha:
                pix = fitz.Pixmap(pix, 0)
            if scale < 1:
                pix = fitz.Pixmap(pix, max(1, round(w * scale)), max(1, round(h * scale)))
            out.append(pix.tobytes('jpeg', jpg_quality=q))
        return out
    data, q, scale = (pdf([f[0] for f in fs]), None, 1.0)
    if cap:
        for q in QUALITY:
            data = pdf(jpeg(q, 1.0))
            if len(data) <= cap:
                break
        while len(data) > cap and max((max(w, h) for _, w, h in fs)) * scale > 400:
            scale *= min(0.92, max(0.5, 0.97 * (cap / len(data)) ** 0.5))
            data = pdf(jpeg(q, scale))
    info = {'pages': len(fs), 'bytes': len(data), 'kb': round(len(data) / 1024), 'quality': q, 'scale': round(scale, 3), 'fits': not cap or len(data) <= cap, 'sizes': [[round(W, 1), round(H, 1)] for W, H in sizes]}
    return (data, info)
