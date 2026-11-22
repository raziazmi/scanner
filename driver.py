"""Glue between the web page and the scanner: one call per photo, one per hand crop, one for the PDF.
Runs in the browser (Pyodide) and, for the build's self-test, on the laptop with the same files."""
import json
import os
import tempfile

import cv2
import numpy as np
import scan_doc as S

TMP = tempfile.gettempdir()
PAGES = {}
ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _collect():
    lines = []
    return lines, lambda *a: lines.append(" ".join(str(x) for x in a).strip())


def _pts(a):
    return None if a is None else np.asarray(a, float).reshape(-1, 2).round(1).tolist()


def _load(pid, data, w, h, o, log):
    """The photo into PAGES and the automatic page; an error dict when the scanner refuses it.
    The photo stays in memory for a hand crop, also when the automatic one refused it."""
    kind, mm = o["kind"], (tuple(o["mm"]) if o.get("mm") else None)
    raw = data.to_bytes() if hasattr(data, "to_bytes") else bytes(data)
    img = cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    del raw                                  # 27 MB at 3000 px; the wasm heap keeps every peak (19/11)
    p = PAGES[pid] = {"img": img, "kind": kind, "mode": o["mode"], "mm": mm, "card": None, "q": None,
                      "auto": None, "pts": None, "rot": 0, "auto_rot": 0, "more": []}
    if o.get("card_pts"):
        # one of the other cards of a photo of several: its outline came with the first card's scan
        q = p["q"] = np.asarray(o["card_pts"], np.float64)
        log("  card %s of this photo" % o.get("card_no", ""))
    else:
        more = [] if kind == "card" else None
        try:
            q = p["q"] = S.detect(img, kind, log=log, more=more)
        except S.NoDocument as e:
            return {"error": "nodoc", "text": str(e)}
        p["more"] = [_pts(m) for m in more or []]
    if o.get("ref") and kind == "doc" and not mm:
        card = S.find_card(img, q, log=log) if q is not None else None
        if card is None:
            return {"error": "nocard"}
        p["card"] = card
        p["mm"] = tuple(float(x) for x in S.size_from_card(card, S.detect(img, kind, log=lambda *x: None, inset=False), img.shape))
        log("  real size from the card: %.1f x %.1f mm" % p["mm"])
    flat, rects, P = _auto(p, log)
    p["auto"] = _pts(S.six_points(P, rects) if rects else q)
    p["rot"] = p["auto_rot"] = 90 if kind == "card" and flat.shape[0] > flat.shape[1] else 0
    # the warnings judge the outline used, not detect's: a spread's detect quad ran past the frame (4463, 21/11)
    p.update(flat=flat, rects=rects, pts=p["auto"], qa=S.quality(img, flat, p["auto"], kind, p["mm"]))
    if p["auto"] is None:
        # no paper edge found and the whole photo became the page: said, not silent (21/11: a card on
        # a cluttered desk came out as the whole desk with no word); a hand crop clears it
        p["qa"]["flags"].append("whole")
    elif kind == "card" and (S.shape_dev(np.asarray(q, np.float64)[:4], w, h, "card") > 0.05 or S.opposite_ratio(q) > 1.3):
        # an outline that is no card's shape is something else on the desk (MIDV 21/11: a sweet
        # wrapper, a memory stick; 14 of the 16 wrong ones, none of the 96 right ones). A card is
        # small in the photo, so perspective hardly shortens one side against the other (right
        # ones 1.12 at most): MIDV v-005 (22/11) had the right ratio, 1.63, from a memory stick,
        # a cable and table with opposite sides 2.6 apart, and said nothing
        p["qa"]["flags"].append("shape")
    elif kind in ("a4", "doc") and not rects and S.not_a_page(q, w, h):
        # a page outline no flat page gives: a wedge along a cable, a corner cut across the page
        # (SmartDoc held-out 22/11, said nothing; any page size, so no ratio is asked)
        p["qa"]["flags"].append("shape")
    return None


def scan(pid, data, w, h, opts):
    """data: RGBA bytes of the photo, w x h; opts: JSON {kind, mode, mm, ref, card_pts, card_no}.
    Same steps as scan_doc.main for one photo; a portrait card is turned clockwise (the laptop asks
    OCR). A photo of several cards answers with "more": the other cards' outlines, which the page
    scans again each with its card_pts (one page per card)."""
    lines, log = _collect()
    err = _load(pid, data, w, h, json.loads(opts), log)
    return json.dumps(dict(err, log=lines)) if err else _render(pid, lines)


def restore(pid, data, w, h, opts):
    """A page the phone kept (IndexedDB) after iOS closed the app: the photo comes back, the scan
    runs again and the saved crop, turn and colour are put back. opts: JSON {kind, mode, mm, ref}
    and, when the page had a result, its pts (null = the whole photo) and rot."""
    o = json.loads(opts)
    lines, log = _collect()
    err = _load(pid, data, w, h, o, log)
    if err:
        return json.dumps(dict(err, log=lines))
    p = PAGES[pid]
    if "pts" in o:
        _crop(p, _pts(o["pts"]), log)
        p["rot"] = int(o.get("rot", p["rot"])) % 360
    return _render(pid, lines)


def _auto(p, log):
    """The automatic page from the detected outline: two planes when there is a fold, else one."""
    pan = S.two_planes(p["img"], p["q"], p["kind"], p["mm"], log=log)
    return pan if pan else (S.flatten(p["img"], p["q"], p["kind"], p["mm"], log=log), None, None)


def _crop(p, pts, log):
    if pts is not None and pts == p["auto"]:
        flat, rects, _ = _auto(p, log)
        q = p["q"]
    else:
        if pts is not None and p["card"] is not None:     # the reference card measures the new outline
            p["mm"] = tuple(float(x) for x in S.size_from_card(p["card"], S.order(pts[:4]), p["img"].shape))
        flat, rects, _ = S.from_points(p["img"], pts, p["kind"], p["mm"], log=log)
        q = pts
    p.update(flat=flat, rects=rects, pts=pts, qa=S.quality(p["img"], flat, pts, p["kind"], p["mm"]))
    if pts is None and p["auto"] is None:              # still no edge: the warning stays (also after a restore)
        p["qa"]["flags"].append("whole")


def crop(pid, opts):
    """Razi's own crop: opts JSON {pts: 4 or 6 [x, y] (corners, then the fold ends) or null for
    the whole photo, rot, mode}. The automatic points give the automatic page again, exactly."""
    o = json.loads(opts)
    p = PAGES[pid]
    lines, log = _collect()
    _crop(p, _pts(o.get("pts")), log)
    p.update(rot=int(o.get("rot", p["rot"])) % 360, mode=o.get("mode", p["mode"]))
    return _render(pid, lines)


def _render(pid, lines=None):
    p = PAGES[pid]
    turn = lambda im: cv2.rotate(im, ROT[p["rot"]]) if p["rot"] else im
    if p["rects"]:                      # two planes: light per panel first, then the page turns
        res = turn(S.enhance(p["flat"], p["kind"], p["mode"], p["rects"]))
    else:
        res = S.enhance(turn(p["flat"]), p["kind"], p["mode"])
    ext = ".png" if p["mode"] == "bw" else ".jpg"
    ok, buf = cv2.imencode(ext, res, [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == ".jpg" else [])
    if not ok:
        raise RuntimeError("could not encode the page")
    if p.get("path") and os.path.exists(p["path"]):
        os.remove(p["path"])
    p["path"] = os.path.join(TMP, "face_%s%s" % (pid, ext))
    buf.tofile(p["path"])
    quad = None if p["pts"] is None else S.order(p["pts"][:4]).round(1).tolist()
    return json.dumps({"path": p["path"], "w": int(res.shape[1]), "h": int(res.shape[0]), "quad": quad,
                       "pts": p["pts"], "auto": p["auto"], "auto_rot": p["auto_rot"], "planes": 2 if p["rects"] else 1,
                       "mm": p["mm"], "rot": p["rot"], "mean": float(res.mean()), "quality": p["qa"],
                       "more": p.get("more") or [], "log": lines or []})


def rotate(pid):
    PAGES[pid]["rot"] = (PAGES[pid]["rot"] + 90) % 360
    return _render(pid)


def restyle(pid, mode):
    PAGES[pid]["mode"] = mode
    return _render(pid)


def drop(pid):
    p = PAGES.pop(pid, None)
    if p and p.get("path") and os.path.exists(p["path"]):
        os.remove(p["path"])


def pdf(opts):
    """opts: JSON {ids, kind, copies, mm, out, cap, paper}. out "print" (default): fotocopia_a4.build
    lays the pages out on A4, as on the laptop; a doc takes the typed size (mm) or each page its own
    size measured by the reference card; a doc larger than A4 goes on A3 or A4 tiles (paper "a3" or
    "tiles", 21/11). out "upload": fotocopia_a4.upload, one page per scan at its own size, under cap
    bytes when cap is given."""
    import fitz
    import fotocopia_a4 as F
    o = json.loads(opts)
    ids, kind = o["ids"], o["kind"]
    paths = [PAGES[i]["path"] for i in ids]
    if o.get("out") == "upload":
        mms = [PAGES[i]["mm"] or (tuple(o["mm"]) if o.get("mm") else None) for i in ids]
        data, up = F.upload(kind, paths, mm=mms, cap=o.get("cap"))
    else:
        mm = tuple(o["mm"]) if o.get("mm") else ([PAGES[i]["mm"] for i in ids] if kind == "doc" else None)
        doc = fitz.open()
        F.build(kind, paths, int(o.get("copies") or 1), doc, mm=mm, paper=o.get("paper") or "a3")
        doc.set_metadata({"title": "Fotocopia A4"})
        data, up = doc.tobytes(garbage=3, deflate=True), {}
    back = fitz.open(stream=data, filetype="pdf")           # read back what was written
    info = dict(up, pages=len(back), images=sum(len(pg.get_images()) for pg in back), kb=round(len(data) / 1024),
                bytes=len(data), mm=[[round(pg.rect.width / F.MM, 1), round(pg.rect.height / F.MM, 1)] for pg in back])
    back.close()
    info["path"] = os.path.join(TMP, "scan.pdf")
    with open(info["path"], "wb") as fh:
        fh.write(data)
    return json.dumps(info)


def preview(opts):
    """The PDF pdf() wrote last, each page as a JPEG for «دیدن فایل» before it is shared (Razi,
    19/11): drawn from the file itself, so what is seen is what is sent. opts: JSON {width} (px)."""
    import fitz
    import fotocopia_a4 as F
    o = json.loads(opts)
    doc = fitz.open(os.path.join(TMP, "scan.pdf"))
    out = os.path.join(TMP, "scan_preview")
    os.makedirs(out, exist_ok=True)
    pages = []
    for i, pg in enumerate(doc):
        z = float(o.get("width") or 1000) / pg.rect.width
        pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width * pix.n]
        a = a.reshape(pix.height, pix.width, pix.n)
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if pix.n == 3 else a
        ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("could not draw page %d" % (i + 1))
        path = os.path.join(out, "page_%d.jpg" % i)
        buf.tofile(path)
        pages.append({"path": path, "w": int(pix.width), "h": int(pix.height),
                      "mm": [round(pg.rect.width / F.MM, 1), round(pg.rect.height / F.MM, 1)]})
    doc.close()
    keep = {os.path.basename(p["path"]) for p in pages}
    for name in os.listdir(out):                         # pages of an earlier, longer file
        if name not in keep:
            os.remove(os.path.join(out, name))
    return json.dumps({"pages": pages, "paths": [p["path"] for p in pages]})
