import argparse, asyncio, io, os, shutil, subprocess, sys, tempfile
import cv2
import numpy as np
WORK = 1000
RATIO = {'card': 85.6 / 53.98, 'passport': 176 / 125, 'a4': 297 / 210}
HIT = 7.0
CARD_MIN = 0.02
DOC_MIN = 0.04

def order(q):
    q = np.asarray(q, dtype=np.float32).reshape(4, 2)
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    return np.roll(q, -int(np.argmin(q.sum(1))), axis=0)

def wipe_ink(img, k):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.medianBlur(cv2.morphologyEx(img, cv2.MORPH_CLOSE, ker, iterations=2), 5 if k < 15 else 7)

def sides(q):
    c = q.mean(0)
    out = []
    for k in range(4):
        a, b = (q[k], q[(k + 1) % 4])
        d = b - a
        n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-09)
        if np.dot((a + b) / 2 - c, n) < 0:
            n = -n
        out.append((a, b, n))
    return out

def sample(lab, p):
    h, w = lab.shape[:2]
    x, y = (np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int))
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    v = np.zeros((len(p), 3), np.float32)
    v[ok] = lab[y[ok], x[ok]]
    return (v, ok)

def side_support(lab, q, memo=None):
    res = []
    for a, b, n in sides(order(q)):
        key = (a.tobytes(), b.tobytes(), n.tobytes())
        if memo is not None and key in memo:
            res.append(memo[key])
            continue
        res.append(side_share(lab, a, b, n))
        if memo is not None:
            memo[key] = res[-1]
    return res

def side_share(lab, a, b, n, hits=False):
    L = float(np.hypot(*b - a))
    s = np.linspace(0.05, 0.95, max(12, int(L / 4)))
    p = a[None] + s[:, None] * (b - a)[None]
    m = len(p)
    offs = np.array([-2.0, 0.0, 2.0])[:, None, None]
    v, ok = sample(lab, np.concatenate([(p + (offs - 4) * n).reshape(-1, 2), (p + (offs + 4) * n).reshape(-1, 2)]))
    d = (v[:3 * m] - v[3 * m:]).reshape(3, m, 3)
    d[~(ok[:3 * m] & ok[3 * m:]).reshape(3, m)] = 0
    nr = np.linalg.norm(d, axis=2)
    j = np.argmax(nr, axis=0)
    d, best = (d[j, np.arange(m)], nr[j, np.arange(m)])
    strong = best >= HIT
    if strong.sum() < 3:
        return (0.0, np.zeros(m, bool)) if hits else 0.0
    u = np.median(d[strong], axis=0)
    u /= np.linalg.norm(u) + 1e-09
    hit = d @ u >= 0.7 * HIT
    return (float(np.mean(hit)), hit) if hits else float(np.mean(hit))

def table_inside(lab, q, d=8):
    inn, out = ([], [])
    for a, b, n in sides(order(q)):
        p = a[None] + np.linspace(0.1, 0.9, 24)[:, None] * (b - a)[None]
        vi, oi = sample(lab, p - d * n)
        vo, oo = sample(lab, p + d * n)
        inn.append(np.median(vi[oi], 0) if oi.any() else None)
        out.append(np.median(vo[oo], 0) if oo.any() else None)
    for k in range(4):
        rest = [out[j] for j in range(4) if j != k and out[j] is not None]
        if inn[k] is None or out[k] is None or len(rest) < 2:
            continue
        bg = np.median(rest, 0)
        if max((np.linalg.norm(x - bg) for x in rest)) > 0.5 * np.linalg.norm(inn[k] - out[k]):
            continue
        if np.linalg.norm(inn[k] - bg) < 0.35 * np.linalg.norm(inn[k] - out[k]):
            return True
    return False

def over_table(lab, q, d=8, share=0.15):
    h, w = lab.shape[:2]
    outs = []
    for a, b, n in sides(order(q)):
        p = a[None] + np.linspace(0.1, 0.9, 24)[:, None] * (b - a)[None]
        vo, oo = sample(lab, p + d * n)
        if oo.sum() >= 6:
            outs.append(np.median(vo[oo], 0))
    if len(outs) < 3:
        return False
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [order(q).astype(np.int32)], 1)
    k = 4
    px = lab[::k, ::k][cv2.erode(m, np.ones((2 * d + 1, 2 * d + 1), np.uint8))[::k, ::k] > 0]
    if len(px) < 100:
        return False
    paper, table = (np.median(px, 0), np.median(outs, 0))
    contrast = np.linalg.norm(paper - table)
    if contrast < 3 * HIT or sorted((np.linalg.norm(x - table) for x in outs))[-2] > 0.5 * contrast:
        return False
    near = ((px - table) ** 2).sum(1) < 0.25 * ((px - paper) ** 2).sum(1)
    return float(near.mean()) > share

def trim_table(lab, q, d=8, least=0.03):
    h, w = lab.shape[:2]
    q = order(q)
    outs = []
    for a, b, n in sides(q):
        p = a[None] + np.linspace(0.1, 0.9, 24)[:, None] * (b - a)[None]
        vo, oo = sample(lab, p + d * n)
        if oo.sum() >= 6:
            outs.append(np.median(vo[oo], 0))
    if len(outs) < 3:
        return None
    x0, y0, bw, bh = cv2.boundingRect(q.astype(np.int32))
    x0, y0 = (max(0, x0 - 2), max(0, y0 - 2))
    x1, y1 = (min(w, x0 + bw + 4), min(h, y0 + bh + 4))
    lab, o = (lab[y0:y1, x0:x1], np.float32([x0, y0]))
    m = np.zeros(lab.shape[:2], np.uint8)
    cv2.fillPoly(m, [(q - o).astype(np.int32)], 1)
    inner = cv2.erode(m, np.ones((2 * d + 1, 2 * d + 1), np.uint8)) > 0
    if inner.sum() < 100:
        return None
    paper, table = (np.median(lab[inner], 0), np.median(outs, 0))
    contrast = np.linalg.norm(paper - table)
    if contrast < 3 * HIT or sorted((np.linalg.norm(x - table) for x in outs))[-2] > 0.5 * contrast:
        return None
    tab = ((((lab - table) ** 2).sum(2) < 0.25 * ((lab - paper) ** 2).sum(2)) & (m > 0)).astype(np.uint8)
    tab = cv2.morphologyEx(tab, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    border = (m > 0) & (cv2.erode(m, np.ones((5, 5), np.uint8)) == 0)
    n, lbl = cv2.connectedComponents(tab)
    touch = np.bincount(lbl[border & (tab > 0)], minlength=n) >= 10
    touch[0] = False
    R = touch[lbl]
    if R.sum() < least * m.sum():
        return None
    rest = cv2.morphologyEx(((m > 0) & ~R).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(rest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in np.arange(0.01, 0.12, 0.005):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            break
    else:
        return None
    ap = order(ap.reshape(4, 2).astype(np.float32)).astype(np.float64)
    pts = c.reshape(-1, 2).astype(np.float64)
    tol = max(3.0, 0.04 * min(np.linalg.norm(ap[1] - ap[0]), np.linalg.norm(ap[3] - ap[0])))
    lines = [fit_side(pts, ap[k], ap[(k + 1) % 4], tol) for k in range(4)]
    out = [cross(lines[k - 1], lines[k]) for k in range(4)]
    if None in out:
        return None
    nq = order(np.array(out, np.float32) + o)
    return nq if good_quad(nq, w, h, CARD_MIN) else None

def good_quad(q, w, h, min_area=0.08):
    q = order(q)
    if not cv2.isContourConvex(q.astype(np.int32)) or cv2.contourArea(q) < min_area * w * h:
        return False
    for k in range(4):
        v1, v2 = (q[k - 1] - q[k], q[(k + 1) % 4] - q[k])
        if abs(np.dot(v1, v2)) > 0.64 * np.linalg.norm(v1) * np.linalg.norm(v2):
            return False
    m = 0.02 * max(w, h)
    return bool((q[:, 0] > -m).all() and (q[:, 0] < w + m).all() and (q[:, 1] > -m).all() and (q[:, 1] < h + m).all())

def contour_quads(mask, w, h, min_area=0.08):
    out = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < min_area * w * h:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.035, 0.05):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                out.append(ap.reshape(4, 2).astype(np.float32))
                break
    return out

def cross(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-06:
        return None
    a, b = (x1 * y2 - y1 * x2, x3 * y4 - y3 * x4)
    return ((a * (x3 - x4) - (x1 - x2) * b) / den, (a * (y3 - y4) - (y1 - y2) * b) / den)

def merge_lines(ls):
    kept = []
    for ln, x1, y1, x2, y2 in sorted(ls, reverse=True):
        mx, my = ((x1 + x2) / 2, (y1 + y2) / 2)
        if all((abs((k[3] - k[1]) * (k[2] - my) - (k[1] - mx) * (k[4] - k[2])) / (np.hypot(k[3] - k[1], k[4] - k[2]) + 1e-09) >= 8 for k in kept)):
            kept.append((ln, x1, y1, x2, y2))
    return kept

def line_quads(edges, w, h, top=9):
    L = cv2.HoughLinesP(edges, 1, np.pi / 360, 50, minLineLength=0.12 * min(w, h), maxLineGap=15)
    if L is None:
        return []
    hor, ver = ([], [])
    for x1, y1, x2, y2 in L.reshape(-1, 4).astype(float):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        ln = (np.hypot(x2 - x1, y2 - y1), x1, y1, x2, y2)
        if ang < 35 or ang > 145:
            hor.append(ln)
        elif 55 < ang < 125:
            ver.append(ln)
    picked = []
    for group, axis in ((merge_lines(hor), 2), (merge_lines(ver), 1)):
        pos = lambda l: (l[axis] + l[axis + 2]) / 2
        keep = group[:top] + sorted(group, key=pos)[:2] + sorted(group, key=pos)[-2:]
        uniq = []
        for l in keep:
            if l not in uniq:
                uniq.append(l)
        picked.append(sorted(uniq, key=pos))
    hor, ver = picked
    out = []
    for i in range(len(hor)):
        for j in range(i + 1, len(hor)):
            for k in range(len(ver)):
                for m in range(k + 1, len(ver)):
                    t, b, l, r = (hor[i][1:], hor[j][1:], ver[k][1:], ver[m][1:])
                    pts = [cross(t, l), cross(t, r), cross(b, r), cross(b, l)]
                    if None not in pts:
                        out.append(np.array(pts, dtype=np.float32))
    return out

def not_table(lab, ring=0.06, cover=0.97, most=24, cell=6.0):
    h, w = lab.shape[:2]
    sm = cv2.resize(lab, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32) * np.float32([0.5, 1, 1])
    hh, ww = sm.shape[:2]
    r = max(2, int(ring * min(hh, ww)))
    px = np.concatenate([sm[:r].reshape(-1, 3), sm[-r:].reshape(-1, 3), sm[r:-r, :r].reshape(-1, 3), sm[r:-r, -r:].reshape(-1, 3)])
    key = np.floor(px / cell).astype(np.int32)
    key = (key[:, 0] * 64 + key[:, 1]) * 64 + key[:, 2]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep = np.argsort(-cnt, kind='stable')
    n = min(most, int(np.searchsorted(np.cumsum(cnt[keep]), cover * len(px))) + 1)
    inv = inv.ravel()
    proto = np.array([px[inv == j].mean(0) for j in keep[:n]], np.float32)
    d = np.full((hh, ww), np.inf, np.float32)
    t = np.empty_like(sm)
    for p in proto:
        np.subtract(sm, p, out=t)
        np.multiply(t, t, out=t)
        np.minimum(d, t.sum(2), out=d)
    d8 = np.clip(np.sqrt(d) * 4, 0, 255).astype(np.uint8)
    _, th = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.resize(th, (w, h), interpolation=cv2.INTER_NEAREST)

def candidates(small, min_area=0.08, clean=None, table=True, pool=False):
    if pool:
        lab, quads = candidates(small, min_area, clean, table)
        k = 0.85
        quads += [order(q / k) for q in candidates(cv2.resize(small, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), min_area, None, False)[1]]
        return (lab, quads)
    h, w = small.shape[:2]
    clean = wipe_ink(small, 9) if clean is None else clean
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    gray = lab[:, :, 0]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = np.max([cv2.Canny(gray, 15, 45), cv2.Canny(lab[:, :, 1], 6, 18), cv2.Canny(lab[:, :, 2], 6, 18)], 0)
    quads = contour_quads(cv2.dilate(edges, k), w, h, min_area)
    quads += contour_quads(cv2.dilate(cv2.Canny(gray, 40, 120), k), w, h, min_area)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    quads += contour_quads(th, w, h, min_area) + contour_quads(255 - th, w, h, min_area)
    if table:
        quads += contour_quads(not_table(lab), w, h, min_area)
    quads += line_quads(edges, w, h)
    return (lab.astype(np.float32), [order(q) for q in quads if good_quad(q, w, h, min_area)])

class NoDocument(Exception):
    pass

def detect(img, kind=None, log=print, inset=True, more=None):
    s = min(1.0, WORK / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    h, w = small.shape[:2]
    lab, quads = candidates(small, min_area=CARD_MIN if kind == 'card' else DOC_MIN)
    ranked, seen, memo = ([], set(), {})
    for q in quads:
        key = tuple(np.round(q.flatten() / 5).astype(int))
        if key in seen:
            continue
        seen.add(key)
        sup = side_support(lab, q, memo)
        if min(sup) < 0.5 or np.mean(sup) < 0.6:
            continue
        area = cv2.contourArea(q) / (w * h)
        score = 0.5 * area + 0.5 * float(np.mean(sup)) + 0.5 * min(sup) if kind == 'card' else area + 0.25 * float(np.mean(sup)) + 0.25 * min(sup)
        if kind in RATIO:
            dev = shape_dev(q, w, h, kind)
            score -= (1.5 if kind == 'card' else 0.5) * dev
        ranked.append((score, sup, q, dev if kind in RATIO else 1.0))
    ranked.sort(key=lambda r: -r[0])
    ink = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0), 60, 160) > 0
    printed = 0
    cropped = kind in RATIO and abs(np.log(max(w, h) / min(w, h) / RATIO[kind])) <= 0.05
    for score, sup, q, dev in ranked:
        if table_inside(lab, q) or (kind != 'card' and over_table(lab, q)):
            continue
        if print_outside(ink, q) and (not (kind == 'card' and min(sup) >= 0.8 and (dev <= 0.08) and (not cropped))):
            printed += 1
            continue
        m = np.zeros(ink.shape, np.uint8)
        cv2.fillPoly(m, [q.astype(np.int32)], 1)
        m = cv2.erode(m, np.ones((9, 9), np.uint8))
        if ink[m > 0].mean() < 0.005:
            continue
        oq = outer_page(lab, ink, ranked, q, sup) if kind in ('a4', 'doc') else None
        if oq is not None:
            log('  a larger outline around it lies on paper edges: taken')
            q, sup = oq
        log('  outline: %.0f%% of the photo, edge support %s (%d candidates)' % (100 * cv2.contourArea(q) / (w * h), ' '.join(('%.2f' % x for x in sup)), len(seen)))
        tq = trim_table(lab, q) if kind in ('card', 'doc', 'a4') else None
        ts = None if tq is None else side_support(lab, tq, memo)
        if tq is not None and (min(ts) >= 0.5 if kind == 'card' else min(ts) >= 0.9 and np.mean(ts) > np.mean(sup)):
            log('  a strip of table inside the outline: cut back to the paper (%.0f%% of it)' % (100 * cv2.contourArea(tq) / cv2.contourArea(q)))
            q = tq
        mq = mend_side(lab, q, w, h, memo) if kind == 'card' else None
        if mq is not None:
            log("  its one weak side moved onto the card's edge")
            q = mq
        if kind == 'card' and more is not None:
            for o in other_cards(lab, ink, ranked, q, memo):
                more.append(snap(img, o / s, 1 / s, inset))
            if more:
                log('  %d more card(s) in this photo: each its own page' % len(more))
        wq, left = wrap_if_paper_spills(small, q) if kind != 'card' else (None, 0)
        if wq is not None:
            log('  %.0f%% of the paper lies outside it (pages not flat): the outline wraps all of the paper' % (100 * left))
            return wq / s
        return snap(img, q / s, 1 / s, inset)
    if ink.mean() < 0.02:
        raise NoDocument('no document in this photo (print density %.3f, a page has 0.04 or more)' % ink.mean())
    if printed:
        log('  print runs outside every outline (%d): the photo looks already cropped, used whole' % printed)
    else:
        log('  [!] no clear paper edge in %d outlines: using the whole photo (photograph it on a darker, plain background)' % len(seen))
    return None

def outer_page(lab, ink, ranked, q, sup, firm=0.9, more=1.15, weak=0.7, more_weak=1.5):
    A = cv2.contourArea(q)
    lim = (more_weak if min(sup) < weak else more) * A
    best = None
    for score, s2, c, _ in ranked:
        if c is q or min(s2) < firm or np.mean(s2) < np.mean(sup) - 0.02:
            continue
        a2 = cv2.contourArea(c)
        if not A < a2 <= lim or cv2.intersectConvexConvex(np.float32(q), np.float32(c))[0] < 0.98 * A:
            continue
        if table_inside(lab, c) or over_table(lab, c) or print_outside(ink, c):
            continue
        if best is None or score > best[0]:
            best = (score, c, s2)
    return None if best is None else (best[1], best[2])

def mend_side(lab, q, w, h, memo=None, weak=0.7, firm=0.85, reach=0.1):
    q = order(q).astype(np.float64)
    sup = side_support(lab, q, memo)
    low = [k for k in range(4) if sup[k] < weak]
    if len(low) != 1 or min((x for k, x in enumerate(sup) if k != low[0])) < firm:
        return None
    k = low[0]
    a, b = (q[k], q[(k + 1) % 4])
    ua, ub = (q[k] - q[k - 1], q[(k + 1) % 4] - q[(k + 2) % 4])
    ra, rb = (reach * np.linalg.norm(ua), reach * np.linalg.norm(ub))
    ua, ub = (ua / np.linalg.norm(ua), ub / np.linalg.norm(ub))
    c = q.mean(0)

    def share(ta, tb):
        na, nb = (a + ta * ua, b + tb * ub)
        d = nb - na
        n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-09)
        return side_share(lab, na, nb, n if np.dot((na + nb) / 2 - c, n) >= 0 else -n)
    best = (sup[k], 0.0, 0.0)
    for step, ta0, tb0, sa, sb in ((2.0, 0.0, 0.0, ra, rb), (1.0, None, None, 2.0, 2.0)):
        ta0, tb0 = (best[1], best[2]) if ta0 is None else (ta0, tb0)
        for ta in ta0 + np.arange(-sa, sa + 1e-09, step):
            for tb in tb0 + np.arange(-sb, sb + 1e-09, step):
                v = share(ta, tb)
                if v > best[0] + 1e-09 or (abs(v - best[0]) <= 1e-09 and abs(ta) + abs(tb) < abs(best[1]) + abs(best[2])):
                    best = (v, ta, tb)
    if best[0] < firm:
        return None
    m = q.copy()
    m[k], m[(k + 1) % 4] = (a + best[1] * ua, b + best[2] * ub)
    m = order(m)
    if shape_dev(m, w, h, 'card') > shape_dev(q, w, h, 'card') + 0.001 or table_inside(lab, m):
        return None
    return m.astype(np.float32)

def plain_table(lab, ring=0.06, cell=6.0, most=60):
    h, w = lab.shape[:2]
    sm = cv2.resize(lab, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32) * np.float32([0.5, 1, 1])
    hh, ww = sm.shape[:2]
    r = max(2, int(ring * min(hh, ww)))
    px = np.concatenate([sm[:r].reshape(-1, 3), sm[-r:].reshape(-1, 3), sm[r:-r, :r].reshape(-1, 3), sm[r:-r, -r:].reshape(-1, 3)])
    key = np.floor(px / cell).astype(np.int32)
    _, cnt = np.unique((key[:, 0] * 64 + key[:, 1]) * 64 + key[:, 2], return_counts=True)
    return int(np.searchsorted(np.cumsum(np.sort(cnt)[::-1]), 0.8 * len(px))) + 1 <= most

def other_cards(lab, ink, ranked, first, memo):
    if not plain_table(lab):
        return []
    a0 = cv2.contourArea(first)
    got = [order(first).astype(np.float32)]
    for score, sup, q, _ in ranked:
        a = cv2.contourArea(q)
        if not 0.5 * a0 <= a <= 2 * a0 or min(sup) < 0.7 or abs(np.log(quad_ratio(q) / RATIO['card'])) > 0.12:
            continue
        if any((cv2.intersectConvexConvex(q.astype(np.float32), g)[0] > 0.05 * min(a, cv2.contourArea(g)) for g in got)):
            continue
        ang = max((abs(90 - np.degrees(np.arccos(np.clip(np.dot(q[k - 1] - q[k], q[(k + 1) % 4] - q[k]) / (np.linalg.norm(q[k - 1] - q[k]) * np.linalg.norm(q[(k + 1) % 4] - q[k])), -1, 1)))) for k in range(4)))
        if ang > 12 or table_inside(lab, q):
            continue
        m = np.zeros(ink.shape, np.uint8)
        cv2.fillPoly(m, [q.astype(np.int32)], 1)
        if ink[cv2.erode(m, np.ones((9, 9), np.uint8)) > 0].mean() < 0.005:
            continue
        tq = trim_table(lab, q)
        if tq is not None and min(side_support(lab, tq, memo)) >= 0.5:
            q = tq
        got.append(order(q).astype(np.float32))
    rest = got[1:]
    rest.sort(key=lambda g: (round(float(g[:, 1].mean()) / (0.5 * np.sqrt(a0))), float(g[:, 0].mean())))
    return rest

def find_card(img, doc, log=print):
    s = min(1.0, WORK / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    lab, quads = candidates(small, min_area=0.01, pool=True)
    dq = order(doc * s).astype(np.float32)
    h, w = small.shape[:2]
    r, _ = true_ratio(dq, w, h)
    M = cv2.getPerspectiveTransform(dq, np.float32([[0, 0], [r, 0], [r, 1], [0, 1]]))

    def on_plane(q):
        c = cv2.perspectiveTransform(order(q).reshape(1, 4, 2).astype(np.float32), M)[0].astype(np.float64)
        ang = max((abs(90 - np.degrees(np.arccos(np.clip(np.dot(c[k - 1] - c[k], c[(k + 1) % 4] - c[k]) / (np.linalg.norm(c[k - 1] - c[k]) * np.linalg.norm(c[(k + 1) % 4] - c[k])), -1, 1)))) for k in range(4)))
        return abs(np.log(quad_ratio(c) / RATIO['card'])) <= 0.07 and ang <= 7
    best, best_a, memo = (None, 0, {})
    for q in quads:
        if abs(np.log(quad_ratio(q) / RATIO['card'])) > 0.12:
            continue
        a = cv2.contourArea(q)
        inter, _ = cv2.intersectConvexConvex(q.astype(np.float32), dq)
        if a <= best_a or inter > 0.02 * a or min(side_support(lab, q, memo)) < 0.6:
            continue
        if not on_plane(q):
            q = trim_table(lab, q)
            if q is None or min(side_support(lab, q, memo)) < 0.6 or (not on_plane(q)):
                continue
            a = cv2.contourArea(q)
            if a <= best_a:
                continue
        best, best_a = (q, a)
    if best is None:
        return None
    log('  reference card: %.1f%% of the photo' % (100 * best_a / (small.shape[0] * small.shape[1])))
    return snap(img, best / s, 1 / s, inset=False)

def size_from_card(card, doc, shape):
    d = order(doc).astype(np.float64)
    r, _ = true_ratio(d, shape[1], shape[0])
    M = cv2.getPerspectiveTransform(d.astype(np.float32), np.float32([[0, 0], [r, 0], [r, 1], [0, 1]]))
    c = cv2.perspectiveTransform(order(card).reshape(1, 4, 2).astype(np.float32), M)[0]
    a = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    b = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    k = (85.6 / max(a, b) + 53.98 / min(a, b)) / 2
    return (float(r * k), float(k))

def paper_blob(small, q, clean=None):
    h, w = small.shape[:2]
    g = cv2.cvtColor(wipe_ink(small, 9) if clean is None else clean, cv2.COLOR_BGR2GRAY)
    level, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    n, lbl, _, _ = cv2.connectedComponentsWithStats(th)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [q.astype(np.int32)], 1)
    over = np.bincount(lbl[m > 0], minlength=n)
    over[0] = 0
    if n < 2 or over.max() == 0:
        return (None, level, m, g)
    return (lbl == int(np.argmax(over)), level, m, g)

def frame_share(comp):
    return float(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]).mean())

def wrap_if_paper_spills(small, q):
    h, w = small.shape[:2]
    comp, _, m, _ = paper_blob(small, q)
    if comp is None:
        return (None, 0)
    border = frame_share(comp)
    d = 2 * int(0.015 * max(h, w)) + 1
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))) > 0
    left = float((comp & ~near).sum() / comp.sum())
    if border >= 0.05 or left <= 0.03:
        return (None, left)
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in np.arange(0.01, 0.12, 0.005):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            break
    else:
        return (None, left)
    pts = hull.reshape(-1, 2).astype(np.float64)
    lines = []
    for a, b, nrm in sides(order(ap)):
        shift = max(0.0, float(np.max((pts - a) @ nrm)))
        lines.append(tuple(a + shift * nrm) + tuple(b + shift * nrm))
    out = [cross(lines[k - 1], lines[k]) for k in range(4)]
    if None in out:
        return (None, left)
    wq = order(np.array(out, dtype=np.float32))
    return (wq if good_quad(wq, w, h) else None, left)

def print_outside(ink, q):
    h, w = ink.shape
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [q.astype(np.int32)], 1)
    d = 2 * int(0.03 * max(h, w)) + 1
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
    inside, outside = (cv2.erode(m, k) > 0, cv2.dilate(m, k) == 0)
    if outside.mean() < 0.03 or not inside.any():
        return False
    out, inn = (ink[outside].mean(), ink[inside].mean())
    return out > max(0.02, 0.3 * inn)

def inset_px(shape):
    return max(2.0, 0.0015 * max(shape[:2]))

def snap(img, q, px, inset=True):
    k = max(9, int(round(9 * px)) | 1)
    r = int(np.ceil(3 * px)) + 2
    reach = 4 * (k // 2) + 3 + 2
    hh, ww = img.shape[:2]
    inset = inset_px(img.shape) if inset else 0.0
    lines = []
    for a, b, n in sides(order(q)):
        s = np.linspace(0.06, 0.94, 80)
        p = a[None] + s[:, None] * (b - a)[None]
        offs = np.arange(-r, r + 1)
        ends = np.concatenate([p[[0, -1]] + o * n for o in (-r, r)])
        x0, y0 = [max(0, int(np.floor(v)) - reach) for v in ends.min(0)]
        x1, y1 = (min(ww, int(np.ceil(ends[:, 0].max())) + reach + 1), min(hh, int(np.ceil(ends[:, 1].max())) + reach + 1))
        if x1 <= x0 or y1 <= y0:
            lines.append((tuple(a - inset * n), tuple(b - inset * n)))
            continue
        clean = cv2.cvtColor(wipe_ink(img[y0:y1, x0:x1], k), cv2.COLOR_BGR2LAB).astype(np.float32)
        o0 = np.array([x0, y0], np.float64)
        prof = np.stack([sample(clean, p + o * n - o0)[0] for o in offs])
        okm = np.stack([sample(clean, p + o * n - o0)[1] for o in offs])
        step = np.linalg.norm(prof[2:] - prof[:-2], axis=2)
        step[~(okm[2:] & okm[:-2])] = 0
        j = np.argmax(step, axis=0)
        strong = step[j, np.arange(len(p))] >= HIT
        pts = p + (offs[1:-1][j] - inset)[:, None] * n[None]
        if strong.sum() >= 12:
            vx, vy, x0, y0 = cv2.fitLine(pts[strong].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            lines.append(((x0, y0), (x0 + vx, y0 + vy)))
        else:
            lines.append((tuple(a - inset * n), tuple(b - inset * n)))
    out = []
    for k in range(4):
        (x1, y1), (x2, y2) = lines[k - 1]
        (x3, y3), (x4, y4) = lines[k]
        p = cross((x1, y1, x2, y2), (x3, y3, x4, y4))
        out.append(p if p is not None else order(q)[k])
    return order(np.array(out, dtype=np.float32))

def quad_ratio(q):
    q = order(q)
    wd = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    ht = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return max(wd, ht) / max(min(wd, ht), 1e-06)

def true_ratio(q, w, h):
    tl, tr, br, bl = order(q)
    wd = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    ht = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    measured = wd / ht
    m1, m2, m3, m4 = [np.array([p[0], p[1], 1.0]) for p in (tl, tr, bl, br)]
    with np.errstate(all='ignore'):
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
        n2, n3 = (k2 * m2 - m1, k3 * m3 - m1)
        u0, v0 = (w / 2, h / 2)
        f2 = -(n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0 + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / (n2[2] * n3[2])
        f = float(np.sqrt(f2)) if f2 > 0 else 0.0
        if not 0.5 * max(w, h) < f < 2 * max(w, h):
            f = 0.8 * max(w, h)
        Ai = np.linalg.inv(np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]))
        r = float(np.sqrt(n2 @ Ai.T @ Ai @ n2 / (n3 @ Ai.T @ Ai @ n3)))
    return (r if np.isfinite(r) and 0.5 < r / measured < 2 else measured, f)

def shape_dev(q, w, h, kind):
    r = true_ratio(q, w, h)[0]
    return min(abs(np.log(max(r, 1 / r) / RATIO[kind])), abs(np.log(quad_ratio(q) / RATIO[kind])))

def flatten(img, q, kind, mm=None, log=print):
    h0, w0 = img.shape[:2]
    if q is None:
        return img.copy()
    tl, tr, br, bl = order(q)
    wd = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    ht = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    meas = max(wd, ht) / min(wd, ht)
    if mm:
        r = max(mm) / min(mm)
    elif kind in RATIO:
        r = RATIO[kind]
    else:
        est, f = true_ratio(q, w0, h0)
        r = max(est, 1 / est)
        log('  shape from perspective %.3f (sides measured %.3f%s)' % (r, meas, ', f=%.0f px' % f if f else ', frontal'))
    if (kind in RATIO or mm) and abs(np.log(meas / r)) > 0.12:
        log('  [!] outline ratio %.2f is far from the real %.2f: check the crop or --kind' % (meas, r))
    long_px = max(wd, ht)
    W, H = (long_px, long_px / r) if wd >= ht else (long_px / r, long_px)
    W, H = (int(round(W)), int(round(H)))
    M = cv2.getPerspectiveTransform(order(q), np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
PAGE = 88 / 125
STRIP = 256

def fill_blob(comp):
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(comp.shape, np.uint8)
    cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    return out

def march(blob, p, u):
    h, w = blob.shape
    for k in range(1, 3 * max(h, w)):
        x, y = (int(round(p[0] + k * u[0])), int(round(p[1] + k * u[1])))
        if not (0 <= x < w and 0 <= y < h) or not blob[y, x]:
            return p + (k - 1) * u
    return p

def crease_line(blob, g, p, u0, short, off=0.04, turn=12.0):
    inner = cv2.erode(blob, np.ones((15, 15), np.uint8)) > 0
    L = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    h, w = L.shape
    ln = np.array([-u0[1], u0[0]])
    offs = np.linspace(-off, off, 2 * int(round(off / 0.005)) + 1) * short
    angs = np.radians(np.linspace(-turn, turn, 2 * int(round(turn / 0.5)) + 1))
    lam = np.arange(0.04, 0.96, 0.01) * short / np.cos(np.radians(turn))
    u = np.stack([np.cos(angs) * u0[0] - np.sin(angs) * u0[1], np.sin(angs) * u0[0] + np.cos(angs) * u0[1]], -1)
    n = np.stack([-u[:, 1], u[:, 0]], -1)
    base = p[None, None, :] + offs[:, None, None] * ln[None, None, :]
    pts = base[:, :, None, :] + lam[None, None, :, None] * u[None, :, None, :]

    def at(img, q):
        x = np.clip(np.round(q[..., 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(q[..., 1]).astype(int), 0, h - 1)
        return img[y, x]
    ok = at(inner, pts)
    r = np.zeros(ok.shape, np.float32)
    for d in (3, 6):
        dn = d * n[None, :, None, :]
        r = np.maximum(r, np.abs(at(L, pts + dn) - at(L, pts - dn)))
        ok &= at(inner, pts + dn) & at(inner, pts - dn)
    cnt = ok.sum(2)
    score = np.where(cnt > 0.5 * len(lam), (np.minimum(r, 40) * ok).sum(2) / np.maximum(cnt, 1), 0)
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    return (float(score[i, j]), base[i, 0], u[j], float(score[len(offs) // 2, len(angs) // 2]))

def notch_vertex(c, p, a, b):
    v = b - a
    L = float(np.linalg.norm(v))
    u = v / L
    t = (c - a) @ u / L
    d = np.abs((c - a) @ np.array([-u[1], u[0]]))
    tp = float((p - a) @ u / L)
    ok = d < 0.15 * L
    segs = [c[ok & (t > tp - 0.22) & (t < tp - 0.04)], c[ok & (t > tp + 0.04) & (t < tp + 0.22)]]
    if min((len(x) for x in segs)) < 8:
        return p
    l1, l2 = [cv2.fitLine(x.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten() for x in segs]
    x = cross((l1[2], l1[3], l1[2] + l1[0], l1[3] + l1[1]), (l2[2], l2[3], l2[2] + l2[0], l2[3] + l2[1]))
    if x is None or np.linalg.norm(np.array(x) - p) > 0.04 * L:
        return p
    return np.array(x, np.float64)

def find_fold(blob, g):
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hi = np.sort(cv2.convexHull(c, returnPoints=False).flatten())
    hull = c[hi]
    peri = cv2.arcLength(hull, True)
    for eps in np.arange(0.01, 0.12, 0.005):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            break
    else:
        return None
    tl, tr, br, bl = order(ap).astype(np.float64)
    nrm = np.linalg.norm
    across = nrm(bl - tl) + nrm(br - tr) >= nrm(tr - tl) + nrm(br - bl)
    A, B = ((tl, bl), (tr, br)) if across else ((tl, tr), (bl, br))
    short = (nrm(tr - tl) + nrm(br - bl)) / 2 if across else (nrm(bl - tl) + nrm(br - tr)) / 2
    try:
        dfx = cv2.convexityDefects(c, hi.reshape(-1, 1).astype(np.int32))
    except cv2.error:
        dfx = None
    ends, depth, dents = ([None, None], [0.0, 0.0], [[], []])
    for s0, e0, f, d in dfx.reshape(-1, 4) if dfx is not None else []:
        p, d = (c[f, 0].astype(np.float64), d / 256.0)
        if d < 0.012 * short:
            continue
        for k, (a, b) in enumerate((A, B)):
            v = b - a
            t = float(np.dot(p - a, v) / np.dot(v, v))
            if 0.25 <= t <= 0.75 and abs(v[0] * (p - a)[1] - v[1] * (p - a)[0]) / nrm(v) < 0.15 * short:
                dents[k].append((s0, e0))
                if d > depth[k]:
                    ends[k], depth[k] = (p, d)
    mid = (tl + tr + br + bl) / 4
    for k, (a, b) in enumerate((A, B)):
        v = b - a
        out = lambda q: -(v[0] * (q - a)[1] - v[1] * (q - a)[0]) * np.sign(v[0] * (mid - a)[1] - v[1] * (mid - a)[0]) / nrm(v)
        for s1, e1 in dents[k]:
            for s2, e2 in dents[k]:
                x, y = (c[e1, 0].astype(np.float64), c[s2, 0].astype(np.float64))
                if (s1, e1) != (s2, e2) and nrm(x - y) <= 0.1 * short and (min(out(x), out(y)) > 0.012 * short):
                    ends[k] = None
    if ends[0] is None and ends[1] is None:
        return None
    u0 = tr - tl + br - bl if across else bl - tl + br - tr
    u0 = u0 / nrm(u0)
    c2 = c.reshape(-1, 2).astype(np.float64)
    if ends[0] is not None and ends[1] is not None:
        fa, fb = (notch_vertex(c2, ends[0], *A), notch_vertex(c2, ends[1], *B))
        u = (fb - fa) / nrm(fb - fa)
        if np.dot(u, u0) < np.cos(np.radians(25)):
            return None
        score, p0, u1, here = crease_line(blob, g, fa, u, short, off=0.03, turn=3.0)
        if score >= 4 and score >= 1.3 * here:
            mid = p0 + 0.5 * short * u1
            fa, fb = (march(blob, mid, -u1), march(blob, mid, u1))
    else:
        k = 0 if ends[0] is not None else 1
        sign = 1 if k == 0 else -1
        score, p0, u, _ = crease_line(blob, g, ends[k], sign * u0, short)
        if score < 4:
            return None
        mid = p0 + 0.5 * short * u
        e1, e2 = (march(blob, mid, -u), march(blob, mid, u))
        fa, fb = (e1, e2) if k == 0 else (e2, e1)
    for p, (a, b) in ((fa, A), (fb, B)):
        v = b - a
        if not 0.2 <= np.dot(p - a, v) / np.dot(v, v) <= 0.8:
            return None
    deg = [np.degrees(np.arctan2(v[1], v[0])) for v in ((tr - tl, br - bl) if across else (bl - tl, br - tr)) + (fb - fa,)]
    if not min(deg[:2]) - 6 <= deg[2] <= max(deg[:2]) + 6:
        return None
    return (tl, tr, br, bl, fa, fb, across)

def fold_at_full_res(gf, mf, fa, fb):
    L = float(np.linalg.norm(fb - fa))
    u = (fb - fa) / L
    n = np.array([-u[1], u[0]])
    R, d = (max(6, int(0.015 * L)), 3)
    offs = np.arange(-R - d, R + d + 1, dtype=np.float32)
    ts = np.linspace(0.08, 0.92, 60)
    P = fa[None] + (ts * L)[:, None] * u[None]
    X = (P[:, 0:1] + offs[None] * n[0]).astype(np.float32)
    Y = (P[:, 1:2] + offs[None] * n[1]).astype(np.float32)
    v = cv2.remap(gf.astype(np.float32), X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    step = np.abs(v[:, 2 * d:] - v[:, :-2 * d])
    j = np.argmax(step, axis=1)
    strong = step[np.arange(len(ts)), j] >= 6
    pts = P + offs[d:-d][j][:, None] * n[None]
    pts = pts[strong]
    if len(pts) < 20:
        return (fa, fb)
    for _ in range(2):
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        r = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        if (r < 2.5).sum() < 20:
            return (fa, fb)
        pts = pts[r < 2.5]
    w = np.array([vx, vy]) * (1 if vx * u[0] + vy * u[1] > 0 else -1)
    if np.dot(w, u) < np.cos(np.radians(3)):
        return (fa, fb)
    mid = np.array([x0, y0]) + ((fa + fb) / 2 - [x0, y0]) @ w * w
    return (march(mf, mid, -w), march(mf, mid, w))

def fit_side(pts, a, b, tol):
    v = b - a
    L2 = float(np.dot(v, v))
    t = (pts - a) @ v / L2
    d = np.abs(v[0] * (pts - a)[:, 1] - v[1] * (pts - a)[:, 0]) / np.sqrt(L2)
    sel = pts[(t > 0.12) & (t < 0.88) & (d < tol)]
    if len(sel) < 10:
        return tuple(a) + tuple(b)
    vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return (x0, y0, x0 + vx, y0 + vy)

def side_offsets(pts, rect, side, n=160, reach=0.08, shift=None):
    x0, y0, x1, y1 = rect
    if side in (0, 2):
        a, b, along, acr, edge = (x0, x1, pts[:, 0], pts[:, 1], y0 if side == 0 else y1)
        mg = reach * (y1 - y0)
    else:
        a, b, along, acr, edge = (y0, y1, pts[:, 1], pts[:, 0], x1 if side == 1 else x0)
        mg = reach * (x1 - x0)
    sel = (along > a) & (along < b) & (np.abs(acr - edge) < mg)
    idx = ((along[sel] - a) / (b - a) * n).astype(int).clip(0, n - 1)
    vals = acr[sel] - edge
    off = np.full(n, np.nan)
    for i in range(int(0.04 * n), n - int(0.04 * n)):
        v = vals[idx == i]
        if len(v):
            off[i] = np.median(v)
    good = np.isfinite(off)
    if good.mean() < 0.4:
        return np.zeros(n)
    off = smooth_edge(off, good, n)
    if off is None:
        return np.zeros(n)
    if side in (1, 2):
        off = -off
    if shift is not None and off.max() < -shift * mg / reach:
        return np.zeros(n)
    return np.clip(off, -mg, mg)

def smooth_edge(off, good, n, jump=4.0, run=8):
    idx = np.flatnonzero(good)
    y = off[good].astype(np.float64)
    cut = np.flatnonzero((np.abs(np.diff(y)) > jump) | (np.diff(idx) > 3)) + 1
    runs = [(a, b) for a, b in zip(np.r_[0, cut], np.r_[cut, len(y)]) if b - a >= run and abs(y[b - 1] - y[a]) <= 0.5 * (idx[b - 1] - idx[a])]
    if not runs:
        return None
    m = max(range(len(runs)), key=lambda k: runs[k][1] - runs[k][0])
    keep = np.zeros(len(y), bool)
    keep[runs[m][0]:runs[m][1]] = True
    for order_ in (range(m + 1, len(runs)), range(m - 1, -1, -1)):
        last = runs[m][1] - 1 if order_.step > 0 else runs[m][0]
        for k in order_:
            a, b = runs[k]
            first = a if order_.step > 0 else b - 1
            if abs(y[first] - y[last]) <= jump + 0.1 * abs(idx[first] - idx[last]):
                keep[a:b] = True
                last = b - 1 if order_.step > 0 else a
    if keep.sum() < 0.4 * n:
        return None
    off = np.interp(np.arange(n), idx[keep], y[keep])
    off = cv2.medianBlur(off.astype(np.float32).reshape(1, -1), 5).flatten()
    return cv2.GaussianBlur(off.reshape(1, -1), (0, 0), 2).flatten()

def cut_to_outline(comp, m, clean, q, edge=0.8):
    d = 2 * int(0.025 * max(m.shape)) + 1
    keep = cv2.dilate(m, np.ones((d, d), np.uint8)) > 0
    sup = side_support(cv2.cvtColor(clean, cv2.COLOR_BGR2LAB).astype(np.float32), q)
    strong = [sp >= edge for sp in sup]
    wide = comp & keep
    if any(strong):
        Y, X = np.mgrid[0:m.shape[0], 0:m.shape[1]]
        e = 0.005 * max(m.shape)
        for (a, b, n), k in zip(sides(order(q)), strong):
            if k:
                keep &= (X - a[0]) * n[0] + (Y - a[1]) * n[1] <= e
    return (wide, comp & keep, strong)

def two_planes(img, q, kind, mm=None, log=print):
    if q is None or kind not in ('passport', 'a4', 'doc'):
        return None
    h0, w0 = img.shape[:2]
    s = min(1.0, WORK / max(h0, w0))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    clean = wipe_ink(small, 9)
    comp, level, m, g = paper_blob(small, np.asarray(q, np.float32) * s, clean)
    edges = []
    joined = comp is not None and frame_share(comp) >= 0.05
    merged = comp is not None and frame_share(comp) > 0 and (comp.sum() > 1.2 * m.sum()) and (kind != 'passport')
    tight = None
    if comp is not None and frame_share(comp) > 0:
        comp, tight, strong = cut_to_outline(comp, m, clean, order(np.float32(q) * s))
        edges = [(a, b) for (a, b, _), k in zip(sides(order(np.asarray(q, np.float64))), strong) if k]
    f = None if comp is None or merged or frame_share(comp) >= 0.05 else find_fold(fill_blob(comp), g)
    if kind == 'passport' and (f is None or joined):
        r = page_pair(img, small, s, kind, mm, log, fold=None if f is None else (f[4], f[5], f[6]), clean=clean)
        if r is None and f is not None:
            r = page_pair(img, small, s, kind, mm, log, clean=clean)
        if r is not None or f is None:
            return r
    if f is None and kind == 'doc' and (comp is not None) and (not merged):
        return page_pair(img, small, s, kind, mm, log, clean=clean, booklet=(order(np.float32(q) * s), comp))
    if f is None:
        return None
    across = f[6]
    gf, mf, cf = paper_full(img, comp if tight is None else tight, level, s)
    tl, tr, br, bl, fa, fb = [np.asarray(p, np.float64) / s for p in f[:6]]
    fa, fb = fold_at_full_res(gf, mf, fa, fb)
    fa, fb = [cf[np.argmin(np.linalg.norm(cf - p, axis=1))] for p in (fa, fb)]
    P, fold_side = panel_quads(tl, tr, br, bl, fa, fb, across)
    tol = 0.03 * max(h0, w0)
    for pk, fs in zip(P, fold_side):
        lines = [fit_side(cf, pk[j], pk[(j + 1) % 4], tol) for j in range(4)]
        for j in range(4):
            pa, pb = (np.asarray(pk[j], np.float64), np.asarray(pk[(j + 1) % 4], np.float64))
            for a, b in edges:
                u = (b - a) / np.linalg.norm(b - a)
                if abs(u[0] * (pa - a)[1] - u[1] * (pa - a)[0]) < tol and abs(u[0] * (pb - a)[1] - u[1] * (pb - a)[0]) < tol and (abs(u @ (pb - pa)) > np.cos(np.radians(5)) * np.linalg.norm(pb - pa)):
                    lines[j] = tuple(a) + tuple(b)
        for i in range(4):
            if (i - 1) % 4 != fs and i != fs:
                c = cross(lines[i - 1], lines[i])
                if c is not None:
                    pk[i][:] = c
    return warp_panels(img, P, across, kind, mm, mf, cf, log=log)

def spine_between(fa, fb, far, slack=1.5):
    a = [np.degrees(np.arctan2(v[1], v[0])) for v in (far[0], far[1], np.asarray(fb) - np.asarray(fa))]
    b, p = (a[0] + (a[k] - a[0] + 180) % 360 - 180 for k in (1, 2))
    return min(a[0], b) - slack <= p <= max(a[0], b) + slack

def page_pair(img, small, s, kind, mm, log=print, fold=None, clean=None, booklet=None):
    h, w = small.shape[:2]
    if booklet is not None:
        bq = order(np.float32(booklet[0]))
        barea = cv2.contourArea(bq)
        paper = booklet[1]
        npaper = np.count_nonzero(paper)
    lab, quads = candidates(small, min_area=0.03 if booklet is None else max(0.03, 0.24 * barea / (w * h)), clean=clean, table=False)
    best, tol = (None, 0.025 * max(h, w))
    if fold is not None:
        fold = (np.asarray(fold[0], np.float64), np.asarray(fold[1], np.float64), bool(fold[2]))
        fv = (fold[1] - fold[0]) / (np.linalg.norm(fold[1] - fold[0]) + 1e-09)
        off = lambda p: abs(fv[0] * (p[1] - fold[0][1]) - fv[1] * (p[0] - fold[0][0]))
        ends = ((3, 2), (0, 1)) if fold[2] else ((1, 2), (0, 3))
    good, seen, memo = ([], set(), {})
    for q in quads:
        key = tuple(np.round(q.flatten() / 5).astype(int))
        if key in seen:
            continue
        seen.add(key)
        if fold is not None and (not any((off(q[i]) <= tol and off(q[j]) <= tol for i, j in ends))):
            continue
        if booklet is not None and (cv2.contourArea(q) < 0.25 * barea or min((cv2.pointPolygonTest(bq.reshape(-1, 1, 2), (float(x), float(y)), True) for x, y in q)) < -0.04 * max(h, w)):
            continue
        sup = side_support(lab, q, memo)
        if sorted(sup)[1] >= 0.5 and np.mean(sup) >= 0.6 and (not table_inside(lab, q)):
            good.append((q, sup))
    for i, (A, sa) in enumerate(good):
        for B, sb in good[i + 1:]:
            for (U, su), (L, sl) in (((A, sa), (B, sb)), ((B, sb), (A, sa))):
                for across in (True, False):
                    e1, e2, ks = ((U[3], U[2]), (L[0], L[1]), (2, 0)) if across else ((U[1], U[2]), (L[0], L[3]), (1, 3))
                    if np.linalg.norm(e1[0] - e2[0]) > tol or np.linalg.norm(e1[1] - e2[1]) > tol:
                        continue
                    if fold is not None and (across != fold[2] or max((abs(fv[0] * (p[1] - fold[0][1]) - fv[1] * (p[0] - fold[0][0])) for p in (*e1, *e2))) > tol):
                        continue
                    fa, fb = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
                    if not spine_between(fa, fb, (U[1] - U[0], L[2] - L[3]) if across else (U[3] - U[0], L[2] - L[1])):
                        continue
                    if booklet is not None:
                        nb = ((U[0], L[3]), (U[1], L[2])) if across else ((U[0], L[1]), (U[3], L[2]))
                        kink = [abs((b - a)[0] * (p - a)[1] - (b - a)[1] * (p - a)[0]) / np.linalg.norm(b - a) for p, (a, b) in zip((fa, fb), nb)]
                        if max(kink) < 0.012 * max(h, w):
                            continue
                    outer = [x for k, x in enumerate(su) if k != ks[0]] + [x for k, x in enumerate(sl) if k != ks[1]]
                    au, al = (cv2.contourArea(U), cv2.contourArea(L))
                    inter, _ = cv2.intersectConvexConvex(U.astype(np.float32), L.astype(np.float32))
                    if min(outer) < 0.5 or inter > 0.05 * min(au, al):
                        continue
                    score = (au + al) / (w * h) + 0.25 * np.mean(outer) + 0.25 * min(outer) - 0.3 * abs(np.log(au / al))
                    if booklet is not None:
                        two = np.zeros((h, w), np.uint8)
                        cv2.fillPoly(two, [np.int32([U[0], U[1], fb, L[2], L[3], fa] if across else [U[0], fa, L[1], L[2], fb, U[3]])], 1)
                        inter = np.count_nonzero(two[paper])
                        score = inter / (npaper + np.count_nonzero(two) - inter)
                    if best is None or score > best[0]:
                        best = (score, U, L, across)
    if best is None:
        return None
    _, U, L, across = best
    U, L = (snap(img, U / s, 1 / s), snap(img, L / s, 1 / s))
    if across:
        fa, fb = ((U[3] + L[0]) / 2, (U[2] + L[1]) / 2)
        P = [[U[0], U[1], fb, fa], [fa, fb, L[2], L[3]]]
    else:
        fa, fb = ((U[1] + L[0]) / 2, (U[2] + L[3]) / 2)
        P = [[U[0], fa, fb, U[3]], [fa, L[1], L[2], fb]]
    hexa = np.float32([P[0][0], P[0][1], P[0][2], P[1][2], P[1][3], P[0][3]] if across else [P[0][0], P[0][1], P[1][1], P[1][2], P[1][3], P[0][3]]) * s
    if booklet is not None:
        one, two = (np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8))
        cv2.fillPoly(one, [bq.astype(np.int32)], 1)
        cv2.fillPoly(two, [hexa.astype(np.int32)], 1)
        one, two = (one > 0, two > 0)
        iou = lambda a, b: (a & b).sum() / max(1, (a | b).sum())
        if iou(two, one) < 0.9 or iou(two, paper) - iou(one, paper) < 0.015:
            return None
    comp, level, _, _ = paper_blob(small, hexa, clean)
    mf = cf = None
    if comp is not None:
        near = np.zeros((h, w), np.uint8)
        cv2.fillPoly(near, [hexa.astype(np.int32)], 1)
        d = 2 * int(0.02 * max(h, w)) + 1
        comp = comp & (cv2.dilate(near, np.ones((d, d), np.uint8)) > 0)
        _, mf, cf = paper_full(img, comp, level, s)
    log('  the pages as two outlines meeting at the fold (they lie at different heights: no one outline fits)' if booklet is not None else '  the pages as two outlines meeting at the spine (the paper mask ran into the table)')
    return warp_panels(img, [np.float32(p) for p in P], across, kind, mm, mf, cf, log=log, shift=0.03)

def paper_full(img, comp, level, s):
    h0, w0 = img.shape[:2]
    k = max(9, int(round(9 / s)) | 1)
    r = 2 * int(np.ceil(4 / s)) + 1
    near = cv2.dilate(cv2.resize(fill_blob(comp), (w0, h0), interpolation=cv2.INTER_NEAREST), np.ones((r, r), np.uint8))
    x, y, bw, bh = cv2.boundingRect(near)
    m = 4 * (k // 2) + 5
    x0, y0, x1, y1 = (max(0, x - m), max(0, y - m), min(w0, x + bw + m), min(h0, y + bh + m))
    gf = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gf[y0:y1, x0:x1] = cv2.cvtColor(wipe_ink(img[y0:y1, x0:x1], k), cv2.COLOR_BGR2GRAY)
    mf = ((gf > level) & (near > 0)).astype(np.uint8)
    mf = fill_blob(cv2.morphologyEx(mf, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))))
    cnts, _ = cv2.findContours(mf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return (gf, mf, max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64))

def panel_quads(tl, tr, br, bl, fa, fb, across):
    if across:
        return ([[tl, tr, fb, fa], [fa, fb, br, bl]], [2, 0])
    return ([[tl, fa, fb, bl], [fa, tr, br, fb]], [1, 3])

def six_points(P, rects):
    a, b = P
    if rects[0][3] == rects[1][1]:
        return [a[0], a[1], b[2], b[3], a[3], a[2]]
    return [a[0], b[1], b[2], a[3], a[1], a[2]]

def fold_ends(tl, tr, br, bl, a, b):
    wd, ht, u = (tr - tl + (br - bl), bl - tl + (br - tr), b - a)
    across = abs(u @ wd) / np.linalg.norm(wd) >= abs(u @ ht) / np.linalg.norm(ht)
    return (a, b, across) if u @ (wd if across else ht) >= 0 else (b, a, across)

def from_points(img, pts, kind, mm=None, log=print):
    if pts is None:
        return (img.copy(), None, None)
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(p) == 4:
        q = order(p)
        return (flatten(img, q, kind, mm, log=log), None, q)
    if len(p) != 6:
        raise ValueError('a crop needs 4 or 6 points, not %d' % len(p))
    tl, tr, br, bl = order(p[:4]).astype(np.float64)
    fa, fb, across = fold_ends(tl, tr, br, bl, p[4], p[5])
    P, _ = panel_quads(tl, tr, br, bl, fa, fb, across)
    hexa = np.float32([P[0][0], P[0][1], P[0][2], P[1][2], P[1][3], P[0][3]] if across else [P[0][0], P[0][1], P[1][1], P[1][2], P[1][3], P[0][3]])
    h0, w0 = img.shape[:2]
    s = min(1.0, WORK / max(h0, w0))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    comp, level, m, _ = paper_blob(small, hexa * s)
    mf = cf = None
    if comp is not None:
        blob = fill_blob(comp) > 0
        iou = (blob & (m > 0)).sum() / max(1, (blob | (m > 0)).sum())
        if iou >= 0.9:
            _, mf, cf = paper_full(img, comp, level, s)
        log('  drawn outline vs paper: IoU %.2f (%s)' % (iou, 'edges follow the paper' if mf is not None else 'drawn sides kept'))
    return warp_panels(img, P, across, kind, mm, mf, cf, log=log, reach=0.04)

def warp_panels(img, P, across, kind, mm, mf, cf, log=print, reach=0.08, shift=None):
    h0, w0 = img.shape[:2]
    fold_side = [2, 0] if across else [1, 3]
    P = [np.float32(pk) for pk in P]
    fa, fb = (P[0][3], P[0][2]) if across else (P[0][1], P[0][2])
    nrm = np.linalg.norm
    if across:
        along = max(nrm(P[0][1] - P[0][0]), nrm(fb - fa), nrm(P[1][2] - P[1][3]))
    else:
        along = max(nrm(P[0][3] - P[0][0]), nrm(fb - fa), nrm(P[1][2] - P[1][1]))
    est = []
    for pk in P:
        rr, _ = true_ratio(pk, w0, h0)
        est.append(along / rr if across else along * rr)
    if kind == 'passport':
        depth = [along * PAGE] * 2
    elif kind == 'a4' or mm:
        total = along * (297 / 210 if kind == 'a4' else max(mm) / min(mm))
        depth = [total * e / sum(est) for e in est]
    else:
        depth = est
    log('  two planes (%s, at the %s): panels %.3f + %.3f of the fold length (from perspective %.3f + %.3f)' % ('fold across' if across else 'fold upright', 'spine' if kind == 'passport' else 'fold', depth[0] / along, depth[1] / along, est[0] / along, est[1] / along))
    along, d1, d2 = (int(round(along)), int(round(depth[0])), int(round(depth[1])))
    rects = [(0, 0, along, d1), (0, d1, along, d1 + d2)] if across else [(0, 0, d1, along), (d1, 0, d1 + d2, along)]
    W, H = (rects[1][2], rects[1][3])
    ins = max(1.5, 0.002 * max(W, H))
    mapx = np.empty((H, W), np.float32)
    mapy = np.empty((H, W), np.float32)
    for pk, rect, fs in zip(P, rects, fold_side):
        x0, y0, x1, y1 = rect
        M = cv2.getPerspectiveTransform(pk, np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        pts = None if cf is None else cv2.perspectiveTransform(cf.reshape(-1, 1, 2), M.astype(np.float64)).reshape(-1, 2)
        off, inset = ([], [])
        for side in range(4):
            if side == fs:
                off.append(np.zeros(160))
                inset.append(0.0)
                continue
            o = np.zeros(160) if pts is None else side_offsets(pts, rect, side, reach=reach, shift=shift)
            if (side - fs) % 2:
                t = np.linspace(0, 1, len(o))
                end = 1.0 if fs in (1, 2) else 0.0
                o = o - o[-1 if end else 0] * np.clip(1 - np.abs(t - end) / 0.15, 0, 1)
            off.append(o)
            inset.append(ins)
        grid = np.linspace(0, 1, 160)
        Mi = np.linalg.inv(M)
        for r0 in range(y0, y1, STRIP):
            r1 = min(y1, r0 + STRIP)
            X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(r0, r1) + 0.5)
            su, tv = ((X - x0) / (x1 - x0), (Y - y0) / (y1 - y0))
            oT, oR, oB, oL = [np.interp(su if side in (0, 2) else tv, grid, off[side]) + inset[side] for side in range(4)]
            SX = X + (1 - su) * oL - su * oR
            SY = Y + (1 - tv) * oT - tv * oB
            src = cv2.perspectiveTransform(np.stack([SX, SY], -1).reshape(-1, 1, 2), Mi).reshape(r1 - r0, x1 - x0, 2)
            mapx[r0:r1, x0:x1], mapy[r0:r1, x0:x1] = (src[..., 0], src[..., 1])
    flat = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if kind in ('a4', 'doc'):
        D, n = line_shift(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), across)
        if D is not None:
            X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            X, Y = (X, Y + D) if across else (X + D, Y)
            mapx = cv2.remap(mapx, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mapy = cv2.remap(mapy, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            flat = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            log('  %d text lines straightened (largest shift %.1f px)' % (n, float(np.abs(D).max())))
    if mf is None:
        return (flat, rects, P)
    e = 2 * int(np.ceil(0.002 * max(h0, w0))) + 1
    a = cv2.remap(cv2.erode(mf, np.ones((e, e), np.uint8)).astype(np.float32), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    z = max(3, int(0.03 * min(W, H)))
    zone = np.ones((H, W), bool)
    zone[z:-z, z:-z] = False
    a[~zone] = 1.0
    if (a < 0.99).any() and (zone & (a > 0.99)).any():
        paper = np.median(flat[zone & (a > 0.99)], 0)
        t = a != 1.0
        at = a[t][:, None]
        flat[t] = np.clip(flat[t] * at + paper[None] * (1 - at), 0, 255).astype(np.uint8)
        log('  %d px of table at the corners/spine ends painted paper-white' % int((a < 0.5).sum()))
    return (flat, rects, P)

def line_shift(g, across, n=12):
    if not across:
        D, k = line_shift(np.ascontiguousarray(g.T), True, n)
        return (None if D is None else np.ascontiguousarray(D.T), k)
    h, w = g.shape
    k = max(9, int(0.03 * max(h, w)) | 1)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    _, ink = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 60), 1)))
    xs = (np.arange(n) + 0.5) * w / n
    runs = []
    for j in range(n):
        prof = ink[:, int(j * w / n):int((j + 1) * w / n)].mean(1) / 255
        d = np.diff(np.r_[0, (prof > 0.3).astype(np.int8), 0])
        st, en = (np.flatnonzero(d == 1), np.flatnonzero(d == -1))
        runs.append([(s + e) / 2 for s, e in zip(st, en) if 0.003 * h <= e - s <= 0.03 * h])
    mid = [r for rr in runs[n // 3:2 * n // 3] for r in rr]
    if len(mid) < 18:
        return (None, 0)
    pitch = float(np.median(np.diff(np.sort(runs[n // 2])))) if len(runs[n // 2]) > 3 else 0.02 * h
    tol = 0.35 * pitch
    lines = []
    for j in range(n):
        free = list(runs[j])
        for ln in sorted(lines, key=lambda l: -len(l)):
            sj, y = ln[-1]
            if j - sj > 2 or not free:
                continue
            c = min(free, key=lambda r: abs(r - y))
            if abs(c - y) < tol:
                ln.append((j, c))
                free.remove(c)
        lines += [[(j, r)] for r in free]
    samples = [[] for _ in range(n)]
    used = 0
    for ln in lines:
        if len(ln) < max(5, n // 2):
            continue
        j = np.array([p[0] for p in ln])
        y = np.array([p[1] for p in ln])
        c = np.polyfit(xs[j], y, 2)
        if np.sqrt(np.mean((np.polyval(c, xs[j]) - y) ** 2)) > 0.15 * pitch:
            continue
        target = float(np.mean(np.polyval(c, xs[j])))
        used += 1
        for jj in range(max(0, j.min() - 1), min(n, j.max() + 2)):
            samples[jj].append((target, float(np.polyval(c, xs[jj])) - target))
    if used < 6:
        return (None, used)
    rows = np.arange(h, dtype=np.float32)
    cols = []
    for jj in range(n):
        s = sorted(samples[jj] + [(0.0, 0.0), (h - 1.0, 0.0)])
        cols.append(np.interp(rows, [p[0] for p in s], [np.clip(p[1], -pitch, pitch) for p in s]))
    D = cv2.resize(np.stack(cols, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return (cv2.GaussianBlur(D, (0, 0), max(1.0, 0.3 * pitch)), used)

def paper_light(L, smooth):
    h, w = L.shape
    s = 256 / max(h, w)
    sm = cv2.resize(L, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA).astype(np.float32)
    k = max(5, int(0.12 * max(sm.shape)) | 1)
    bg = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, 5)
    if smooth == 'surface':
        yy, xx = np.mgrid[0:sm.shape[0], 0:sm.shape[1]].astype(np.float32)
        xx, yy = (xx / sm.shape[1] - 0.5, yy / sm.shape[0] - 0.5)
        A = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1).reshape(-1, 6)
        z = bg.reshape(-1)
        keep = np.ones(len(z), bool)
        for _ in range(3):
            coef = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
            keep = z > A @ coef - 12
        bg = (A @ coef).reshape(sm.shape).astype(np.float32)
    else:
        bg = cv2.GaussianBlur(bg, (0, 0), k / 5, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)

def light_split(L, share):
    h, w = L.shape
    fy, r, k = (int(round(share * h)), max(4, int(0.06 * h)), max(2, int(0.01 * h)))
    lo, hi = (max(k, fy - r), min(h - k, fy + r))
    if hi <= lo:
        return fy
    p = np.percentile(L[lo - k:hi + k, int(0.05 * w):int(0.95 * w)], 85, axis=1)
    c = np.r_[0, np.cumsum(p)]
    i = np.arange(k, hi - lo + k)
    step = np.abs(c[i] - c[i - k] - (c[i + k] - c[i])) / k
    j = int(np.argmax(step))
    return lo + j if step[j] >= 6 else fy

def fold_band(L, bg, fy):
    h, w = L.shape
    band = max(4, int(0.06 * h))
    rel = L / bg
    lo, hi = (max(0, fy - 2 * band), min(h, fy + 2 * band))
    p = np.percentile(rel[lo:hi, int(0.05 * w):int(0.95 * w)], 85, axis=1)
    corr = np.ones(hi - lo, np.float32)
    for a, b, ra, rb in ((fy - band, fy, fy - 2 * band, fy - band), (fy, fy + band, fy + band, fy + 2 * band)):
        a, b, ra, rb = [min(max(v, lo), hi) - lo for v in (a, b, ra, rb)]
        if b - a < 2 or rb - ra < 2:
            continue
        ref = float(np.median(p[ra:rb]))
        c = np.clip(ref / np.maximum(p[a:b], 0.001), 1.0, 1.4)
        c = cv2.GaussianBlur(c.reshape(-1, 1).astype(np.float32), (1, 0), max(1.0, 0.003 * h), borderType=cv2.BORDER_REPLICATE).flatten()
        t = np.abs(np.arange(a, b) - (fy - lo - 0.5)) / band
        corr[a:b] = 1 + (c - 1) * np.clip(1.5 * (1 - t), 0, 1)
    out = bg.copy()
    out[lo:hi] /= corr[:, None]
    return out

def enhance(img, kind, mode, panels=None):
    text = kind in ('a4', 'doc')
    h0, w0 = img.shape[:2]
    up = min(2.0, 2400 / max(img.shape[:2]))
    if up > 1.05:
        img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_LANCZOS4)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if panels:
        across = panels[0][3] == panels[1][1]
        Lt = L if across else np.ascontiguousarray(L.T)
        fy = light_split(Lt, panels[0][3] / h0 if across else panels[0][2] / w0)
        bt = np.empty_like(Lt)
        bt[:fy] = paper_light(Lt[:fy], 'local' if text else 'surface')
        bt[fy:] = paper_light(Lt[fy:], 'local' if text else 'surface')
        bt = fold_band(Lt, np.maximum(bt, 1), fy)
        bg = bt if across else np.ascontiguousarray(bt.T)
        cut = (slice(0, fy), slice(fy, None))
        parts = [(c, slice(None)) for c in cut] if across else [(slice(None), c) for c in cut]
    else:
        parts = [(slice(None), slice(None))]
        bg = np.maximum(paper_light(L, 'local' if text else 'surface'), 1)
    target = 250.0 if text else min(250.0, float(np.percentile(bg, 95)) * 1.08)
    gain = np.clip(target / bg, 0.85, 2.5)
    del bg
    if panels:
        del bt, Lt
    f = img.astype(np.float32)
    f *= gain[..., None]
    L *= gain
    Lg = L
    del gain
    papers = [np.median(f[sl][Lg[sl] > np.percentile(Lg[sl], 70)], 0) for sl in parts]
    del L, Lg
    if text:
        for sl, paper in zip(parts, papers):
            f[sl] *= np.clip(target / np.maximum(paper, 1), 0.8, 1.35)[None, None, :]
    elif len(parts) > 1:
        mean = np.mean(papers, 0)
        for sl, paper in zip(parts, papers):
            f[sl] *= np.clip(mean / np.maximum(paper, 1), 0.85, 1.15)[None, None, :]
    np.clip(f, 0, 255, out=f)
    g = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lo = float(np.percentile(g, 0.8 if text else 0.2))
    del g
    f -= lo
    f *= 255.0
    f /= max(252.0 - lo, 1)
    np.clip(f, 0, 255, out=f)
    blur = cv2.GaussianBlur(f, (0, 0), max(up, 1.0))
    np.subtract(f, blur, out=blur)
    blur *= 0.6 if text else 0.4
    blur += f
    del f
    f = np.clip(blur, 0, 255, out=blur).astype(np.uint8)
    del blur
    if mode == 'color':
        return f
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if mode == 'gray':
        return g
    blk = max(15, int(max(g.shape) / 40) | 1)
    local = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 10)
    T, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.where((local == 0) & (g < T + 20), 0, 255).astype(np.uint8)
SHARP_MIN = 0.25
SHARP_PXMM = {'card': 12.0, 'passport': 12.0}
LIGHT_MIN = 100
CLIP, GLARE_MIN = (250, 0.002)
DPI_MIN, PHOTO_MIN = (120, 1500)
EDGE = 0.004
LONG_MM = {'card': 85.6, 'passport': 176.0, 'a4': 297.0}

def sharpness(g, t=32):
    g = g.astype(np.float32)
    H, W = (g.shape[0] // t * t, g.shape[1] // t * t)
    if not H or not W:
        return None
    tiles = lambda a: a[:H, :W].reshape(H // t, t, W // t, t).std(axis=(1, 3))
    sg = tiles(g)
    busy = sg > 15
    if busy.sum() < 6:
        return None
    r = None
    for dx, dy in ((2, 0), (0, 2)):
        s = tiles(cv2.Sobel(g, cv2.CV_32F, dx, dy, ksize=3))
        e = np.sqrt(np.maximum(s ** 2 - np.percentile(s, 10) ** 2, 0))
        r = e if r is None else np.minimum(r, e)
    return float(np.percentile(r[busy] / sg[busy], 80))

def page_goes_on(small, quad, comp, m, cover=0.8, near=30.0):
    sh, sw = small.shape[:2]
    inside = cv2.erode(m, np.ones((15, 15), np.uint8)) > 0
    if not inside.any():
        return False
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper = np.median(lab[inside], 0)
    for a, b, n in sides(quad):
        p = a[None] + np.linspace(0.05, 0.95, 60)[:, None] * (b - a)[None]
        pts = np.concatenate([p + t * n for t in (8, 12, 16)])
        x, y = (np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int))
        ok = (x >= 0) & (x < sw) & (y >= 0) & (y < sh)
        if ok.sum() < 30:
            continue
        on = comp[y[ok], x[ok]]
        if on.mean() >= cover and np.linalg.norm(np.median(lab[y[ok], x[ok]][on], 0) - paper) <= near:
            return True
    return False

def cut_by_frame(img, q, kind=None):
    h, w = img.shape[:2]
    s = min(1.0, WORK / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    sh, sw = small.shape[:2]
    P = np.asarray(q, np.float32).reshape(-1, 2) if q is not None else None
    quad = order(P[:4] * s) if q is not None else np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])
    comp, level, m, g = paper_blob(small, quad)
    lim = max(3.0, EDGE * max(h, w))
    if q is not None and any((min(x, y, w - 1 - x, h - 1 - y) < lim for x, y in P)):
        by, bx = (np.concatenate([np.zeros(sw), np.full(sw, sh - 1), np.arange(sh), np.arange(sh)]), np.concatenate([np.arange(sw), np.arange(sw), np.zeros(sh), np.full(sh, sw - 1)]))
        yi, xi = (by.astype(int), bx.astype(int))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        inside = cv2.erode(m, np.ones((15, 15), np.uint8)) > 0
        paper = np.median(lab[inside], 0) if inside.any() else np.float32([255, 128, 128])
        lit = (cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)[yi, xi] > level) & (np.linalg.norm(lab[yi, xi, 1:] - paper[1:], axis=1) <= 10)
        for x, y in P:
            if min(x, y, w - 1 - x, h - 1 - y) < lim:
                near = np.hypot(bx - x * s, by - y * s) <= 0.03 * max(sh, sw)
                if near.any() and lit[near].mean() >= 0.2:
                    return True
    if kind == 'card' and q is not None and (len(P) == 4) and (shape_dev(P, w, h, 'card') <= 0.05) and (opposite_ratio(P) <= 1.3):
        return False
    if comp is None:
        return False
    share = frame_share(comp)
    if share < 0.01 or share > 0.8:
        return False
    if q is None:
        return True
    if not page_goes_on(small, quad, comp, m):
        return False
    ink = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0), 60, 160) > 0
    d = 2 * int(0.015 * max(sh, sw)) + 1
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
    out = comp & (cv2.dilate(m, k) == 0)
    inn = cv2.erode(m, k) > 0
    if out.sum() < 0.01 * sh * sw or not inn.any():
        return False
    return bool(ink[out].mean() > max(0.01, 0.3 * ink[inn].mean()))

def quality(img, flat, q=None, kind=None, mm=None):
    long_mm = float(max(mm)) if mm else LONG_MM.get(kind, 297.0)
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY) if flat.ndim == 3 else flat
    k = min(1.0, SHARP_PXMM.get(kind, 6.0) * long_mm / max(g.shape))
    sharp = sharpness(cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else g)
    light = float(np.percentile(g[::2, ::2], 95))
    glare = 0.0
    if kind in ('card', 'passport') and flat.ndim == 3:
        hot = (flat[::2, ::2].min(axis=2) >= CLIP).astype(np.uint8)
        hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(hot)
        if n > 1:
            glare = float(st[1:, cv2.CC_STAT_AREA].max() / hot.size)
    photo = max(img.shape[:2])
    dpi = max(flat.shape[:2]) / (long_mm / 25.4)
    cut = cut_by_frame(img, q, kind)
    flags = [f for f, on in (('blur', sharp is not None and sharp < SHARP_MIN), ('glare', glare >= GLARE_MIN), ('dark', light < LIGHT_MIN), ('lowres', photo < PHOTO_MIN or dpi < DPI_MIN), ('cut', cut), ('edge', kind in ('a4', 'doc') and weak_edges(img, q))) if on]
    return {'sharp': None if sharp is None else round(sharp, 3), 'light': round(light), 'glare': round(glare, 4), 'dpi': round(dpi), 'photo_px': int(photo), 'cut': bool(cut), 'flags': flags}

def side_contrast(lab, q, d=8):
    h, w = lab.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [order(q).astype(np.int32)], 1)
    inner = cv2.erode(m, np.ones((2 * d + 1, 2 * d + 1), np.uint8)) > 0
    if not inner.any():
        return [None] * 4
    paper, out = (np.median(lab[inner], 0), [])
    for a, b, n in sides(order(q)):
        vo, oo = sample(lab, a[None] + np.linspace(0.1, 0.9, 24)[:, None] * (b - a)[None] + d * n)
        out.append(float(np.linalg.norm(np.median(vo[oo], 0) - paper)) if oo.sum() >= 6 else None)
    return out

def weak_edges(img, q, small_page=0.2, mean=0.85, low=0.75, beyond=0.15):
    if q is None or len(q) != 4:
        return False
    h, w = img.shape[:2]
    s = min(1.0, WORK / max(h, w))
    quad = order(np.asarray(q, np.float32).reshape(-1, 2) * s)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    lab = cv2.cvtColor(wipe_ink(small, 9), cv2.COLOR_BGR2LAB).astype(np.float32)
    con = [c for c in side_contrast(lab, quad) if c is not None]
    if len(con) == 4:
        k = int(np.argmin(con))
        rest = float(np.median(con[:k] + con[k + 1:]))
        if con[k] < 2 * HIT and con[k] < beyond * rest and (rest >= 2 * HIT):
            return True
    if cv2.contourArea(quad) >= small_page * small.shape[0] * small.shape[1]:
        return False
    sup = float(np.mean(side_support(lab, quad)))
    return sup < low or (sup < mean and len(con) >= 3 and (float(np.median(con)) >= 3 * HIT))

def rect_off(q, w, h, f_lo=0.4, f_hi=3.0):
    L = float(max(w, h))
    P = [np.array([(x - w / 2) / L, (y - h / 2) / L, 1.0]) for x, y in order(np.asarray(q, np.float64).reshape(-1, 2)[:4])]
    v1 = np.cross(np.cross(P[0], P[1]), np.cross(P[3], P[2]))
    v2 = np.cross(np.cross(P[0], P[3]), np.cross(P[1], P[2]))
    v1, v2 = (v1 / (np.linalg.norm(v1) + 1e-12), v2 / (np.linalg.norm(v2) + 1e-12))
    f = np.linspace(f_lo, f_hi, 53)[:, None]
    d1 = np.hstack([np.broadcast_to(v1[:2], (len(f), 2)), v1[2] * f])
    d2 = np.hstack([np.broadcast_to(v2[:2], (len(f), 2)), v2[2] * f])
    c = np.abs((d1 * d2).sum(1)) / (np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) + 1e-12)
    return float(c.min())

def not_a_page(q, w, h):
    return opposite_ratio(q) > 2.0 or rect_off(q, w, h) > 0.1

def opposite_ratio(q):
    q = order(np.asarray(q, np.float64).reshape(-1, 2)[:4])
    L = [np.linalg.norm(q[(k + 1) % 4] - q[k]) for k in range(4)]
    return max(max(L[0], L[2]) / max(min(L[0], L[2]), 1e-06), max(L[1], L[3]) / max(min(L[1], L[3]), 1e-06))
