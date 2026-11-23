#!/usr/bin/env python3
"""Float32 model of the translated image_bvh step, and a check that a BVH loses no hit its leaves give.

tools/xdxr_bvh.py traces in double precision. The translated instruction (tools/rdna2spirv.py image_bvh, every float
operation NoContraction) works in float32: the box test rounds (plane - origin) * inverse direction, the triangle
test rounds its shear and edge functions, and the two decisions are made apart. Where the edge two triangles share
also lies on the face of their boxes, a ray can belong to one triangle by the edge rule and fall outside that
triangle's box by one rounding step: both are skipped and the ray passes through a closed mesh. This module runs
the step with float32 rounding after every operation and compares a whole BLAS traversal with a brute force over
every leaf of the same buffer: the tree must not lose a hit that the leaf test finds (docs/bvh.md 7a, box padding).

box_rounding "nearest" is the translator on main; "zero" is the slab rounding measured on RDNA2 for #4957,
t = rz(rz(plane - origin) * inverse direction). The triangle test rounds to nearest in both.

Standard library only. Divisions round the double quotient to float32 (Vulkan allows 2.5 ulp for OpFDiv).
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xdxr_bvh as X  # noqa: E402

INF = math.inf
FLT_MAX = X.FLT_MAX


def _r32(x):
    """x rounded to the nearest float32; inf past the range."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(INF, x)


def _step32(x, up):
    """The float32 next to the float32 x, above (up) or below."""
    if x == 0.0:
        return struct.unpack("<f", struct.pack("<I", 0x00000001 if up else 0x80000001))[0]
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    b = b + 1 if (x > 0.0) == up else b - 1
    return struct.unpack("<f", struct.pack("<I", b))[0]


def _round(s, e=0.0, rtz=False):
    """float32 rounding of the exact value s + e (s a double, e the error of the operation that made it)."""
    if not math.isfinite(s):
        return s
    r = _r32(s)
    if math.isinf(r):
        return math.copysign(FLT_MAX, s) if rtz else r
    if rtz:
        if abs(r) > abs(s) or (r == s and e != 0.0 and (e < 0.0) == (s > 0.0)):
            r = _step32(r, r < 0.0)
        return r
    if e != 0.0 and r != s:
        other = _step32(r, s > r)
        if (r + other) / 2 == s:                     # a float32 midpoint in double: the error decides
            r = max(r, other) if e > 0.0 else min(r, other)
    return r


def _sub(a, b, rtz=False):
    s = a - b
    if not math.isfinite(s):
        return s
    bb = s - a
    e = (a - (s - bb)) + (-b - bb)                   # TwoSum of a and -b: exact error of s
    return _round(s, e, rtz)


def _add(a, b):
    return _sub(a, -b)


def _mul(a, b, rtz=False):
    return _round(a * b, 0.0, rtz)                   # a product of two float32 values is exact in double


def _div(a, b):
    try:
        return _r32(a / b)
    except ZeroDivisionError:
        if a == 0.0 or math.isnan(a):
            return math.nan
        return math.copysign(INF, a) * math.copysign(1.0, b)


def box_test32(extent, org, inv, lo, hi, rtz=False):
    """Entry distance into one box child, or inf: the translator's slab test (rdna2spirv.py image_bvh)."""
    near, far, nn, fn = [], [], [], []
    for c in range(3):
        b0 = _mul(_sub(lo[c], org[c], rtz), inv[c], rtz)
        b1 = _mul(_sub(hi[c], org[c], rtz), inv[c], rtz)
        n_, f_ = (b0, b1) if inv[c] >= 0.0 else (b1, b0)
        nn.append(math.isnan(n_))
        fn.append(math.isnan(f_))
        near.append(-INF if nn[-1] else n_)
        far.append(INF if fn[-1] else f_)
    tmin, tmax = max(near), min(far)
    if all(nn) or all(fn):
        return INF
    tnear = max(0.0, tmin)
    return tnear if tmax >= tnear and tmin < extent else INF


def triangle_test32(v, org, d, inv, remap=X.REMAP_IDENTITY):
    """(t/|det|, sign(det), b_i/|det|, b_j/|det|), or (inf, 1, 0, 0): the translator's triangle step."""
    ad = [abs(x) for x in d]
    if ad[0] >= ad[1]:
        kx0, ky0, kz = (1, 2, 0) if ad[0] >= ad[2] else (0, 1, 2)
    else:
        kx0, ky0, kz = (2, 0, 1) if ad[1] >= ad[2] else (0, 1, 2)
    kx, ky = (ky0, kx0) if d[kz] < 0.0 else (kx0, ky0)
    sz = inv[kz]
    sx, sy = _mul(d[kx], sz), _mul(d[ky], sz)

    def shear(p):
        r = [_sub(p[c], org[c]) for c in range(3)]
        return _sub(r[kx], _mul(sx, r[kz])), _sub(r[ky], _mul(sy, r[kz])), r[kz]
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = (shear(p) for p in v)
    uu = _sub(_mul(cx, by), _mul(cy, bx))
    vv = _sub(_mul(ax, cy), _mul(ay, cx))
    ww = _sub(_mul(bx, ay), _mul(by, ax))
    cond = not ((uu < 0 or vv < 0 or ww < 0) and (uu > 0 or vv > 0 or ww > 0))
    ea, eb, ec = uu == 0.0, vv == 0.0, ww == 0.0

    def edge(x0, y0, x1, y1):
        tx, ty = _sub(x1, x0), _sub(y1, y0)
        return ty < 0.0 if tx == 0.0 else tx < 0.0

    def vert(x0, x1, x2):
        return _sub(x1, x0) <= 0.0 and _sub(x2, x0) > 0.0
    ie = cond and ((ea and edge(bx, by, cx, cy)) or (eb and edge(cx, cy, ax, ay)) or (ec and edge(ax, ay, bx, by)))
    isva, isvb, isvc = eb and ec, ea and ec, ea and eb
    iv = not (isva or isvb or isvc) or (isva and vert(ax, bx, cx)) or (isvb and vert(bx, cx, ax)) or \
        (isvc and vert(cx, ax, bx))
    hitc = (ie and iv) if (ea or eb or ec) else cond
    det = _add(uu, _add(vv, ww))
    tt = _add(_add(_mul(uu, _mul(sz, az)), _mul(vv, _mul(sz, bz))), _mul(ww, _mul(sz, cz)))
    sg = 1.0 if det > 0.0 else -1.0 if det < 0.0 else 0.0
    if not (hitc and not _mul(sg, tt) < 0.0):
        return (INF, 1.0, 0.0, 0.0)
    dab = abs(det)
    pick = lambda s: ww if s == 2 else vv if s == 1 else uu                    # noqa: E731
    return (_div(tt, dab), sg, _div(pick(remap & 3), dab), _div(pick((remap >> 2) & 3), dab))


def inverse_dir32(d):
    return [_div(1.0, c) if c != 0.0 else math.copysign(FLT_MAX, c) for c in d]


def _leaf_hit(mem, bvh_va, p, org, d, inv, best):
    va = bvh_va + X.ptr_offset(p)
    kind = X.ptr_type(p)
    verts = [mem.f32(va + 12 * s, 3) for s in X.TRI_VERTS[kind]]
    remap = (mem.u32(va + 60)[0] >> (8 * kind)) & 0xFF
    rt, rs, _, _ = triangle_test32(verts, org, d, inv, remap)
    if rt == INF:
        return None
    t = _div(rt, rs)
    if not t <= best:
        return None
    gif, base, prim2 = mem.u32(va + 48, 3)
    return {"t": t, "geometry": gif >> 8, "primitive": prim2 if kind == X.TRI1 else base}


def trace32(mem, bvh_va, org, d, tmax=INF, box_rounding="nearest"):
    """Closest hit in the BLAS at bvh_va through its tree, with the float32 step: a dict with t, geometry and
    primitive, or None."""
    rtz = box_rounding == "zero"
    inv = inverse_dir32(d)
    hit = None
    stack = [mem.u32(bvh_va + 48)[0]]
    while stack:
        p = stack.pop()
        if p == X.INVALID:
            continue
        kind = X.ptr_type(p)
        best = hit["t"] if hit else tmax
        if kind <= X.TRI3:
            h = _leaf_hit(mem, bvh_va, p, org, d, inv, best)
            hit = h or hit
            continue
        if kind not in (X.BOX16, X.BOX32):
            continue
        va = bvh_va + X.ptr_offset(p)
        kids = list(mem.u32(va, 4))
        dist = []
        for i in range(4):
            if kind == X.BOX16:
                lo, hi = X.decode_box16(mem.u32(va + 16 + 12 * i, 3))
            else:
                b = mem.f32(va + 16 + 24 * i, 6)
                lo, hi = b[:3], b[3:]
            dist.append(box_test32(best, org, inv, lo, hi, rtz))
        kids = [k if dist[i] != INF else X.INVALID for i, k in enumerate(kids)]
        for a, b in ((0, 2), (1, 3), (0, 1), (2, 3), (1, 2)):
            if dist[b] < dist[a]:
                dist[a], dist[b] = dist[b], dist[a]
                kids[a], kids[b] = kids[b], kids[a]
        stack.extend(reversed([k for k in kids if k != X.INVALID]))
    return hit


def leaves32(mem, bvh_va, org, d, tmax=INF):
    """Closest hit over every leaf of the BLAS at bvh_va, no boxes: what its tree must not lose."""
    inv = inverse_dir32(d)
    # numElements counts triangle references; a paired record (TRI0 + TRI1) holds two, so walk the 64-byte leaf
    # records from offsetToElements up to offsetToNodes instead
    nodes, elements = mem.u32(bvh_va + 40)[0], mem.u32(bvh_va + 52)[0]
    hit = None
    for e in range((nodes - elements) // 64):
        off = elements + 64 * e
        kinds = [X.TRI0] + ([X.TRI1] if mem.u32(bvh_va + off + 56)[0] != X.INVALID else [])
        for kind in kinds:
            h = _leaf_hit(mem, bvh_va, (off // X.UNIT) << 3 | kind, org, d, inv, hit["t"] if hit else tmax)
            hit = h or hit
    return hit


def edge_rays(triangles, count, rng, spread=12.0):
    """Rays from random origins through a vertex or a point on an edge of a random triangle (float32 values)."""
    tris = [t for t in triangles if all(math.isfinite(c) for p in t for c in p)]
    rays = []
    for _ in range(count if tris else 0):
        t = rng.choice(tris)
        i = rng.randrange(3)
        a, b = t[i], t[(i + 1) % 3]
        s = 0.0 if rng.random() < 0.5 else rng.random()
        p = [a[c] + s * (b[c] - a[c]) for c in range(3)]
        o = [X.f32(rng.uniform(-spread, spread)) for _ in range(3)]
        rays.append((o, [X.f32(p[c] - o[c]) for c in range(3)]))
    return rays


def lost_hits(mem, bvh_va, rays, box_rounding="nearest"):
    """The rays whose leaf test finds a nearer hit than the traversal of the tree: (origin, direction, tree, leaves)."""
    lost = []
    for o, d in rays:
        want = leaves32(mem, bvh_va, o, d)
        if want is None:
            continue
        got = trace32(mem, bvh_va, o, d, box_rounding=box_rounding)
        if got is None or got["t"] > want["t"]:
            lost.append((o, d, got, want))
    return lost
