"""A BVH must keep every hit its leaves give, with the float32 step of the translated instruction.

tools/xdxr_bvh32.py traces a BLAS through its tree and over all its leaves with float32 rounding after every
operation, as tools/rdna2spirv.py's image_bvh does. A ray through the edge two triangles share, where that edge is
also the face of their boxes, used to be cut by the box of the triangle the edge rule gives it to and pass between
both (docs/bvh.md 7a, box padding). The structures here come from tools/xdxr_bvh.py build_blas; the GPU builder's
are checked the same way by check-build on vk_xdxr_build_test's watertight_shared_edge and watertight_closed_cube.

    python -I tests/test_xdxr_bvh32.py
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import xdxr_bvh as X  # noqa: E402
import xdxr_bvh32 as B  # noqa: E402

BASE = 0x10000
# found by tools/xdxr_bvh32.py on a GPU-built pair: the ray crosses the shared edge x = 1 at y ~ 0.765
RAY = ([3.84750438, 1.09573054, 0.410529882], [-0.983276129, -0.114332438, -0.141760707])
PAIR = [[(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(1, 0, 0), (2, 0, 0), (1, 1, 0)]]


def closed_cube(n=4, h=2.0):
    tris = []
    for axis in range(3):
        for side in (-h, h):
            def at(i, j):
                p = [0.0, 0.0, 0.0]
                p[axis] = side
                p[(axis + 1) % 3] = -h + 2 * h * i / n
                p[(axis + 2) % 3] = -h + 2 * h * j / n
                return tuple(p)
            for j in range(n):
                for i in range(n):
                    tris.append([at(i, j), at(i + 1, j), at(i + 1, j + 1)])
                    tris.append([at(i, j), at(i + 1, j + 1), at(i, j + 1)])
    return tris


def placed(tris):
    data, _ = X.build_blas(tris)
    mem = X.Memory()
    mem.place(BASE, data)
    return mem


def ray32(r):
    return [X.f32(c) for c in r[0]], [X.f32(c) for c in r[1]]


class TreeKeepsLeafHits(unittest.TestCase):
    def test_ray_through_the_shared_edge_hits(self):
        mem = placed(PAIR)
        o, d = ray32(RAY)
        leaf = B.leaves32(mem, BASE, o, d)
        self.assertIsNotNone(leaf)                                 # the edge rule gives the ray to triangle 1
        self.assertEqual(leaf["primitive"], 1)
        for rounding in ("nearest", "zero"):
            with self.subTest(box_rounding=rounding):
                got = B.trace32(mem, BASE, o, d, box_rounding=rounding)
                self.assertIsNotNone(got)
                self.assertEqual((got["primitive"], got["t"]), (1, leaf["t"]))

    def test_paired_leaves(self):
        # build_blas(pair=True) puts two triangles in one leaf record (TRI0 and TRI1): numElements counts triangle
        # references, not records, so the leaf brute force must walk the records up to offsetToNodes
        cube = closed_cube()
        data, info = X.build_blas(cube, pair=True)
        self.assertLess(info["leaves"], info["triangles"])
        mem = X.Memory()
        mem.place(BASE, data)
        rays = B.edge_rays(cube, 300, random.Random(11))
        for o, d in rays[:40]:
            self.assertEqual(B.leaves32(mem, BASE, o, d) is None, B.trace32(mem, BASE, o, d) is None, (o, d))
        self.assertEqual(len(B.lost_hits(mem, BASE, rays)), 0)

    def test_edge_and_vertex_rays(self):
        for name, tris in (("pair", PAIR), ("closed cube", closed_cube())):
            mem = placed(tris)
            rays = B.edge_rays(tris, 400, random.Random(5))
            for rounding in ("nearest", "zero"):
                with self.subTest(mesh=name, box_rounding=rounding):
                    self.assertEqual(len(B.lost_hits(mem, BASE, rays, rounding)), 0)


class SameStepAsTheTranslator(unittest.TestCase):
    """The float32 model gives the words tools/rdna2spirv.py's image_bvh gives on the ray above (exact boxes)."""

    def run_step(self, node_type, mem):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic"))
        from support import mrt0_words, translate_lines
        o, d = ray32(RAY)
        inv = B.inverse_dir32(d)
        lines = [f"v_mov_b32_e32 v1, 0x{0x08 | node_type:x}", "v_mov_b32_e32 v2, 0",
                 f"v_mov_b32_e32 v3, 0x{X.struct.unpack('<I', X.struct.pack('<f', 100.0))[0]:x}"]
        for i, x in enumerate(o + d + inv):
            lines.append(f"v_mov_b32_e32 v{4 + i}, 0x{X.struct.unpack('<I', X.struct.pack('<f', x))[0]:x}")
        lines += ["s_mov_b32 s4, 0x10", "s_mov_b32 s5, 0x80000000", "s_mov_b32 s6, -1", "s_mov_b32 s7, 0",
                  "image_bvh64_intersect_ray v[20:23], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]",
                  "exp mrt0 v20, v21, v22, v23 done vm", "s_endpgm"]
        return list(mrt0_words(translate_lines(f"ps_bvh32_model{node_type}", lines), memory=mem)), o, d, inv

    def test_box_node(self):
        node, words = 0x1040, {}
        boxes = [(0, 0, 0, 1, 1, 0), (1, 0, 0, 2, 1, 0)]
        for i, kid in enumerate((0x100, 0x108, X.INVALID, X.INVALID)):
            words[node + 4 * i] = kid
        for i, b in enumerate(boxes + [(1, 1, 1, -1, -1, -1)] * 2):
            for c, v in enumerate(b):
                words[node + 16 + 24 * i + 4 * c] = X.struct.unpack("<I", X.struct.pack("<f", v))[0]
        got, o, d, inv = self.run_step(X.BOX32, words)
        model = [B.box_test32(100.0, o, inv, b[:3], b[3:]) for b in boxes]
        self.assertEqual(got[0], 0x100)                        # only the first triangle's box is entered
        self.assertEqual(got[1:], [X.INVALID] * 3)
        self.assertEqual([m != X.INF for m in model], [True, False])

    def test_triangle_nodes(self):
        for k, tri in enumerate(PAIR):
            words = {0x1040 + 12 * s + 4 * c: X.struct.unpack("<I", X.struct.pack("<f", float(tri[s][c])))[0]
                     for s in range(3) for c in range(3)}
            words[0x1040 + 60] = X.REMAP_IDENTITY
            got, o, d, inv = self.run_step(X.TRI0, words)
            model = B.triangle_test32(tri, o, d, inv)
            with self.subTest(triangle=k):
                self.assertEqual(got, [X.struct.unpack("<I", X.struct.pack("<f", x))[0] for x in model])


class PaddedBoxes(unittest.TestCase):
    def test_box_holds_its_triangle_and_stays_close(self):
        lo, hi = [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]
        plo, phi = X.pad_box32(lo, hi)
        for c in range(3):
            self.assertLess(plo[c], lo[c])
            self.assertGreater(phi[c], hi[c])
            self.assertLess(phi[c] - hi[c], 1e-5)

    def test_empty_box_stays_empty(self):
        e = ([X.INF] * 3, [-X.INF] * 3)
        self.assertEqual(X.pad_box32(*e), e)


if __name__ == "__main__":
    unittest.main()
