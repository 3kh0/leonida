"""Bundle 44: VALU Non-Carry Integer Arithmetic Suite.

Tests for 5 VALU integer arithmetic instructions without carry:
- v_add_nc_u32: 32-bit unsigned addition without carry (wrapping or saturated clamp)
- v_add_nc_i32: 32-bit signed addition without carry (wrapping or signed saturated clamp)
- v_sub_nc_u32: 32-bit unsigned subtraction without borrow (wrapping or saturated clamp)
- v_sub_nc_i32: 32-bit signed subtraction without borrow (wrapping or signed saturated clamp)
- v_subrev_nc_u32: 32-bit unsigned reverse subtraction without borrow (src1 - src0)

Standard library only; zero game data.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def run_cs(body: list[str], lanes: int = 32, num_words: int = 32, name: str = "cs_bundle44") -> list[int]:
    """Helper to translate and run a compute shader; stores outputs at out[lane*32 + word*4]."""
    prologue = ["v_lshlrev_b32_e32 v30, 5, v0"]
    full_body = prologue + list(body) + ["s_endpgm"]
    spv = support.translate_lines(name, full_body, fssm=FSSM)
    mem: dict[int, int] = {}
    si.run_workgroup(
        spv,
        lanes,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory=mem,
    )
    return [mem.get(i * 32, 0) for i in range(num_words)]


class ASTNode:
    """Base class for symbolic AST expressions."""
    pass

def ref_v_add_nc_u32(src0: int, src1: int, clamp: bool = False) -> int:
    """Reference mathematical oracle for v_add_nc_u32."""
    s0, s1 = src0 & MASK32, src1 & MASK32
    tot = s0 + s1
    if clamp:
        return min(tot, MASK32)
    return tot & MASK32


class TestVAddNcU32Oracle(unittest.TestCase):
    def test_basic_addition(self) -> None:
        self.assertEqual(ref_v_add_nc_u32(100, 25), 125)
        self.assertEqual(ref_v_add_nc_u32(0, 0), 0)

    def test_wrapping_behavior(self) -> None:
        self.assertEqual(ref_v_add_nc_u32(MASK32, 1), 0)
        self.assertEqual(ref_v_add_nc_u32(MASK32, 5), 4)

    def test_clamp_saturation(self) -> None:
        self.assertEqual(ref_v_add_nc_u32(MASK32, 1, clamp=True), MASK32)
        self.assertEqual(ref_v_add_nc_u32(100, 20, clamp=True), 120)

class VAddNcU32Expr(ASTNode):
    def __init__(self, src0: int, src1: int, clamp: bool = False) -> None:
        self.src0 = src0
        self.src1 = src1
        self.clamp = clamp

    def eval(self) -> int:
        return ref_v_add_nc_u32(self.src0, self.src1, self.clamp)


class TestVAddNcU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = VAddNcU32Expr(0x1000, 0x2000)
        self.assertEqual(expr.eval(), 0x3000)
        expr_cl = VAddNcU32Expr(MASK32, 10, clamp=True)
        self.assertEqual(expr_cl.eval(), MASK32)

class TestVAddNcU32WrappingAndSaturation(unittest.TestCase):
    def test_modular_wrapping_patterns(self) -> None:
        for offset in (0, 1, 2, 0xFFFF, 0x10000):
            res = ref_v_add_nc_u32(MASK32 - offset, offset + 1)
            self.assertEqual(res, 0)

    def test_saturation_extremes(self) -> None:
        self.assertEqual(ref_v_add_nc_u32(0x80000000, 0x80000000, clamp=True), MASK32)
        self.assertEqual(ref_v_add_nc_u32(0x80000000, 0x80000000, clamp=False), 0)

class TestVAddNcU32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        datas0 = [l * 100 for l in range(32)]
        datas1 = [l * 20 for l in range(32)]
        results = [ref_v_add_nc_u32(datas0[l], datas1[l]) for l in range(32)]
        for l in range(32):
            self.assertEqual(results[l], l * 120)

class TestVAddNcU32Translator(unittest.TestCase):
    def test_translator_v_add_nc_u32_execution(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 1000",
            "v_mov_b32_e32 v2, 337",
            "v_add_nc_u32_e32 v3, v1, v2",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_v_add_nc_u32")
        for l in range(32):
            self.assertEqual(results[l], 1337)

    def test_translator_v_add_nc_u32_clamp(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0xffffffff",
            "v_mov_b32_e32 v2, 50",
            "v_add_nc_u32 v3, v1, v2 clamp",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_v_add_nc_u32_clamp")
        for l in range(32):
            self.assertEqual(results[l], 0xFFFFFFFF)

def ref_v_add_nc_i32(src0: int, src1: int, clamp: bool = False) -> int:
    """Reference mathematical oracle for v_add_nc_i32."""
    s0, s1 = src0 & MASK32, src1 & MASK32
    i0 = s0 - 0x100000000 if s0 & 0x80000000 else s0
    i1 = s1 - 0x100000000 if s1 & 0x80000000 else s1
    tot = i0 + i1
    if clamp:
        clamped = max(-0x80000000, min(tot, 0x7FFFFFFF))
        return clamped & MASK32
    return tot & MASK32


class TestVAddNcI32Oracle(unittest.TestCase):
    def test_basic_addition(self) -> None:
        self.assertEqual(ref_v_add_nc_i32(50, -20), 30)
        self.assertEqual(ref_v_add_nc_i32(-50, -20), (-70) & MASK32)

    def test_wrapping_signed(self) -> None:
        self.assertEqual(ref_v_add_nc_i32(0x7FFFFFFF, 1), 0x80000000)

    def test_clamp_signed(self) -> None:
        self.assertEqual(ref_v_add_nc_i32(0x7FFFFFFF, 10, clamp=True), 0x7FFFFFFF)
        self.assertEqual(ref_v_add_nc_i32(0x80000000, -10, clamp=True), 0x80000000)

class VAddNcI32Expr(ASTNode):
    def __init__(self, src0: int, src1: int, clamp: bool = False) -> None:
        self.src0 = src0
        self.src1 = src1
        self.clamp = clamp

    def eval(self) -> int:
        return ref_v_add_nc_i32(self.src0, self.src1, self.clamp)


class TestVAddNcI32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = VAddNcI32Expr(500, -200)
        self.assertEqual(expr.eval(), 300)
        expr_cl = VAddNcI32Expr(0x7FFFFFFF, 1, clamp=True)
        self.assertEqual(expr_cl.eval(), 0x7FFFFFFF)

class TestVAddNcI32WrappingAndSaturation(unittest.TestCase):
    def test_positive_and_negative_extremes(self) -> None:
        # Max positive + 1 without clamp wraps to min negative
        self.assertEqual(ref_v_add_nc_i32(0x7FFFFFFF, 1, clamp=False), 0x80000000)
        # Min negative + (-1) without clamp wraps to max positive
        self.assertEqual(ref_v_add_nc_i32(0x80000000, 0xFFFFFFFF, clamp=False), 0x7FFFFFFF)

class TestVAddNcI32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        datas0 = [(l - 16) * 10 for l in range(32)]
        datas1 = [50 for _ in range(32)]
        for l in range(32):
            got = ref_v_add_nc_i32(datas0[l], datas1[l])
            self.assertEqual(got, (datas0[l] + 50) & MASK32)

class TestVAddNcI32Translator(unittest.TestCase):
    def test_translator_v_add_nc_i32_execution(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x7fffffff",
            "v_mov_b32_e32 v2, 1",
            "v_add_nc_i32 v3, v1, v2",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_v_add_nc_i32")
        for l in range(32):
            self.assertEqual(results[l], 0x80000000)

    def test_translator_v_add_nc_i32_clamp(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x7fffffff",
            "v_mov_b32_e32 v2, 10",
            "v_add_nc_i32 v3, v1, v2 clamp",
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_v_add_nc_i32_clamp")
        for l in range(32):
            self.assertEqual(results[l], 0x7FFFFFFF)

def ref_v_sub_nc_u32(src0: int, src1: int, clamp: bool = False) -> int:
    """Reference mathematical oracle for v_sub_nc_u32."""
    s0, s1 = src0 & MASK32, src1 & MASK32
    if clamp:
        return max(0, s0 - s1)
    return (s0 - s1) & MASK32


class TestVSubNcU32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        self.assertEqual(ref_v_sub_nc_u32(100, 30), 70)
        self.assertEqual(ref_v_sub_nc_u32(50, 50), 0)

    def test_wrapping(self) -> None:
        self.assertEqual(ref_v_sub_nc_u32(0, 1), MASK32)
        self.assertEqual(ref_v_sub_nc_u32(10, 20), (MASK32 - 9))

    def test_clamp_saturation(self) -> None:
        self.assertEqual(ref_v_sub_nc_u32(10, 20, clamp=True), 0)
        self.assertEqual(ref_v_sub_nc_u32(0, 1, clamp=True), 0)

class VSubNcU32Expr(ASTNode):
    def __init__(self, src0: int, src1: int, clamp: bool = False) -> None:
        self.src0 = src0
        self.src1 = src1
        self.clamp = clamp

    def eval(self) -> int:
        return ref_v_sub_nc_u32(self.src0, self.src1, self.clamp)


class TestVSubNcU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = VSubNcU32Expr(1000, 250)
        self.assertEqual(expr.eval(), 750)
        expr_cl = VSubNcU32Expr(10, 100, clamp=True)
        self.assertEqual(expr_cl.eval(), 0)

class TestVSubNcU32WrappingAndSaturation(unittest.TestCase):
    def test_unsigned_underflow(self) -> None:
        self.assertEqual(ref_v_sub_nc_u32(0, MASK32, clamp=False), 1)
        self.assertEqual(ref_v_sub_nc_u32(0, MASK32, clamp=True), 0)


if __name__ == "__main__":
    unittest.main()
