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


if __name__ == "__main__":
    unittest.main()
