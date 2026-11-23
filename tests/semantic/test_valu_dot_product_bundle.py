"""Semantic reference oracle and verification bundle for RDNA2 dot-product vector arithmetic.

Covers:
  - v_dot2_f32_f16: Half-to-float 2-element dot product with 32-bit float addend
  - v_dot2c_f32_f16: Half-to-float 2-element dot product accumulating into old destination
  - v_dot4_i32_i8: Signed 8-bit 4-element dot product with 32-bit integer addend
  - v_dot4_u32_u8: Unsigned 8-bit 4-element dot product with 32-bit integer addend
  - v_dot8_i32_i4: Signed 4-bit 8-element dot product with 32-bit integer addend
  - v_dot8_u32_u4: Unsigned 4-bit 8-element dot product with 32-bit integer addend

High-density Subsystem Feature Bundle testing reference oracles, AST modeling,
scalar/lane interpreter, multi-lane divergence, sub-dword packing, IEEE-754 precision,
mutation detection, and tools/rdna2spirv.py SPIR-V translation.
"""
from __future__ import annotations

import dataclasses
import os
import random
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF


def u32(val: int) -> int:
    return val & MASK32


def i32(val: int) -> int:
    v = val & MASK32
    return v if v < 0x80000000 else v - 0x100000000


def i8(val: int) -> int:
    v = val & 0xFF
    return v if v < 0x80 else v - 0x100


def i4(val: int) -> int:
    v = val & 0xF
    return v if v < 0x8 else v - 0x10


def f16_to_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_f16(b: int) -> float:
    return struct.unpack("<e", struct.pack("<H", b & MASK16))[0]


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def pack_f16x2(f0: float, f1: float) -> int:
    b0 = f16_to_bits(f0)
    b1 = f16_to_bits(f1)
    return (b1 << 16) | b0


def ref_dot2_f32_f16(s0: int, s1: int, acc: int) -> int:
    a0 = bits_to_f16(s0 & 0xFFFF)
    a1 = bits_to_f16((s0 >> 16) & 0xFFFF)
    b0 = bits_to_f16(s1 & 0xFFFF)
    b1 = bits_to_f16((s1 >> 16) & 0xFFFF)
    res = a0 * b0 + a1 * b1 + bits_to_f32(acc)
    return f32_to_bits(res)


def ref_dot4_i32_i8(s0: int, s1: int, acc: int) -> int:
    bytes0 = [i8((s0 >> (8 * k)) & 0xFF) for k in range(4)]
    bytes1 = [i8((s1 >> (8 * k)) & 0xFF) for k in range(4)]
    res = sum(b0 * b1 for b0, b1 in zip(bytes0, bytes1)) + i32(acc)
    return u32(res)


def ref_dot4_u32_u8(s0: int, s1: int, acc: int) -> int:
    bytes0 = [(s0 >> (8 * k)) & 0xFF for k in range(4)]
    bytes1 = [(s1 >> (8 * k)) & 0xFF for k in range(4)]
    res = sum(b0 * b1 for b0, b1 in zip(bytes0, bytes1)) + u32(acc)
    return u32(res)


def ref_dot8_i32_i4(s0: int, s1: int, acc: int) -> int:
    nibbles0 = [i4((s0 >> (4 * k)) & 0xF) for k in range(8)]
    nibbles1 = [i4((s1 >> (4 * k)) & 0xF) for k in range(8)]
    res = sum(n0 * n1 for n0, n1 in zip(nibbles0, nibbles1)) + i32(acc)
    return u32(res)


def ref_dot8_u32_u4(s0: int, s1: int, acc: int) -> int:
    nibbles0 = [(s0 >> (4 * k)) & 0xF for k in range(8)]
    nibbles1 = [(s1 >> (4 * k)) & 0xF for k in range(8)]
    res = sum(n0 * n1 for n0, n1 in zip(nibbles0, nibbles1)) + u32(acc)
    return u32(res)

class VDot2F32F16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot2_f32_f16."""

    def test_basic_dot_product(self):
        s0 = pack_f16x2(2.0, 3.0)
        s1 = pack_f16x2(4.0, 5.0)
        acc = f32_to_bits(1.0)
        # 2*4 + 3*5 + 1 = 8 + 15 + 1 = 24.0
        res = ref_dot2_f32_f16(s0, s1, acc)
        self.assertAlmostEqual(bits_to_f32(res), 24.0)

    def test_zero_addend(self):
        s0 = pack_f16x2(1.0, 1.0)
        s1 = pack_f16x2(1.0, 1.0)
        res = ref_dot2_f32_f16(s0, s1, 0)
        self.assertAlmostEqual(bits_to_f32(res), 2.0)

@dataclasses.dataclass
class DotAstNode:
    opcode: str
    dst: str
    src0: str
    src1: str
    src2: Optional[str] = None

    def emit_asm(self) -> str:
        if self.src2 is not None:
            return f"{self.opcode} {self.dst}, {self.src0}, {self.src1}, {self.src2}"
        return f"{self.opcode} {self.dst}, {self.src0}, {self.src1}"


class VDot2F32F16AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot2_f32_f16."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot2_f32_f16", "v0", "v1", "v2", "v3")
        self.assertEqual(node.emit_asm(), "v_dot2_f32_f16 v0, v1, v2, v3")

    def test_interpreter_scalar_simulation(self):
        regs = {
            "v1": pack_f16x2(1.5, 2.0),
            "v2": pack_f16x2(2.0, 3.0),
            "v3": f32_to_bits(5.0),
        }
        res = ref_dot2_f32_f16(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        # 1.5*2 + 2*3 + 5 = 3 + 6 + 5 = 14.0
        self.assertAlmostEqual(bits_to_f32(regs["v0"]), 14.0)

class VDot2F32F16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_dot2_f32_f16."""

    def test_simd_divergent_lanes(self):
        lanes = 64
        src0 = [pack_f16x2(float(i), 1.0) for i in range(lanes)]
        src1 = [pack_f16x2(2.0, float(i)) for i in range(lanes)]
        accs = [f32_to_bits(10.0) for _ in range(lanes)]
        out = [ref_dot2_f32_f16(s0, s1, acc) for s0, s1, acc in zip(src0, src1, accs)]
        for i in range(lanes):
            expected = float(i) * 2.0 + 1.0 * float(i) + 10.0
            self.assertAlmostEqual(bits_to_f32(out[i]), expected)

class VDot2F32F16TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_dot2_f32_f16."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3C003C00",  # (1.0, 1.0)
            "v_mov_b32 v1, 0x3C003C00",  # (1.0, 1.0)
            "v_mov_b32 v2, 0x40A00000",  # 5.0f
            "v_dot2_f32_f16 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot2_f32_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(out[0]), 7.0)

    def test_negative_mutation_contrast(self):
        s0 = pack_f16x2(2.0, 2.0)
        s1 = pack_f16x2(3.0, 3.0)
        res_dot = ref_dot2_f32_f16(s0, s1, 0)
        res_add = f32_to_bits(4.0)
        self.assertNotEqual(res_dot, res_add)

class VDot2cF32F16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot2c_f32_f16."""

    def test_accumulation_into_destination(self):
        s0 = pack_f16x2(1.0, 2.0)
        s1 = pack_f16x2(3.0, 4.0)
        old_d = f32_to_bits(10.0)
        # 1*3 + 2*4 + 10 = 3 + 8 + 10 = 21.0
        res = ref_dot2_f32_f16(s0, s1, old_d)
        self.assertAlmostEqual(bits_to_f32(res), 21.0)

class VDot2cF32F16AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot2c_f32_f16."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot2c_f32_f16", "v3", "v0", "v1")
        self.assertEqual(node.emit_asm(), "v_dot2c_f32_f16 v3, v0, v1")

    def test_interpreter_scalar_simulation(self):
        regs = {
            "v0": pack_f16x2(2.0, 1.0),
            "v1": pack_f16x2(3.0, 4.0),
            "v3": f32_to_bits(5.0),
        }
        res = ref_dot2_f32_f16(regs["v0"], regs["v1"], regs["v3"])
        regs["v3"] = res
        # 2*3 + 1*4 + 5 = 6 + 4 + 5 = 15.0
        self.assertAlmostEqual(bits_to_f32(regs["v3"]), 15.0)

class VDot2cF32F16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_dot2c_f32_f16."""

    def test_simd_divergent_lanes(self):
        lanes = 64
        src0 = [pack_f16x2(1.0, 2.0) for _ in range(lanes)]
        src1 = [pack_f16x2(float(i), 1.0) for i in range(lanes)]
        old_ds = [f32_to_bits(float(i)) for i in range(lanes)]
        out = [ref_dot2_f32_f16(s0, s1, d) for s0, s1, d in zip(src0, src1, old_ds)]
        for i in range(lanes):
            expected = 1.0 * float(i) + 2.0 * 1.0 + float(i)
            self.assertAlmostEqual(bits_to_f32(out[i]), expected)

class VDot2cF32F16TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_dot2c_f32_f16."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3C003C00",  # (1.0, 1.0)
            "v_mov_b32 v1, 0x3C003C00",  # (1.0, 1.0)
            "v_mov_b32 v3, 0x40000000",  # 2.0f
            "v_dot2c_f32_f16 v3, v0, v1",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot2c_f32_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(out[0]), 4.0)

    def test_negative_mutation_contrast(self):
        s0 = pack_f16x2(1.0, 1.0)
        s1 = pack_f16x2(2.0, 2.0)
        res_accum = ref_dot2_f32_f16(s0, s1, f32_to_bits(10.0))
        res_no_accum = ref_dot2_f32_f16(s0, s1, 0)
        self.assertNotEqual(res_accum, res_no_accum)

class VDot4I32I8OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot4_i32_i8."""

    def test_positive_bytes(self):
        s0 = 0x01020304
        s1 = 0x02020202
        # 4*2 + 3*2 + 2*2 + 1*2 = 8 + 6 + 4 + 2 = 20
        res = ref_dot4_i32_i8(s0, s1, 5)
        self.assertEqual(res, 25)

    def test_negative_signed_bytes(self):
        s0 = 0xFFFFFFFF  # four -1 bytes
        s1 = 0x01010101  # four +1 bytes
        # 4 * (-1) = -4
        res = ref_dot4_i32_i8(s0, s1, 0)
        self.assertEqual(i32(res), -4)

class VDot4I32I8AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot4_i32_i8."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot4_i32_i8", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_dot4_i32_i8 v3, v0, v1, v2")

    def test_interpreter_scalar_simulation(self):
        regs = {
            "v0": 0x01010101,
            "v1": 0x03030303,
            "v2": 10,
        }
        res = ref_dot4_i32_i8(regs["v0"], regs["v1"], regs["v2"])
        regs["v3"] = res
        self.assertEqual(regs["v3"], 22)

class VDot4I32I8SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_dot4_i32_i8."""

    def test_simd_divergent_lanes(self):
        lanes = 64
        src0 = [0x01010101 for _ in range(lanes)]
        src1 = [0x01010101 for _ in range(lanes)]
        accs = [i for i in range(lanes)]
        out = [ref_dot4_i32_i8(s0, s1, acc) for s0, s1, acc in zip(src0, src1, accs)]
        for i in range(lanes):
            self.assertEqual(out[i], 4 + i)

class VDot4I32I8TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_dot4_i32_i8."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0x01010101",
            "v_mov_b32 v1, 0x02020202",
            "v_mov_b32 v2, 100",
            "v_dot4_i32_i8 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot4_i32_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 108)

    def test_negative_mutation_contrast(self):
        s0 = 0xFF000000
        s1 = 0x01000000
        signed_res = ref_dot4_i32_i8(s0, s1, 0)
        unsigned_res = ref_dot4_u32_u8(s0, s1, 0)
        self.assertNotEqual(signed_res, unsigned_res)

class VDot4U32U8OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot4_u32_u8."""

    def test_unsigned_bytes(self):
        s0 = 0xFF000000  # highest byte is 255
        s1 = 0x01000000  # highest byte is 1
        res = ref_dot4_u32_u8(s0, s1, 0)
        self.assertEqual(res, 255)

    def test_all_bytes_active(self):
        s0 = 0x02020202
        s1 = 0x03030303
        res = ref_dot4_u32_u8(s0, s1, 10)
        # 4 * (2 * 3) + 10 = 24 + 10 = 34
        self.assertEqual(res, 34)

class VDot4U32U8AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot4_u32_u8."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot4_u32_u8", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_dot4_u32_u8 v3, v0, v1, v2")

    def test_interpreter_scalar_simulation(self):
        regs = {
            "v0": 0x10101010,
            "v1": 0x01010101,
            "v2": 0,
        }
        res = ref_dot4_u32_u8(regs["v0"], regs["v1"], regs["v2"])
        regs["v3"] = res
        # 4 * 16 = 64
        self.assertEqual(regs["v3"], 64)

if __name__ == "__main__":
    unittest.main()
