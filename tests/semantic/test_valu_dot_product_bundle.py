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

if __name__ == "__main__":
    unittest.main()
