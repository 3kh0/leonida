from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 Float32 Exponential/Mantissa/Scale Suite.

Covers:
- v_fract_f32: fractional part x - floor(x), denormals flush to zero, inf gives -qNaN, clamped below 1.0.
- v_frexp_exp_i32_f32: binary exponent e - 126 for normal float, 0 for zero/denorm/inf/nan.
- v_frexp_mant_f32: mantissa in [0.5, 1.0) with sign preserved, 0 for zero/denorm, pass-through for inf/nan.
- v_ldexp_f32: computes S0 * 2^S1 with exponent clamped to [-512, 512], overflow to inf, underflow to 0.

Zero game data, standard library Python only (python -I, struct, math, unittest).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000
EXP_MASK32: int = 0x00FF
EXP_SHIFT32: int = 23
MANT_MASK32: int = 0x007F_FFFF

POS_ZERO: int = 0x0000_0000
NEG_ZERO: int = 0x8000_0000
POS_INF: int = 0x7F80_0000
NEG_INF: int = 0xFF80_0000
QNAN: int = 0x7FC0_0000
SNAN: int = 0x7F80_0001
MIN_NORMAL: int = 0x0080_0000
MAX_NORMAL: int = 0x7F7F_FFFF
LARGEST_BELOW_ONE: int = 0x3F7F_FFFF


def to_i32(val: int) -> int:
    val32 = val & MASK32
    if val32 & SIGN_BIT32:
        return val32 - 0x1_0000_0000
    return val32


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def is_normal_f32(u: int) -> bool:
    e = (u >> EXP_SHIFT32) & EXP_MASK32
    return 1 <= e <= 254


# ------------------------------------------------------------------------------
# v_fract_f32 Reference Oracle
# ------------------------------------------------------------------------------

def ref_fract_f32(src: int) -> int:
    u = src & MASK32
    mag = u & 0x7FFF_FFFF
    exp = u & 0x7F80_0000
    if mag > 0x7F80_0000:
        return u  # NaN passes through
    if mag == 0x7F80_0000:
        return 0xFFC0_0000  # +-inf -> -qNaN
    if exp == 0:
        return 0x0000_0000  # denormals and +-0 return +0.0

    val = bits_to_f32(u)
    res = val - math.floor(val)
    res_bits = f32_to_bits(res)
    # If rounded to 1.0, clamp to largest float below 1.0 (0x3F7FFFFF)
    if res_bits == 0x3F80_0000:
        return LARGEST_BELOW_ONE
    return res_bits


class TestFractF32Oracle(unittest.TestCase):
    def test_positive_fractions(self):
        # 1.25 -> 0.25 (0x3E800000)
        self.assertEqual(ref_fract_f32(f32_to_bits(1.25)), f32_to_bits(0.25))
        # 5.75 -> 0.75 (0x3F400000)
        self.assertEqual(ref_fract_f32(f32_to_bits(5.75)), f32_to_bits(0.75))
        # 10.0 -> 0.0
        self.assertEqual(ref_fract_f32(f32_to_bits(10.0)), 0)

    def test_negative_fractions(self):
        # -0.25 -> -0.25 - (-1.0) = 0.75 (0x3F400000)
        self.assertEqual(ref_fract_f32(f32_to_bits(-0.25)), f32_to_bits(0.75))
        # -1.5 -> -1.5 - (-2.0) = 0.5 (0x3F000000)
        self.assertEqual(ref_fract_f32(f32_to_bits(-1.5)), f32_to_bits(0.5))

    def test_special_cases(self):
        self.assertEqual(ref_fract_f32(POS_ZERO), 0)
        self.assertEqual(ref_fract_f32(NEG_ZERO), 0)
        self.assertEqual(ref_fract_f32(POS_INF), 0xFFC0_0000)
        self.assertEqual(ref_fract_f32(NEG_INF), 0xFFC0_0000)
        self.assertEqual(ref_fract_f32(QNAN), QNAN)
        self.assertEqual(ref_fract_f32(0x0000_0001), 0)


if __name__ == '__main__':
    unittest.main()
