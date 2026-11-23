from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_sub_f16.

Instruction definition:
    v_sub_f16 vdst, vsrc0, vsrc1
    v_sub_f16_e64 vdst, vsrc0, vsrc1 [clamp] [negate] [abs]

Semantics:
    Half-precision floating-point subtraction (RDNA2 VOP2 / VOP3).
    D.f16 = S0.f16 - S1.f16
    Binary16 IEEE 754 arithmetic with round to nearest even.
    The instruction writes the low 16 bits of the destination VGPR and preserves
    the high 16 bits of the destination VGPR:
        D[15:0] = S0.f16 - S1.f16
        D[31:16] = D_old[31:16]

Standard library only: struct, math, fractions, unittest. Zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
import math
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF
OLD: int = 0xDEAD_BEEF

# IEEE 754 Binary16 Constants
POS_ZERO: int = 0x0000
NEG_ZERO: int = 0x8000
ONE: int = 0x3C00
TWO: int = 0x4000
THREE: int = 0x4200
HALF: int = 0x3800
QUARTER: int = 0x3400
NEG_ONE: int = 0xBC00
POS_INF: int = 0x7C00
NEG_INF: int = 0xFC00
QNAN: int = 0x7E00
SNAN: int = 0x7C01
MIN_NORMAL: int = 0x0400
MAX_NORMAL: int = 0x7BFF
MIN_SUB: int = 0x0001
MAX_SUB: int = 0x03FF


# ==============================================================================
# Bit Manipulation and Half-Precision Floating-Point Helpers
# ==============================================================================

def f16_value(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & MASK16))[0]


def is_nan(h: int) -> bool:
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def is_inf(h: int) -> bool:
    return (h & 0x7FFF) == 0x7C00


def same_f16(a: int, b: int) -> bool:
    return a == b or (is_nan(a) and is_nan(b))


def round_f16(x: Fraction) -> int:
    sign = 0x8000 if x < 0 else 0
    x = abs(x)
    if x == 0:
        return 0
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    q = Fraction(2) ** (max(e, -14) - 10)
    m, rem = divmod(x, q)
    m = int(m)
    if rem > q / 2 or (rem == q / 2 and m & 1):
        m += 1
    if m < 0x400:
        return sign | m
    exp = max(e, -14)
    while m >= 0x800:
        m >>= 1
        exp += 1
    bits = ((exp + 15) << 10) | (m - 0x400)
    return sign | (0x7C00 if bits >= 0x7C00 else bits)


def to_f16(v: Union[float, Fraction]) -> int:
    if isinstance(v, float):
        if math.isnan(v):
            return QNAN
        if math.isinf(v):
            return POS_INF if v > 0 else NEG_INF
        v = Fraction(v)
    return round_f16(v)


def ref_add(a: int, b: int) -> int:
    a &= MASK16
    b &= MASK16
    if (a & 0x7C00) == 0x7C00 or (b & 0x7C00) == 0x7C00:
        return to_f16(f16_value(a) + f16_value(b))
    if a == NEG_ZERO and b == NEG_ZERO:
        return NEG_ZERO
    return round_f16(Fraction(f16_value(a)) + Fraction(f16_value(b)))


def ref_sub(a: int, b: int) -> int:
    a &= MASK16
    b &= MASK16
    return ref_add(a, b ^ 0x8000) if not is_nan(b) else ref_add(a, b)


def clamp01(h: int) -> int:
    if is_nan(h):
        return POS_ZERO
    v = f16_value(h)
    return ONE if v > 1.0 else (POS_ZERO if v < 0.0 or h == NEG_ZERO else h)


def ref_v_sub_f16(
    src0: int,
    src1: int,
    clamp: bool = False,
    abs_s0: bool = False,
    abs_s1: bool = False,
    neg_s0: bool = False,
    neg_s1: bool = False,
    old_dst: int = OLD,
) -> int:
    """Architectural reference oracle for v_sub_f16 (VOP2/VOP3)."""
    s0 = src0 & MASK16
    s1 = src1 & MASK16

    if abs_s0:
        s0 &= 0x7FFF
    if abs_s1:
        s1 &= 0x7FFF
    if neg_s0:
        s0 ^= 0x8000
    if neg_s1:
        s1 ^= 0x8000

    res = ref_sub(s0, s1)
    if clamp:
        res = clamp01(res)

    # Destination low 16 bits get result; high 16 bits keep old_dst
    return ((old_dst & 0xFFFF_0000) | (res & MASK16)) & MASK32


@dataclass
class VGPRWord:
    raw: int = 0

    @property
    def u32(self) -> int:
        return self.raw & MASK32

    @property
    def lo(self) -> int:
        return self.raw & MASK16

    @property
    def hi(self) -> int:
        return (self.raw >> 16) & MASK16


class TestSubF16ReferenceOracle(unittest.TestCase):
    def test_standard_subtractions(self):
        # 3 - 1 = 2
        self.assertEqual(ref_sub(THREE, ONE), TWO)
        # 1 - 2 = -1
        self.assertEqual(ref_sub(ONE, TWO), NEG_ONE)
        # 0.5 - 0.25 = 0.25
        self.assertEqual(ref_sub(HALF, QUARTER), QUARTER)

    def test_signed_zeros(self):
        # x - x = +0
        self.assertEqual(ref_sub(0x3E00, 0x3E00), POS_ZERO)
        # -0 - +0 = -0
        self.assertEqual(ref_sub(NEG_ZERO, POS_ZERO), NEG_ZERO)
        # +0 - +0 = +0
        self.assertEqual(ref_sub(POS_ZERO, POS_ZERO), POS_ZERO)
        # +0 - -0 = +0
        self.assertEqual(ref_sub(POS_ZERO, NEG_ZERO), POS_ZERO)
        # -0 - -0 = +0
        self.assertEqual(ref_sub(NEG_ZERO, NEG_ZERO), POS_ZERO)

    def test_infinities_and_nans(self):
        # inf - inf = NaN
        self.assertTrue(is_nan(ref_sub(POS_INF, POS_INF)))
        # inf - -inf = inf
        self.assertEqual(ref_sub(POS_INF, NEG_INF), POS_INF)
        # 1 - NaN = NaN
        self.assertTrue(is_nan(ref_sub(ONE, QNAN)))

    def test_high_half_preservation(self):
        got = ref_v_sub_f16(THREE, ONE, old_dst=0xDEADBEEF)
        self.assertEqual(got >> 16, 0xDEAD)
        self.assertEqual(got & 0xFFFF, TWO)

    def test_clamp_modifier(self):
        # 3 - 1 = 2 -> clamped to 1.0 (0x3C00)
        got = ref_v_sub_f16(THREE, ONE, clamp=True, old_dst=0x12340000)
        self.assertEqual(got & 0xFFFF, ONE)
        # 1 - 2 = -1 -> clamped to 0.0 (0x0000)
        got2 = ref_v_sub_f16(ONE, TWO, clamp=True, old_dst=0x12340000)
        self.assertEqual(got2 & 0xFFFF, POS_ZERO)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int

    def __post_init__(self):
        if self.op_type == OperandType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.op_type == OperandType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    @property
    def name(self) -> str:
        prefix = "v" if self.op_type == OperandType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class LiteralOperand:
    value: int

    def __post_init__(self):
        if not (0 <= self.value <= MASK32):
            raise ValueError(f"Literal 32-bit value out of range: {self.value}")


@dataclass(frozen=True)
class VOPModifiers:
    clamp: bool = False
    abs_s0: bool = False
    abs_s1: bool = False
    neg_s0: bool = False
    neg_s1: bool = False


@dataclass
class VSubF16Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]
    modifiers: VOPModifiers = field(default_factory=VOPModifiers)

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination of v_sub_f16 must be a VGPR")


class TestSubF16AST(unittest.TestCase):
    def test_valid_ast_construction(self):
        dst = RegisterOperand(OperandType.VGPR, 0)
        s0 = RegisterOperand(OperandType.VGPR, 1)
        s1 = RegisterOperand(OperandType.VGPR, 2)
        mods = VOPModifiers(clamp=True, abs_s0=True)
        insn = VSubF16Instruction(dst=dst, src0=s0, src1=s1, modifiers=mods)
        self.assertEqual(insn.dst.name, "v0")
        self.assertEqual(insn.src0.name, "v1")
        self.assertEqual(insn.src1.name, "v2")
        self.assertTrue(insn.modifiers.clamp)
        self.assertTrue(insn.modifiers.abs_s0)

    def test_invalid_register_indices(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 256)
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.SGPR, 106)

    def test_invalid_dst_type(self):
        with self.assertRaises(TypeError):
            VSubF16Instruction(
                dst=RegisterOperand(OperandType.SGPR, 0),
                src0=RegisterOperand(OperandType.VGPR, 1),
                src1=RegisterOperand(OperandType.VGPR, 2),
            )

    def test_literal_operand_bounds(self):
        with self.assertRaises(ValueError):
            LiteralOperand(-1)
        with self.assertRaises(ValueError):
            LiteralOperand(0x1_0000_0000)


if __name__ == '__main__':
    unittest.main()
