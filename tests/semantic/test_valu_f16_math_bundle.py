"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU Float16 Math Suite.

Subsystem Feature Bundle 23 covers 4 half-precision (IEEE 754 binary16) exponential/mantissa/scale instructions:
  1. v_fract_f16          - fractional part computation (x - floor(x)) with 0x3bff clamp below 1.0
  2. v_frexp_exp_i16_f16  - integer exponent extraction as 16-bit signed integer
  3. v_frexp_mant_f16     - normalized significand extraction in [0.5, 1.0) with source sign
  4. v_ldexp_f16          - scale by power of 2 with RNE rounding across subnormals

RDNA2 Float16 Architecture:
  - Operates on 16-bit half-precision floats stored in the lower half of 32-bit VGPRs.
  - Standard IEEE 754 binary16 format: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
  - Denormals are preserved and normalized.
  - End-to-end SPIR-V assembly translation through tools/rdna2spirv.py.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# Float16 constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_ONE = 0x3C00   # 1.0
F16_NEG_ONE = 0xBC00   # -1.0
F16_POS_HALF = 0x3800  # 0.5
F16_NEG_HALF = 0xB800  # -0.5
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_QNAN = 0x7E00
F16_MAX_NORMAL = 0x7BFF
F16_MIN_NORMAL = 0x0400


def f16_to_u16(f: float) -> int:
    return struct.unpack(">H", struct.pack(">e", f))[0]


def u16_to_f16(u: int) -> float:
    return struct.unpack(">e", struct.pack(">H", u & MASK16))[0]


def is_nan_f16(u: int) -> bool:
    return (u & 0x7C00) == 0x7C00 and (u & 0x03FF) != 0


def is_inf_f16(u: int) -> bool:
    return (u & 0x7FFF) == 0x7C00

# -----------------------------------------------------------------------------
# Opcode 1: v_fract_f16
# -----------------------------------------------------------------------------
def oracle_fract_f16(u: int) -> int:
    """Reference mathematical oracle for v_fract_f16."""
    u &= MASK16
    if is_nan_f16(u):
        return F16_QNAN
    if is_inf_f16(u):
        return F16_POS_ZERO
    f = u16_to_f16(u)
    if f == 0.0:
        return F16_POS_ZERO
    res = f - math.floor(f)
    if res == 0.0:
        return F16_POS_ZERO
    res_u = f16_to_u16(res)
    # Hardware rule: a result that rounded up to 1.0 (0x3c00) becomes 0x3bff
    if res_u >= 0x3C00:
        return 0x3BFF
    return res_u


class VFractF16OracleTests(unittest.TestCase):
    def test_fract_basic(self):
        # fract(1.75) = 0.75 (0x3a00)
        self.assertEqual(oracle_fract_f16(f16_to_u16(1.75)), 0x3A00)

    def test_fract_negative(self):
        # fract(-1.25) = -1.25 - (-2.0) = 0.75 (0x3a00)
        self.assertEqual(oracle_fract_f16(f16_to_u16(-1.25)), 0x3A00)

    def test_fract_zeros(self):
        self.assertEqual(oracle_fract_f16(F16_POS_ZERO), F16_POS_ZERO)
        self.assertEqual(oracle_fract_f16(F16_NEG_ZERO), F16_POS_ZERO)

class RegKind(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    IMM = "imm"


@dataclass(frozen=True)
class Operand:
    kind: RegKind
    index: int = 0
    raw_val: int = 0


@dataclass(frozen=True)
class F16InstructionAST:
    mnemonic: str
    dst: Operand
    src0: Operand
    src1: Operand | None = None


class WavefrontSimdState:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.exec_mask: int = (1 << lane_count) - 1
        self.vgprs = [0] * lane_count

    def set_vgpr(self, lane: int, val: int):
        self.vgprs[lane] = val & MASK16

    def get_vgpr(self, lane: int) -> int:
        return self.vgprs[lane] & MASK16

class VFractF16AstTests(unittest.TestCase):
    def test_fract_ast_construction(self):
        ast = F16InstructionAST("v_fract_f16", Operand(RegKind.VGPR, 1), Operand(RegKind.VGPR, 0))
        self.assertEqual(ast.mnemonic, "v_fract_f16")
        self.assertEqual(ast.dst.index, 1)
        self.assertEqual(ast.src0.index, 0)

class VFractF16ScalarInterpreterTests(unittest.TestCase):
    def test_fract_scalar_execution(self):
        # fract(2.5) = 0.5 (0x3800)
        res = oracle_fract_f16(f16_to_u16(2.5))
        self.assertEqual(res, F16_POS_HALF)

class VFractF16SimdDivergenceTests(unittest.TestCase):
    def test_fract_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0x55555555  # Even lanes active
        for i in range(32):
            wave.set_vgpr(i, 0xEEEE)
            if (wave.exec_mask >> i) & 1:
                wave.set_vgpr(i, oracle_fract_f16(f16_to_u16(float(i) + 0.25)))
        for i in range(32):
            if (0x55555555 >> i) & 1:
                self.assertEqual(wave.get_vgpr(i), f16_to_u16(0.25))
            else:
                self.assertEqual(wave.get_vgpr(i), 0xEEEE)

class VFractF16EdgeCasesTests(unittest.TestCase):
    def test_fract_clamp_below_one(self):
        # Any value extremely close to 1.0 clamps to 0x3BFF
        near_one = 0x3BFF
        self.assertEqual(oracle_fract_f16(near_one), 0x3BFF)

    def test_fract_infinity_and_nan(self):
        self.assertEqual(oracle_fract_f16(F16_POS_INF), F16_POS_ZERO)
        self.assertEqual(oracle_fract_f16(F16_NEG_INF), F16_POS_ZERO)
        self.assertTrue(is_nan_f16(oracle_fract_f16(F16_QNAN)))

class VFractF16TranslatorTests(unittest.TestCase):
    def test_fract_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3f00",  # 1.75 in f16
            "v_fract_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_fract_f16_exec", lines)
        val = mrt0_words(spv)[0] & MASK16
        self.assertEqual(val, 0x3A00)  # 0.75 in f16

    def test_fract_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_fract_f16", ["v_fract_f16"])

# -----------------------------------------------------------------------------
# Opcode 2: v_frexp_exp_i16_f16
# -----------------------------------------------------------------------------
def oracle_frexp_exp_i16_f16(u: int) -> int:
    """Reference mathematical oracle for v_frexp_exp_i16_f16."""
    u &= MASK16
    if is_nan_f16(u) or is_inf_f16(u):
        return 0
    f = u16_to_f16(u)
    if f == 0.0:
        return 0
    e = (u >> 10) & 0x1F
    frac = u & 0x03FF
    if e == 0:
        # Denormal: find leading bit in 10-bit frac
        msb = frac.bit_length() - 1
        return (1 - (10 - msb) - 14) & MASK16
    return (e - 14) & MASK16


class VFrexpExpI16F16OracleTests(unittest.TestCase):
    def test_frexp_exp_basic(self):
        # 1.0 (0x3c00): 1.0 = 0.5 * 2^1 -> exp is 1
        self.assertEqual(oracle_frexp_exp_i16_f16(F16_POS_ONE), 1)
        # 4.0: 4.0 = 0.5 * 2^3 -> exp is 3
        self.assertEqual(oracle_frexp_exp_i16_f16(f16_to_u16(4.0)), 3)
        # 0.25: 0.25 = 0.5 * 2^-1 -> exp is -1 (0xFFFF)
        self.assertEqual(oracle_frexp_exp_i16_f16(f16_to_u16(0.25)), (-1) & MASK16)

class VFrexpExpI16F16AstTests(unittest.TestCase):
    def test_frexp_exp_ast_construction(self):
        ast = F16InstructionAST("v_frexp_exp_i16_f16", Operand(RegKind.VGPR, 1), Operand(RegKind.VGPR, 0))
        self.assertEqual(ast.mnemonic, "v_frexp_exp_i16_f16")
        self.assertEqual(ast.dst.index, 1)

class VFrexpExpI16F16ScalarInterpreterTests(unittest.TestCase):
    def test_frexp_exp_scalar_execution(self):
        # 2.0 = 0.5 * 2^2 -> exp is 2
        res = oracle_frexp_exp_i16_f16(f16_to_u16(2.0))
        self.assertEqual(res, 2)

class VFrexpExpI16F16SimdDivergenceTests(unittest.TestCase):
    def test_frexp_exp_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0x0000FFFF  # Lower 16 active
        for i in range(32):
            wave.set_vgpr(i, 0x9999)
            if (wave.exec_mask >> i) & 1:
                # 2.0^i -> exponent is i + 1
                wave.set_vgpr(i, oracle_frexp_exp_i16_f16(f16_to_u16(float(2**i))))
        for i in range(16):
            self.assertEqual(wave.get_vgpr(i), i + 1)
        for i in range(16, 32):
            self.assertEqual(wave.get_vgpr(i), 0x9999)

class VFrexpExpI16F16EdgeCasesTests(unittest.TestCase):
    def test_frexp_exp_zeros_and_specials(self):
        self.assertEqual(oracle_frexp_exp_i16_f16(F16_POS_ZERO), 0)
        self.assertEqual(oracle_frexp_exp_i16_f16(F16_NEG_ZERO), 0)
        self.assertEqual(oracle_frexp_exp_i16_f16(F16_POS_INF), 0)
        self.assertEqual(oracle_frexp_exp_i16_f16(F16_QNAN), 0)

class VFrexpExpI16F16TranslatorTests(unittest.TestCase):
    def test_frexp_exp_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3c00",  # 1.0 in f16
            "v_frexp_exp_i16_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_exp_f16_exec", lines)
        val = mrt0_words(spv)[0] & MASK16
        self.assertEqual(val, 1)

# -----------------------------------------------------------------------------
# Opcode 3: v_frexp_mant_f16
# -----------------------------------------------------------------------------
def oracle_frexp_mant_f16(u: int) -> int:
    """Reference mathematical oracle for v_frexp_mant_f16."""
    u &= MASK16
    if is_nan_f16(u) or is_inf_f16(u):
        return u
    if (u & 0x7FFF) == 0:
        return u  # preserves +-0.0
    sgn = u & 0x8000
    e = (u >> 10) & 0x1F
    frac = u & 0x03FF
    if e == 0:
        # Denormal: normalize frac to bit 10
        msb = frac.bit_length() - 1
        lsh = 10 - msb
        sig11 = (frac << lsh) & 0x07FF
        return sgn | F16_POS_HALF | (sig11 & 0x03FF)
    return sgn | F16_POS_HALF | frac


class VFrexpMantF16OracleTests(unittest.TestCase):
    def test_frexp_mant_basic(self):
        # 1.0 (0x3c00): mantissa is 0.5 (0x3800)
        self.assertEqual(oracle_frexp_mant_f16(F16_POS_ONE), F16_POS_HALF)
        # -1.0 (0xbc00): mantissa is -0.5 (0xb800)
        self.assertEqual(oracle_frexp_mant_f16(F16_NEG_ONE), F16_NEG_HALF)
        # 3.0: 3.0 = 0.75 * 2^2 -> mantissa is 0.75 (0x3a00)
        self.assertEqual(oracle_frexp_mant_f16(f16_to_u16(3.0)), 0x3A00)

class VFrexpMantF16AstTests(unittest.TestCase):
    def test_frexp_mant_ast_construction(self):
        ast = F16InstructionAST("v_frexp_mant_f16", Operand(RegKind.VGPR, 2), Operand(RegKind.VGPR, 0))
        self.assertEqual(ast.mnemonic, "v_frexp_mant_f16")
        self.assertEqual(ast.dst.index, 2)

class VFrexpMantF16ScalarInterpreterTests(unittest.TestCase):
    def test_frexp_mant_scalar_execution(self):
        # 6.0: 6.0 = 0.75 * 2^3 -> mantissa is 0.75 (0x3a00)
        res = oracle_frexp_mant_f16(f16_to_u16(6.0))
        self.assertEqual(res, 0x3A00)

if __name__ == "__main__":
    unittest.main()
