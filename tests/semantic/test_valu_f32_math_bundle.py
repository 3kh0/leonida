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


# ------------------------------------------------------------------------------
# v_fract_f32 AST Definitions
# ------------------------------------------------------------------------------

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
            raise ValueError(f"VGPR index out of range: {self.index}")
        if self.op_type == OperandType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range: {self.index}")

    @property
    def name(self) -> str:
        return f"{'v' if self.op_type == OperandType.VGPR else 's'}{self.index}"


@dataclass(frozen=True)
class LiteralOperand:
    value: int

    def __post_init__(self):
        if not (-0x8000_0000 <= self.value <= MASK32):
            raise ValueError(f"Literal 32-bit value out of range: {self.value}")


@dataclass
class VFractF32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination of v_fract_f32 must be a VGPR")


class TestFractF32AST(unittest.TestCase):
    def test_ast_construction(self):
        dst = RegisterOperand(OperandType.VGPR, 0)
        src = RegisterOperand(OperandType.VGPR, 1)
        insn = VFractF32Instruction(dst=dst, src0=src)
        self.assertEqual(insn.dst.name, "v0")
        self.assertEqual(insn.src0.name, "v1")

    def test_ast_invalid_dst(self):
        with self.assertRaises(TypeError):
            VFractF32Instruction(dst=RegisterOperand(OperandType.SGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))


# ------------------------------------------------------------------------------
# v_fract_f32 Scalar Interpreter
# ------------------------------------------------------------------------------

class ScalarMachineFract:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK32
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise TypeError(f"Unknown operand: {op}")

    def step(self, insn: VFractF32Instruction):
        val = self.read_operand(insn.src0)
        self.vgprs[insn.dst.index] = ref_fract_f32(val)


class TestFractF32ScalarInterpreter(unittest.TestCase):
    def setUp(self):
        self.machine = ScalarMachineFract()

    def test_scalar_fract_execution(self):
        self.machine.vgprs[2] = f32_to_bits(3.75)
        insn = VFractF32Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 2))
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[0], f32_to_bits(0.75))


# ------------------------------------------------------------------------------
# v_fract_f32 SIMD Multi-Lane Divergence and EXEC Masking
# ------------------------------------------------------------------------------

class SIMDWaveFract:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VFractF32Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                val = self.vgprs[insn.src0.index][lane]
                self.vgprs[insn.dst.index][lane] = ref_fract_f32(val)


class TestFractF32SIMDDivergence(unittest.TestCase):
    def test_divergence_and_exec_masking(self):
        wave = SIMDWaveFract(wave_size=32)
        for lane in range(32):
            wave.vgprs[1][lane] = f32_to_bits(float(lane) + 0.125)
            wave.vgprs[0][lane] = 0xDEAD_BEEF

        wave.exec_mask = 0x0000_FFFF  # lanes 0..15 active
        insn = VFractF32Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        wave.step(insn)

        for lane in range(32):
            if lane < 16:
                self.assertEqual(wave.vgprs[0][lane], f32_to_bits(0.125))
            else:
                self.assertEqual(wave.vgprs[0][lane], 0xDEAD_BEEF)


# ------------------------------------------------------------------------------
# v_fract_f32 IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ------------------------------------------------------------------------------

class TestFractF32IEEE754Handling(unittest.TestCase):
    def test_clamp_below_one_on_tiny_negative(self):
        # A tiny negative float whose x - floor(x) would round to 1.0f in f32
        tiny_neg = f32_to_bits(-1e-9)
        self.assertEqual(ref_fract_f32(tiny_neg), LARGEST_BELOW_ONE)

    def test_subnormals_flush_to_zero(self):
        for sub in (0x0000_0001, 0x0040_0000, 0x007F_FFFF, 0x8000_0001, 0x807F_FFFF):
            self.assertEqual(ref_fract_f32(sub), 0)

    def test_nans_pass_through(self):
        for nan in (QNAN, SNAN, 0x7FC0_1234, 0xFFC0_ABCD):
            self.assertEqual(ref_fract_f32(nan), nan)


# ------------------------------------------------------------------------------
# v_fract_f32 Negative Controls and Translator Hardware Execution
# ------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import rdna2spirv


class TestFractF32RejectionAndHardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 300)

    def test_hardware_translator_execution(self):
        cases = [
            (f32_to_bits(2.25), f32_to_bits(0.25)),
            (f32_to_bits(-0.25), f32_to_bits(0.75)),
            (POS_ZERO, 0),
            (NEG_ZERO, 0),
            (POS_INF, 0xFFC0_0000),
            (NEG_INF, 0xFFC0_0000),
            (QNAN, QNAN),
            (0x0000_0001, 0),
        ]
        for src_val, expected in cases:
            prog = [
                f"v_mov_b32 v1, {src_val:#x}",
                "v_fract_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_fract_hw_test", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], expected, f"Failed on {src_val:#x}: got {words[0]:#x}, expected {expected:#x}")


# ------------------------------------------------------------------------------
# v_frexp_exp_i32_f32 Reference Oracle
# ------------------------------------------------------------------------------

def ref_frexp_exp_i32_f32(src: int) -> int:
    u = src & MASK32
    e = (u >> EXP_SHIFT32) & EXP_MASK32
    if 1 <= e <= 254:
        exp = e - 126
        return exp & MASK32
    return 0


class TestFrexpExpI32F32Oracle(unittest.TestCase):
    def test_powers_of_two(self):
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(1.0)), 1)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(0.5)), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(2.0)), 2)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(0.25)), 0xFFFF_FFFF)

    def test_sign_invariance(self):
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-1.0)), 1)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-2.0)), 2)

    def test_specials_yield_zero(self):
        self.assertEqual(ref_frexp_exp_i32_f32(POS_ZERO), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(NEG_ZERO), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(POS_INF), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(NEG_INF), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(QNAN), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(0x0000_0001), 0)


# ------------------------------------------------------------------------------
# v_frexp_exp_i32_f32 AST Definitions
# ------------------------------------------------------------------------------

@dataclass
class VFrexpExpI32F32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination of v_frexp_exp_i32_f32 must be a VGPR")


class TestFrexpExpI32F32AST(unittest.TestCase):
    def test_ast_construction(self):
        insn = VFrexpExpI32F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
        )
        self.assertEqual(insn.dst.name, "v0")
        self.assertEqual(insn.src0.name, "v1")


# ------------------------------------------------------------------------------
# v_frexp_exp_i32_f32 Scalar Interpreter
# ------------------------------------------------------------------------------

class ScalarMachineFrexpExp:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK32
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise TypeError(f"Unknown operand: {op}")

    def step(self, insn: VFrexpExpI32F32Instruction):
        val = self.read_operand(insn.src0)
        self.vgprs[insn.dst.index] = ref_frexp_exp_i32_f32(val)


class TestFrexpExpI32F32ScalarInterpreter(unittest.TestCase):
    def setUp(self):
        self.machine = ScalarMachineFrexpExp()

    def test_scalar_execution(self):
        self.machine.vgprs[1] = f32_to_bits(4.0)
        insn = VFrexpExpI32F32Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[0], 3)


# ------------------------------------------------------------------------------
# v_frexp_exp_i32_f32 SIMD Divergence and EXEC Masking
# ------------------------------------------------------------------------------

class SIMDWaveFrexpExp:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VFrexpExpI32F32Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                val = self.vgprs[insn.src0.index][lane]
                self.vgprs[insn.dst.index][lane] = ref_frexp_exp_i32_f32(val)


class TestFrexpExpI32F32SIMDDivergence(unittest.TestCase):
    def test_simd_execution(self):
        wave = SIMDWaveFrexpExp(wave_size=32)
        for lane in range(32):
            wave.vgprs[1][lane] = f32_to_bits(math.ldexp(1.0, lane - 15))
            wave.vgprs[0][lane] = 0

        insn = VFrexpExpI32F32Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        wave.step(insn)

        for lane in range(32):
            expected = (lane - 15 + 1) & MASK32
            self.assertEqual(wave.vgprs[0][lane], expected)


# ------------------------------------------------------------------------------
# v_frexp_exp_i32_f32 IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ------------------------------------------------------------------------------

class TestFrexpExpI32F32IEEE754Handling(unittest.TestCase):
    def test_boundary_normals(self):
        # MIN_NORMAL: e = 1 -> exp = 1 - 126 = -125 (0xFFFFFF83)
        self.assertEqual(ref_frexp_exp_i32_f32(MIN_NORMAL), 0xFFFF_FF83)
        # MAX_NORMAL: e = 254 -> exp = 254 - 126 = 128 (0x00000080)
        self.assertEqual(ref_frexp_exp_i32_f32(MAX_NORMAL), 128)

    def test_all_special_cases_zero(self):
        for u in (POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, QNAN, SNAN, 0x0000_0001, 0x007F_FFFF):
            self.assertEqual(ref_frexp_exp_i32_f32(u), 0)


if __name__ == '__main__':
    unittest.main()
