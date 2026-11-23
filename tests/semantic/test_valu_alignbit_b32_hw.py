"""Semantic reference oracle and regression tests for RDNA2 v_alignbit_b32.

RDNA2 / GFX10 VALU 32-bit Funnel Shift / Bit Alignment:
    v_alignbit_b32 vd, src0, src1, src2

Hardware Semantics:
    Concatenates {src0, src1} into a 64-bit value where src0 occupies bits [63:32]
    and src1 occupies bits [31:0]. Shifts logically right by (src2 & 31) and extracts
    the low 32 bits:
        shift = src2 & 31
        data64 = ((src0 & 0xFFFFFFFF) << 32) | (src1 & 0xFFFFFFFF)
        vd = (data64 >> shift) & 0xFFFFFFFF

Properties:
  - Masked shift count: shift is masked with 0x1F (modulo 32).
  - Shift count zero: returns src1 without shifting or undefined operations.
  - S0 == S1: 32-bit rotate right (ror).
  - S0 == 0: 32-bit logical shift right (lsr).
  - S1 == 0: 32-bit logical shift left of S0 by (32 - shift) for shift in 1..31.
  - Condition codes: Neither SCC nor VCC is modified.
  - IEEE 754: Raw bitwise operation; no denormal flushing, quietening, or NaN modification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv
    import support
except Exception:
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    return val & MASK32


def ref_alignbit_b32(src0: int, src1: int, src2: int) -> int:
    """RDNA2 v_alignbit_b32 reference mathematical oracle."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    shift = src2 & 0x1F
    if shift == 0:
        return s1
    return ((s0 << (32 - shift)) | (s1 >> shift)) & MASK32


class TestValuAlignbitB32Oracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle unit tests."""

    def test_basic_funnel_shifts(self) -> None:
        self.assertEqual(ref_alignbit_b32(0x12345678, 0x9ABCDEF0, 4), 0x89ABCDEF)
        self.assertEqual(ref_alignbit_b32(0x12345678, 0x9ABCDEF0, 8), 0x789ABCDE)
        self.assertEqual(ref_alignbit_b32(0x12345678, 0x9ABCDEF0, 16), 0x56789ABC)
        self.assertEqual(ref_alignbit_b32(0x12345678, 0x9ABCDEF0, 24), 0x3456789A)

    def test_shift_count_masking(self) -> None:
        s0, s1 = 0xDEADBEEF, 0xCAFEBABE
        self.assertEqual(ref_alignbit_b32(s0, s1, 32), ref_alignbit_b32(s0, s1, 0))
        self.assertEqual(ref_alignbit_b32(s0, s1, 33), ref_alignbit_b32(s0, s1, 1))
        self.assertEqual(ref_alignbit_b32(s0, s1, 65), ref_alignbit_b32(s0, s1, 1))
        self.assertEqual(ref_alignbit_b32(s0, s1, 0xFFFFFFFF), ref_alignbit_b32(s0, s1, 31))

    def test_zero_shift_identity(self) -> None:
        s0, s1 = 0x11223344, 0x55667788
        self.assertEqual(ref_alignbit_b32(s0, s1, 0), s1)
        self.assertEqual(ref_alignbit_b32(s0, s1, 64), s1)

    def test_rotate_right_idiom(self) -> None:
        val = 0x87654321
        for shift in range(32):
            expected = ((val >> shift) | (val << ((32 - shift) & 31))) & MASK32
            self.assertEqual(ref_alignbit_b32(val, val, shift), expected)

    def test_logical_shift_idioms(self) -> None:
        s1 = 0xFEDCBA98
        for shift in range(32):
            self.assertEqual(ref_alignbit_b32(0, s1, shift), s1 >> shift)

        s0 = 0x12345678
        for shift in range(1, 32):
            self.assertEqual(ref_alignbit_b32(s0, 0, shift), (s0 << (32 - shift)) & MASK32)

    def test_boundary_shifts(self) -> None:
        s0, s1 = 0x80000000, 0x00000002
        self.assertEqual(ref_alignbit_b32(s0, s1, 1), 0x00000001)
        s0_31, s1_31 = 1, 0x80000000
        self.assertEqual(ref_alignbit_b32(s0_31, s1_31, 31), 3)


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    IMM = auto()


@dataclass(frozen=True)
class Operand:
    kind: OperandKind
    index: int = 0
    value: int = 0

    @staticmethod
    def vgpr(idx: int) -> Operand:
        return Operand(OperandKind.VGPR, index=idx)

    @staticmethod
    def sgpr(idx: int) -> Operand:
        return Operand(OperandKind.SGPR, index=idx)

    @staticmethod
    def imm(val: int) -> Operand:
        return Operand(OperandKind.IMM, value=val & MASK32)


@dataclass(frozen=True)
class VAlignbitB32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def disasm(self) -> str:
        def fmt(op: Operand) -> str:
            if op.kind == OperandKind.VGPR:
                return f"v{op.index}"
            if op.kind == OperandKind.SGPR:
                return f"s{op.index}"
            return str(op.value)
        return f"v_alignbit_b32 {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}, {fmt(self.src2)}"


class TestValuAlignbitB32AST(unittest.TestCase):
    """Stage 2: AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.imm(8),
        )
        self.assertEqual(instr.disasm(), "v_alignbit_b32 v0, v1, v2, 8")

    def test_ast_mixed_operands(self) -> None:
        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.sgpr(2),
            src1=Operand.vgpr(10),
            src2=Operand.sgpr(4),
        )
        self.assertEqual(instr.disasm(), "v_alignbit_b32 v5, s2, v10, s4")


class ScalarMachineState:
    def __init__(self, num_vgprs: int = 256, num_sgprs: int = 106) -> None:
        self.vgprs: list[int] = [0] * num_vgprs
        self.sgprs: list[int] = [0] * num_sgprs
        self.scc: int = 0
        self.vcc: int = 0

    def read_operand(self, op: Operand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.kind == OperandKind.SGPR:
            return self.sgprs[op.index] & MASK32
        return op.value & MASK32

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32


class ScalarInterpreter:
    def step(self, instr: VAlignbitB32Instruction, state: ScalarMachineState) -> int:
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        s2 = state.read_operand(instr.src2)
        res = ref_alignbit_b32(s0, s1, s2)
        if instr.dst.kind == OperandKind.VGPR:
            state.write_vgpr(instr.dst.index, res)
        return res


class TestValuAlignbitB32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter."""

    def test_scalar_step_all_vgprs(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(1, 0x12345678)
        state.write_vgpr(2, 0x9ABCDEF0)
        state.write_vgpr(3, 8)
        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        res = ScalarInterpreter().step(instr, state)
        self.assertEqual(res, 0x789ABCDE)
        self.assertEqual(state.vgprs[0], 0x789ABCDE)

    def test_scalar_step_mixed_sgpr_imm(self) -> None:
        state = ScalarMachineState()
        state.sgprs[0] = 0x12345678
        state.write_vgpr(1, 0x9ABCDEF0)
        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(2),
            src0=Operand.sgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.imm(16),
        )
        res = ScalarInterpreter().step(instr, state)
        self.assertEqual(res, 0x56789ABC)
        self.assertEqual(state.vgprs[2], 0x56789ABC)


class SIMDMachineState:
    def __init__(self, num_lanes: int = 64, exec_mask: int = MASK64) -> None:
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask
        self.lanes_vgpr: list[list[int]] = [[0] * 256 for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * 106
        self.scc: int = 0

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def read_lane_operand(self, lane: int, op: Operand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.lanes_vgpr[lane][op.index] & MASK32
        if op.kind == OperandKind.SGPR:
            return self.sgprs[op.index] & MASK32
        return op.value & MASK32


class SIMDInterpreter:
    def step(self, instr: VAlignbitB32Instruction, state: SIMDMachineState) -> list[Optional[int]]:
        results: list[Optional[int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                s0 = state.read_lane_operand(lane, instr.src0)
                s1 = state.read_lane_operand(lane, instr.src1)
                s2 = state.read_lane_operand(lane, instr.src2)
                res = ref_alignbit_b32(s0, s1, s2)
                if instr.dst.kind == OperandKind.VGPR:
                    state.lanes_vgpr[lane][instr.dst.index] = res
                results.append(res)
            else:
                results.append(None)
        return results


class TestValuAlignbitB32SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_full_and_partial_divergence(self) -> None:
        state = SIMDMachineState(num_lanes=32, exec_mask=0x55555555)  # even lanes active
        for lane in range(32):
            state.lanes_vgpr[lane][1] = 0x12345678
            state.lanes_vgpr[lane][2] = 0x9ABCDEF0
            state.lanes_vgpr[lane][0] = 0xDEADBEEF  # canary in dst

        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.imm(8),
        )
        results = SIMDInterpreter().step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0x789ABCDE)
                self.assertEqual(state.lanes_vgpr[lane][0], 0x789ABCDE)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.lanes_vgpr[lane][0], 0xDEADBEEF)

    def test_wave64_zero_mask_preservation(self) -> None:
        state = SIMDMachineState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.lanes_vgpr[lane][0] = 0xCAFE0000 | lane

        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.imm(8),
        )
        results = SIMDInterpreter().step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(64):
            self.assertEqual(state.lanes_vgpr[lane][0], 0xCAFE0000 | lane)


if __name__ == "__main__":
    unittest.main()
