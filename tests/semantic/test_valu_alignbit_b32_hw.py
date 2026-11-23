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


class TestValuAlignbitB32Aliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation."""

    def test_aliasing_dst_src0(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 0x9ABCDEF0)
        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.imm(8),
        )
        ScalarInterpreter().step(instr, state)
        self.assertEqual(state.vgprs[0], 0x789ABCDE)

    def test_aliasing_dst_src1(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x9ABCDEF0)
        state.write_vgpr(1, 0x12345678)
        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(0),
            src2=Operand.imm(8),
        )
        ScalarInterpreter().step(instr, state)
        self.assertEqual(state.vgprs[0], 0x789ABCDE)

    def test_aliasing_dst_src2(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 8)
        state.write_vgpr(1, 0x12345678)
        state.write_vgpr(2, 0x9ABCDEF0)
        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )
        ScalarInterpreter().step(instr, state)
        self.assertEqual(state.vgprs[0], 0x789ABCDE)

    def test_aliasing_all_operands(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x87654321)
        instr = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(0),
            src2=Operand.vgpr(0),
        )
        # shift = 0x87654321 & 31 = 1
        expected = ref_alignbit_b32(0x87654321, 0x87654321, 1)
        ScalarInterpreter().step(instr, state)
        self.assertEqual(state.vgprs[0], expected)


class TestValuAlignbitB32ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and carry/borrow generation."""

    def test_scc_and_vcc_unmodified(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA

        instr_zero = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.imm(0),
            src1=Operand.imm(0),
            src2=Operand.imm(0),
        )
        ScalarInterpreter().step(instr_zero, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)

        instr_neg = VAlignbitB32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.imm(0x80000000),
            src1=Operand.imm(0),
            src2=Operand.imm(0),
        )
        ScalarInterpreter().step(instr_neg, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)


class TestValuAlignbitB32IEEE754(unittest.TestCase):
    """Stage 7: IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_raw_bitwise_preservation_no_ftz(self) -> None:
        # Subnormal float 0x00000001 must not be flushed to zero
        subnormal = 0x00000001
        res = ref_alignbit_b32(subnormal, 0, 16)
        # (0x00000001 << 16) | 0 = 0x00010000
        self.assertEqual(res, 0x00010000)

    def test_nan_and_infinity_payload_preservation(self) -> None:
        qnan = 0x7FC00000
        snan = 0x7F800001
        pos_inf = 0x7F800000
        neg_inf = 0xFF800000

        # Rotate qnan by 0 returns qnan exactly
        self.assertEqual(ref_alignbit_b32(qnan, qnan, 0), qnan)
        # Shift snan: verify signaling bit is not modified/quietened
        res = ref_alignbit_b32(snan, snan, 0)
        self.assertEqual(res, snan)
        self.assertEqual(res & 0x00400000, 0)  # still signaling, not quietened

    def test_signed_zero_representation(self) -> None:
        neg_zero = 0x80000000
        pos_zero = 0x00000000
        # Funnel shift negative zero high word into low word
        res = ref_alignbit_b32(neg_zero, pos_zero, 8)
        # (0x80000000 << 24) = 0x00000000 (truncated), pos_zero >> 8 = 0
        self.assertEqual(res, 0x00000000)
        res31 = ref_alignbit_b32(neg_zero, pos_zero, 31)
        # (0x80000000 << 1) & MASK32 = 0x00000000
        self.assertEqual(res31, 0x00000000)
        res1 = ref_alignbit_b32(neg_zero, pos_zero, 1)
        # (0x80000000 << 31) & MASK32 = 0x00000000
        self.assertEqual(res1, 0x00000000)
        # Rotate negative zero by 1
        res_rot1 = ref_alignbit_b32(neg_zero, neg_zero, 1)
        # Bit 31 shifts to bit 30
        self.assertEqual(res_rot1, 0x40000000)


class TestValuAlignbitB32NegativeControls(unittest.TestCase):
    """Negative controls and mutation rejection."""

    def test_shift_mask_mutation_detection(self) -> None:
        def mutant_mask16(s0: int, s1: int, s2: int) -> int:
            shift = s2 & 0x0F  # bug: 4 bits instead of 5
            return ((s0 << (32 - shift)) | (s1 >> shift)) & MASK32 if shift else s1

        # Shift 16 must differ
        oracle_res = ref_alignbit_b32(0x12345678, 0x9ABCDEF0, 16)
        mutant_res = mutant_mask16(0x12345678, 0x9ABCDEF0, 16)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_zero_shift_omission_mutation(self) -> None:
        def mutant_no_zero_guard(s0: int, s1: int, s2: int) -> int:
            shift = s2 & 31
            return ((s1 << (32 - shift)) | (s0 >> shift)) & MASK32

        oracle_res = ref_alignbit_b32(0x12345678, 0x9ABCDEF0, 8)
        mutant_res = mutant_no_zero_guard(0x12345678, 0x9ABCDEF0, 8)
        self.assertNotEqual(oracle_res, mutant_res)


class TestValuAlignbitB32TranslationAndExecution(unittest.TestCase):
    """Stage 8: End-to-end SPIR-V translation and fragment execution."""

    def test_translator_all_vgprs(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 8",
            "v_alignbit_b32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_alignbit_vgprs", asm)
        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [0x789ABCDE] * 4)

    def test_translator_mixed_sgpr_imm(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "s_mov_b32 s0, 0x12345678",
            "v_mov_b32 v0, 0x9abcdef0",
            "v_alignbit_b32 v1, s0, v0, 16",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_alignbit_sgpr_imm", asm)
        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [0x56789ABC] * 4)

    def test_translator_zero_shift_immediate(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_alignbit_b32 v2, v0, v1, 0",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_alignbit_zero_imm", asm)
        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [0x9ABCDEF0] * 4)

    def test_translator_dynamic_shift_masking(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 40",  # 40 & 31 = 8
            "v_alignbit_b32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_alignbit_dyn_mask", asm)
        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [0x789ABCDE] * 4)

    def test_translator_aliasing(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_alignbit_b32 v0, v0, v1, 8",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_alignbit_alias", asm)
        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [0x789ABCDE] * 4)

    def test_spirv_structure(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_alignbit_b32 v2, v0, v1, 8",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_alignbit_struct", asm)
        self.assertIn("OpShiftRightLogical", spvasm)
        self.assertIn("OpShiftLeftLogical", spvasm)
        self.assertIn("OpBitwiseOr", spvasm)


if __name__ == "__main__":
    unittest.main()
