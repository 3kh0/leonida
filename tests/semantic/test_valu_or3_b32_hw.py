"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_or3_b32.

Models AMD RDNA2 / GFX10 Vector ALU (VALU) v_or3_b32 3-way bitwise OR semantics:
  - Computes the bitwise OR of three 32-bit unsigned integers: D = (S0 | S1 | S2) & 0xFFFFFFFF.
  - VOP3 / VOP3A encoding with 3 input operands (VGPR, SGPR, inline constants, or 32-bit literals).
  - Pure integer bitwise operation: no condition codes (SCC / VCC) are modified.
  - Mathematical properties:
      * Associativity: (A | B) | C == A | (B | C)
      * Commutativity: Permutations of (A, B, C) yield identical outputs
      * Identity element: A | 0 | 0 == A
      * Annihilator element: A | 0xFFFFFFFF | B == 0xFFFFFFFF
      * Idempotence: A | A | A == A

Synthesizes test listings, invokes tools/rdna2spirv.py via tests/semantic/support.py,
executes translated SPIR-V shaders in tests/semantic/spirv_mini_interp.py, and compares mrt0 outputs.

Standard library Python only (unittest, struct, math, dataclasses, enum), strictly zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Optional, Sequence, Tuple
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
import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def ref_or3_b32(s0: int, s1: int, s2: int) -> int:
    """Architectural reference oracle for RDNA2 v_or3_b32."""
    return ((s0 & MASK32) | (s1 & MASK32) | (s2 & MASK32)) & MASK32


def run_ps_or3(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_or3_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_bitwise_or(self):
        # 1 | 2 | 4 -> 7
        self.assertEqual(ref_or3_b32(1, 2, 4), 7)
        # 0x55555555 | 0xAAAAAAAA | 0 -> 0xFFFFFFFF
        self.assertEqual(ref_or3_b32(0x55555555, 0xAAAAAAAA, 0), MASK32)

    def test_permutations_and_commutativity(self):
        a, b, c = 0x12000000, 0x00340000, 0x00005678
        expected = 0x12345678
        self.assertEqual(ref_or3_b32(a, b, c), expected)
        self.assertEqual(ref_or3_b32(b, c, a), expected)
        self.assertEqual(ref_or3_b32(c, a, b), expected)

    def test_translator_basic_execution(self):
        words = run_ps_or3([
            "v_mov_b32 v0, 0x00000001",
            "v_mov_b32 v1, 0x00000002",
            "v_mov_b32 v2, 0x00000004",
            "v_or3_b32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 0x7)

    def test_translator_sparse_bitfields(self):
        words = run_ps_or3([
            "v_mov_b32 v0, 0x12000000",
            "v_mov_b32 v1, 0x00340000",
            "v_mov_b32 v2, 0x00005678",
            "v_or3_b32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 0x12345678)

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    literal_val: int = 0

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of bounds [0, 255]")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of bounds [0, 105]")

    @classmethod
    def vgpr(cls, idx: int) -> Operand:
        return cls(RegType.VGPR, index=idx)

    @classmethod
    def sgpr(cls, idx: int) -> Operand:
        return cls(RegType.SGPR, index=idx)

    @classmethod
    def literal(cls, val: int) -> Operand:
        return cls(RegType.LITERAL, literal_val=val & MASK32)

    def to_asm(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v{self.index}"
        elif self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        else:
            return f"{self.literal_val:#x}"


@dataclass(frozen=True)
class VOr3B32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination must be a VGPR for VALU instruction v_or3_b32")

    def to_asm(self) -> str:
        return f"v_or3_b32 {self.dst.to_asm()}, {self.src0.to_asm()}, {self.src1.to_asm()}, {self.src2.to_asm()}"


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn = VOr3B32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.sgpr(4),
            src2=Operand.literal(0x12345678),
        )
        self.assertEqual(insn.to_asm(), "v_or3_b32 v3, v0, s4, 0x12345678")

    def test_ast_immutability(self):
        insn = VOr3B32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        with self.assertRaises(Exception):
            insn.dst = Operand.vgpr(4)  # type: ignore

    def test_ast_validation_errors(self):
        with self.assertRaises(TypeError):
            VOr3B32Instruction(
                dst=Operand.sgpr(3),
                src0=Operand.vgpr(0),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(2),
            )
        with self.assertRaises(ValueError):
            Operand.vgpr(300)

class ScalarRegisterState:
    def __init__(self):
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.vcc = 0
        self.scc = 0

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index]
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index]
        else:
            return op.literal_val

    def write_vgpr(self, idx: int, val: int):
        self.vgprs[idx] = val & MASK32


class ScalarInterpreter:
    def __init__(self, state: Optional[ScalarRegisterState] = None):
        self.state = state if state is not None else ScalarRegisterState()

    def step(self, insn: VOr3B32Instruction) -> int:
        u0 = self.state.read_operand(insn.src0)
        u1 = self.state.read_operand(insn.src1)
        u2 = self.state.read_operand(insn.src2)
        res = ref_or3_b32(u0, u1, u2)
        self.state.write_vgpr(insn.dst.index, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step_and_chaining(self):
        interp = ScalarInterpreter()
        interp.state.vgprs[0] = 0x00000001
        interp.state.vgprs[1] = 0x00000002
        interp.state.vgprs[2] = 0x00000004
        interp.state.vgprs[4] = 0x00000008
        interp.state.vgprs[5] = 0x00000010

        # Insn 1: v3 = v0 | v1 | v2 -> 1 | 2 | 4 = 7
        insn1 = VOr3B32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        res1 = interp.step(insn1)
        self.assertEqual(res1, 0x7)
        self.assertEqual(interp.state.vgprs[3], 0x7)

        # Insn 2: v6 = v3 | v4 | v5 -> 7 | 8 | 16 = 31 (0x1F)
        insn2 = VOr3B32Instruction(
            dst=Operand.vgpr(6),
            src0=Operand.vgpr(3),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr(5),
        )
        res2 = interp.step(insn2)
        self.assertEqual(res2, 0x1F)
        self.assertEqual(interp.state.vgprs[6], 0x1F)

    def test_scalar_interpreter_matches_translator_chain(self):
        lines = [
            "v_mov_b32 v0, 0x00000001",
            "v_mov_b32 v1, 0x00000002",
            "v_mov_b32 v2, 0x00000004",
            "v_mov_b32 v4, 0x00000008",
            "v_mov_b32 v5, 0x00000010",
            "v_or3_b32 v3, v0, v1, v2",
            "v_or3_b32 v3, v3, v4, v5",
        ]
        words = run_ps_or3(lines)
        self.assertEqual(words[0], 0x1F)

class WaveSize(Enum):
    WAVE32 = 32
    WAVE64 = 64


class SIMDWaveState:
    def __init__(self, size: WaveSize = WaveSize.WAVE32):
        self.size = size
        self.num_lanes = size.value
        self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        self.sgprs = [0] * 106
        self.exec_mask = (1 << self.num_lanes) - 1

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def read_operand(self, lane: int, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index]
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index]
        else:
            return op.literal_val

    def write_vgpr(self, lane: int, idx: int, val: int):
        self.vgprs[lane][idx] = val & MASK32


class SIMDInterpreter:
    @staticmethod
    def step(insn: VOr3B32Instruction, wave: SIMDWaveState):
        for lane in range(wave.num_lanes):
            if not wave.is_lane_active(lane):
                continue
            u0 = wave.read_operand(lane, insn.src0)
            u1 = wave.read_operand(lane, insn.src1)
            u2 = wave.read_operand(lane, insn.src2)
            res = ref_or3_b32(u0, u1, u2)
            wave.write_vgpr(lane, insn.dst.index, res)


class TestSIMDDivergence(unittest.TestCase):
    def test_wave32_divergence_and_exec_masking(self):
        wave = SIMDWaveState(WaveSize.WAVE32)
        wave.exec_mask = 0x55555555

        sentinel = 0xDEADBEEF
        for lane in range(32):
            wave.vgprs[lane][3] = sentinel
            wave.vgprs[lane][0] = 1 << (lane % 8)
            wave.vgprs[lane][1] = 1 << ((lane + 1) % 8)
            wave.vgprs[lane][2] = 1 << ((lane + 2) % 8)

        insn = VOr3B32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        SIMDInterpreter.step(insn, wave)

        for lane in range(32):
            if lane % 2 == 0:
                expected = (1 << (lane % 8)) | (1 << ((lane + 1) % 8)) | (1 << ((lane + 2) % 8))
                self.assertEqual(wave.vgprs[lane][3], expected)
            else:
                self.assertEqual(wave.vgprs[lane][3], sentinel)

    def test_wave64_divergence(self):
        wave = SIMDWaveState(WaveSize.WAVE64)
        wave.exec_mask = 0x00000000FFFFFFFF

        sentinel = 0xCAFEBABE
        for lane in range(64):
            wave.vgprs[lane][3] = sentinel
            wave.vgprs[lane][0] = 0x01
            wave.vgprs[lane][1] = 0x02
            wave.vgprs[lane][2] = 0x04

        insn = VOr3B32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        SIMDInterpreter.step(insn, wave)

        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][3], 0x07)
        for lane in range(32, 64):
            self.assertEqual(wave.vgprs[lane][3], sentinel)

class TestRegisterAliasing(unittest.TestCase):
    def test_alias_dst_equals_src0(self):
        # v0 = v0 | v1 | v2
        words = run_ps_or3([
            "v_mov_b32 v0, 0x00000001",
            "v_mov_b32 v1, 0x00000002",
            "v_mov_b32 v2, 0x00000004",
            "v_or3_b32 v0, v0, v1, v2",
            "v_mov_b32 v3, v0",
        ])
        self.assertEqual(words[0], 0x7)

    def test_alias_dst_equals_src1(self):
        # v1 = v0 | v1 | v2
        words = run_ps_or3([
            "v_mov_b32 v0, 0x00000001",
            "v_mov_b32 v1, 0x00000002",
            "v_mov_b32 v2, 0x00000004",
            "v_or3_b32 v1, v0, v1, v2",
            "v_mov_b32 v3, v1",
        ])
        self.assertEqual(words[0], 0x7)

    def test_alias_dst_equals_src2(self):
        # v2 = v0 | v1 | v2
        words = run_ps_or3([
            "v_mov_b32 v0, 0x00000001",
            "v_mov_b32 v1, 0x00000002",
            "v_mov_b32 v2, 0x00000004",
            "v_or3_b32 v2, v0, v1, v2",
            "v_mov_b32 v3, v2",
        ])
        self.assertEqual(words[0], 0x7)

    def test_alias_quadruple_collision(self):
        # v0 = v0 | v0 | v0
        words = run_ps_or3([
            "v_mov_b32 v0, 0x12345678",
            "v_or3_b32 v0, v0, v0, v0",
            "v_mov_b32 v3, v0",
        ])
        self.assertEqual(words[0], 0x12345678)


if __name__ == "__main__":
    unittest.main()
