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


if __name__ == "__main__":
    unittest.main()
