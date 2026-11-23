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


if __name__ == "__main__":
    unittest.main()
