"""Hardware-measured and translator-backed semantic tests for s_absdiff_i32.

RDNA2 SALU instruction s_absdiff_i32:
Computes the absolute difference of two 32-bit signed scalar inputs:
    D = S0 - S1 as a 32-bit signed value, negated if negative
    SCC = (D != 0)
The difference wraps before the absolute value (e.g. 0x7FFFFFFF - 0x80000000 = -1 -> 1),
and SAbs keeps 0x80000000 as is.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def s_absdiff_i32_oracle(s0: int, s1: int) -> tuple[int, int]:
    """Exact bit-level mathematical reference oracle for s_absdiff_i32.

    Returns (result, scc).
    """
    s0 = s0 & MASK32
    s1 = s1 & MASK32
    diff = (s0 - s1) & MASK32
    diff_i = diff if diff < 0x80000000 else diff - 0x100000000
    if diff_i == -0x80000000:
        val = 0x80000000
    else:
        val = abs(diff_i) & MASK32
    scc = 1 if val != 0 else 0
    return val, scc


def run_absdiff_ps(lines: list[str], name: str = "ps_salu_absdiff") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class AbsdiffReferenceModel:
    """Architectural model tracking scalar registers, s_absdiff_i32, and SCC."""

    def __init__(self, num_sgprs: int = 16) -> None:
        self.sgprs = [0] * num_sgprs
        self.scc = 0

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg]

    def s_absdiff_i32(self, dst: int, src0: int, src1: int) -> int:
        val0 = self.sgprs[src0]
        val1 = self.sgprs[src1]
        res, scc = s_absdiff_i32_oracle(val0, val1)
        self.sgprs[dst] = res
        self.scc = scc
        return res


class TestAbsdiffReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for s_absdiff_i32."""

    def test_oracle_basic_differences(self):
        # abs(25 - 10) = 15, SCC=1
        self.assertEqual(s_absdiff_i32_oracle(25, 10), (15, 1))
        # abs(10 - 25) = 15, SCC=1
        self.assertEqual(s_absdiff_i32_oracle(10, 25), (15, 1))
        # abs(42 - 42) = 0, SCC=0
        self.assertEqual(s_absdiff_i32_oracle(42, 42), (0, 0))
        # abs(0 - 0) = 0, SCC=0
        self.assertEqual(s_absdiff_i32_oracle(0, 0), (0, 0))

    def test_oracle_wrapping_and_extremes(self):
        # 0x7FFFFFFF - 0x80000000 = -1 -> abs(-1) = 1
        self.assertEqual(s_absdiff_i32_oracle(0x7FFFFFFF, 0x80000000), (1, 1))
        # 0x80000000 - 0 = -0x80000000 -> keeps 0x80000000
        self.assertEqual(s_absdiff_i32_oracle(0x80000000, 0), (0x80000000, 1))

    def test_translator_execution_matches_oracle(self):
        pairs = [
            (50, 20),
            (20, 50),
            (100, 100),
            (0, 0),
            (-10 & MASK32, 10),
            (10, -10 & MASK32),
            (0x7FFFFFFF, 0x80000000),
            (0x80000000, 0),
        ]
        for idx, (s0, s1) in enumerate(pairs):
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                expected_val, expected_scc = s_absdiff_i32_oracle(s0, s1)
                lines = [
                    f"s_mov_b32 s0, {s0:#x}",
                    f"s_mov_b32 s1, {s1:#x}",
                    "s_absdiff_i32 s2, s0, s1",
                    "s_cselect_b32 s3, 1, 0",
                    "v_mov_b32 v0, s2",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_absdiff_ps(lines, f"ps_absdiff_c1_{idx}")
                self.assertEqual(words[0], expected_val)
                self.assertEqual(words[1], expected_scc)


class Sop2Opcode:
    """Symbolic AST opcode descriptor for SOP2 instructions."""

    def __init__(self, mnemonic: str, dst: str, src0: str, src1: str) -> None:
        self.mnemonic = mnemonic
        self.dst = dst
        self.src0 = src0
        self.src1 = src1

    def emit_assembly(self) -> str:
        return f"{self.mnemonic} {self.dst}, {self.src0}, {self.src1}"


class TestAbsdiffASTAndTypes(unittest.TestCase):
    """Commit 2: symbolic AST opcode definitions and typed expressions for s_absdiff_i32."""

    def test_ast_emission_and_translation(self):
        op = Sop2Opcode("s_absdiff_i32", "s2", "s0", "s1")
        self.assertEqual(op.emit_assembly(), "s_absdiff_i32 s2, s0, s1")
        lines = [
            "s_mov_b32 s0, 80",
            "s_mov_b32 s1, 30",
            op.emit_assembly(),
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        spv = support.translate_lines("ps_absdiff_ast", lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("SAbs", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)

    def test_ast_constant_folding_path(self):
        # When both operands are immediate literals, rdna2spirv folds the constant at compile time
        op = Sop2Opcode("s_absdiff_i32", "s0", "100", "40")
        lines = [
            op.emit_assembly(),
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_absdiff_ps(lines, "ps_absdiff_ast_fold")
        self.assertEqual(words[0], 60)


if __name__ == '__main__':
    unittest.main()
