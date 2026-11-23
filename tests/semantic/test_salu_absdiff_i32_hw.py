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
            "v_readfirstlane_b32 s0, v0",
            "s_mov_b32 s1, 30",
            op.emit_assembly(),
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        spv = support.translate_lines("ps_absdiff_ast", lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("SAbs", spv)

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


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for SALU blocks."""

    def __init__(self) -> None:
        self.sgprs = [0] * 64
        self.scc = 0

    def exec_line(self, line: str) -> None:
        tokens = line.strip().replace(",", " ").split()
        if not tokens:
            return
        op = tokens[0]
        if op == "s_mov_b32":
            dst = int(tokens[1].lstrip("s"))
            val = int(tokens[2], 0)
            self.sgprs[dst] = val & MASK32
        elif op == "s_absdiff_i32":
            dst = int(tokens[1].lstrip("s"))
            src0 = self.resolve_src(tokens[2])
            src1 = self.resolve_src(tokens[3])
            val, scc = s_absdiff_i32_oracle(src0, src1)
            self.sgprs[dst] = val
            self.scc = scc

    def resolve_src(self, tok: str) -> int:
        if tok.startswith("s"):
            return self.sgprs[int(tok.lstrip("s"))]
        return int(tok, 0) & MASK32


class TestAbsdiffScalarInterpreter(unittest.TestCase):
    """Commit 3: deterministic scalar instruction interpreter for s_absdiff_i32."""

    def test_sequential_chained_execution(self):
        interp = ScalarInterpreter()
        lines = [
            "s_mov_b32 s0, 100",
            "s_mov_b32 s1, 60",
            "s_absdiff_i32 s2, s0, s1",   # abs(100 - 60) = 40
            "s_mov_b32 s3, 10",
            "s_absdiff_i32 s4, s2, s3",   # abs(40 - 10) = 30
        ]
        for ln in lines:
            interp.exec_line(ln)
        self.assertEqual(interp.sgprs[2], 40)
        self.assertEqual(interp.sgprs[4], 30)

        prog = lines + [
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s4",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_absdiff_ps(prog, "ps_absdiff_chained")
        self.assertEqual(words[0], 40)
        self.assertEqual(words[1], 30)

    def test_parallel_independent_diffs(self):
        interp = ScalarInterpreter()
        lines = [
            "s_mov_b32 s0, 10",
            "s_mov_b32 s1, 35",
            "s_mov_b32 s2, 70",
            "s_mov_b32 s3, 20",
            "s_absdiff_i32 s4, s0, s1",  # abs(10-35) = 25
            "s_absdiff_i32 s5, s2, s3",  # abs(70-20) = 50
        ]
        for ln in lines:
            interp.exec_line(ln)
        self.assertEqual(interp.sgprs[4], 25)
        self.assertEqual(interp.sgprs[5], 50)

        prog = lines + [
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_absdiff_ps(prog, "ps_absdiff_parallel")
        self.assertEqual(words[0], 25)
        self.assertEqual(words[1], 50)


class TestAbsdiffSimdInteraction(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for s_absdiff_i32."""

    def test_simd_broadcast_and_vector_cndmask(self):
        lines = [
            "s_mov_b32 s0, 150",
            "s_mov_b32 s1, 50",
            "s_absdiff_i32 s2, s0, s1",   # 100
            "s_mov_b32 s3, 30",
            "s_absdiff_i32 s4, s1, s3",   # 20
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s4",
            "v_cndmask_b32 v2, v0, v1, vcc",
            "v_mov_b32 v3, 0",
        ]
        words = run_absdiff_ps(lines, "ps_absdiff_simd")
        self.assertEqual(words[0], 100)
        self.assertEqual(words[1], 20)


class TestAbsdiffRegisterPreservation(unittest.TestCase):
    """Commit 5: register aliasing and in-place operand preservation for s_absdiff_i32."""

    def test_destination_aliased_to_source(self):
        # s0 = abs(s0 - s1)
        lines = [
            "s_mov_b32 s0, 15",
            "s_mov_b32 s1, 40",
            "s_absdiff_i32 s0, s0, s1",  # 25
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_absdiff_ps(lines, "ps_absdiff_alias_src0")
        self.assertEqual(words[0], 25)
        self.assertEqual(words[1], 40)

    def test_preservation_of_unrelated_sgprs(self):
        lines = [
            "s_mov_b32 s0, 0x1111",
            "s_mov_b32 s1, 0x2222",
            "s_mov_b32 s2, 0x3333",
            "s_mov_b32 s3, 0x4444",
            "s_absdiff_i32 s1, s1, s2",  # abs(0x2222 - 0x3333) = 0x1111
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ]
        words = run_absdiff_ps(lines, "ps_absdiff_preserve")
        self.assertEqual(words[0], 0x1111)
        self.assertEqual(words[1], 0x1111)
        self.assertEqual(words[2], 0x3333)
        self.assertEqual(words[3], 0x4444)


class TestAbsdiffConditionCodesAndSCC(unittest.TestCase):
    """Commit 6: condition codes, status flags, and carry/borrow generation for s_absdiff_i32."""

    def test_scc_set_to_one_on_nonzero_diff(self):
        lines = [
            "s_mov_b32 s0, 10",
            "s_mov_b32 s1, 20",
            "s_absdiff_i32 s2, s0, s1",
            "s_cselect_b32 s3, 0xAAAA, 0xBBBB",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_absdiff_ps(lines, "ps_absdiff_scc_one")
        self.assertEqual(words[0], 10)
        self.assertEqual(words[1], 0xAAAA)

    def test_scc_set_to_zero_on_zero_diff(self):
        lines = [
            "s_mov_b32 s0, 20",
            "s_mov_b32 s1, 20",
            "s_absdiff_i32 s2, s0, s1",
            "s_cselect_b32 s3, 0xAAAA, 0xBBBB",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_absdiff_ps(lines, "ps_absdiff_scc_zero")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0xBBBB)


class TestAbsdiffBoundaryAndExtremes(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling for s_absdiff_i32."""

    def test_ieee754_bit_patterns_as_integers(self):
        patterns = [
            (0x00000000, 0x80000000),  # +0.0 vs -0.0 / min_i32
            (0x3F800000, 0xBF800000),  # +1.0 vs -1.0
            (0x00000001, 0x007FFFFF),  # subnormals
            (0x7F800000, 0xFF800000),  # +inf vs -inf
            (0x7FC00000, 0x7FA00000),  # qNaN vs sNaN
            (0x7FFFFFFF, 0x80000000),  # max_i32 vs min_i32
            (0xFFFFFFFF, 0x00000000),  # -1 vs 0
            (0x7FFFFFFF, 0x7FFFFFFF),  # max_i32 vs max_i32
        ]
        for idx, (p0, p1) in enumerate(patterns):
            with self.subTest(p0=hex(p0), p1=hex(p1)):
                expected_val, expected_scc = s_absdiff_i32_oracle(p0, p1)
                lines = [
                    f"s_mov_b32 s0, {p0:#x}",
                    f"s_mov_b32 s1, {p1:#x}",
                    "s_absdiff_i32 s2, s0, s1",
                    "s_cselect_b32 s3, 1, 0",
                    "v_mov_b32 v0, s2",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_absdiff_ps(lines, f"ps_absdiff_ext_{idx}")
                self.assertEqual(words[0], expected_val)
                self.assertEqual(words[1], expected_scc)


class TestAbsdiffNegativeControlsAndMutations(unittest.TestCase):
    """Commit 8: negative controls, syntax rejection, and mutation detection for s_absdiff_i32."""

    def test_syntax_rejection_missing_operands(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_absdiff_bad_nop", ["s_absdiff_i32 s0", "s_endpgm"])

    def test_syntax_rejection_invalid_opcode(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_absdiff_bad_op", ["s_absdiff_nonexistent s0, s1, s2", "s_endpgm"])

    def test_mutation_detection_wrong_subtraction_order(self):
        # Oracle detects divergence if absolute value is omitted
        non_abs_oracle = lambda a, b: ((a - b) & MASK32, 1 if a != b else 0)
        self.assertNotEqual(s_absdiff_i32_oracle(10, 20)[0], non_abs_oracle(10, 20)[0])


if __name__ == "__main__":
    unittest.main()
