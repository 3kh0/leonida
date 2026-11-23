"""Hardware-semantic translator-backed test suite for RDNA2 v_lshl_add_u32.

Models RDNA2 / GFX10 VALU 3-operand shift-left and add:
  v_lshl_add_u32 vd, src0, src1, src2

RDNA2 Architecture & Semantics:
  1. Computation:
     shift = S1.u32 & 0x1f
     D.u32 = ((S0.u32 << shift) + S2.u32) mod 2^32
  2. Shift Count Masking:
     Shift amount is strictly masked to 5 bits (0..31).
  3. Modulo wrapping:
     Both left shift and addition truncate to 32 bits (modulo 2^32).
  4. Operands:
     - vd: Destination VGPR
     - src0, src1, src2: VGPR, SGPR, or literal/inline constant (VOP3 rules apply).
  5. Condition Codes:
     - Neither SCC nor VCC is modified by v_lshl_add_u32.
  6. Translator Emission:
     - Lowers to OpBitwiseAnd (mask 31), OpShiftLeftLogical, and OpIAdd in SPIR-V.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv  # noqa: E402
    import support  # noqa: E402
except Exception:  # noqa: E722
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF


def ref_lshl_add_u32(src0: int, src1: int, src2: int) -> int:
    """Pure mathematical reference oracle for RDNA2 v_lshl_add_u32.

    Computes: ((src0 << (src1 & 31)) + src2) mod 2^32
    """
    s0 = src0 & MASK32
    shift = src1 & 0x1F
    s2 = src2 & MASK32
    return ((s0 << shift) + s2) & MASK32


class TestValuLshlAddU32Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and algebraic properties."""

    def test_basic_shift_and_add(self) -> None:
        self.assertEqual(ref_lshl_add_u32(10, 2, 5), 45)
        self.assertEqual(ref_lshl_add_u32(1, 4, 16), 32)
        self.assertEqual(ref_lshl_add_u32(0, 5, 100), 100)

    def test_shift_count_masking(self) -> None:
        # Shift count masked to 5 bits (0..31)
        self.assertEqual(ref_lshl_add_u32(1, 32, 0), ref_lshl_add_u32(1, 0, 0))
        self.assertEqual(ref_lshl_add_u32(1, 33, 0), ref_lshl_add_u32(1, 1, 0))
        self.assertEqual(ref_lshl_add_u32(1, 65, 0), ref_lshl_add_u32(1, 1, 0))
        self.assertEqual(ref_lshl_add_u32(1, 0xFFFFFFFF, 0), ref_lshl_add_u32(1, 31, 0))

    def test_identity_operands(self) -> None:
        val = 0x12345678
        # shift 0, add 0
        self.assertEqual(ref_lshl_add_u32(val, 0, 0), val)
        # shift 0 + addend
        self.assertEqual(ref_lshl_add_u32(val, 0, 10), (val + 10) & MASK32)
        # val 0 + addend
        self.assertEqual(ref_lshl_add_u32(0, 10, val), val)

    def test_overflow_and_wrapping(self) -> None:
        # Shift overflow
        self.assertEqual(ref_lshl_add_u32(0x80000000, 1, 0), 0)
        self.assertEqual(ref_lshl_add_u32(0x80000000, 1, 1), 1)

        # Addition overflow
        self.assertEqual(ref_lshl_add_u32(0x40000000, 1, 0x80000000), 0)
        self.assertEqual(ref_lshl_add_u32(0xFFFFFFFF, 0, 1), 0)

    def test_boundaries_and_extrema(self) -> None:
        for shift in range(32):
            self.assertEqual(ref_lshl_add_u32(1, shift, 0), 1 << shift)
            self.assertEqual(ref_lshl_add_u32(0xFFFFFFFF, shift, 0), (0xFFFFFFFF << shift) & MASK32)


class TestValuLshlAddU32TranslationAndExecution(unittest.TestCase):
    """End-to-end SPIR-V translation and fragment execution via support and spirv_mini_interp."""

    def test_translator_all_vgprs(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 3",
            "v_mov_b32 v2, 7",
            "v_lshl_add_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_lshl_add_vgprs", asm)
        self.assertIn("OpShiftLeftLogical", spvasm)
        self.assertIn("OpIAdd", spvasm)

        words = support.mrt0_words(spvasm)
        # (10 << 3) + 7 = 80 + 7 = 87
        self.assertEqual(words, [87, 87, 87, 87])

    def test_translator_mixed_sgpr_and_imm(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "s_mov_b32 s0, 5",
            "v_mov_b32 v0, 12",
            "v_lshl_add_u32 v1, s0, 4, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_lshl_add_sgpr_imm", asm)
        words = support.mrt0_words(spvasm)
        # (5 << 4) + 12 = 80 + 12 = 92
        self.assertEqual(words, [92, 92, 92, 92])

    def test_translator_aliased_destination(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 3",
            "v_lshl_add_u32 v0, v0, 2, 10",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_lshl_add_alias", asm)
        words = support.mrt0_words(spvasm)
        # (3 << 2) + 10 = 12 + 10 = 22
        self.assertEqual(words, [22, 22, 22, 22])

    def test_translator_shift_count_masking_32bit(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 65",  # 65 & 31 = 1
            "v_lshl_add_u32 v2, v0, v1, 0",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_lshl_add_masking", asm)
        words = support.mrt0_words(spvasm)
        # (1 << (65 & 31)) + 0 = 1 << 1 = 2
        self.assertEqual(words, [2, 2, 2, 2])

    def test_spirv_structure(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 3",
            "v_lshl_add_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_lshl_add_structure", asm)
        self.assertIn("OpBitwiseAnd", spvasm)
        self.assertIn("OpShiftLeftLogical", spvasm)
        self.assertIn("OpIAdd", spvasm)


if __name__ == "__main__":
    unittest.main()
