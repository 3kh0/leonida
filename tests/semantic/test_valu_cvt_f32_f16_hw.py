from __future__ import annotations

"""Hardware-measured semantic tests for RDNA2 v_cvt_f32_f16.

Tests the vector 16-bit half-precision float to 32-bit float conversion:
    v_cvt_f32_f16 vdst, vsrc
Lowered semantics:
    Converts a binary16 half-precision float in src[15:0] to a binary32 float in dst[31:0].
    Fully defines the 32-bit destination VGPR (completely overwrites dst).
    Bits [31:16] of the source VGPR are ignored.

VOP3 modifiers:
    - Source modifiers: absolute value (|src|), negation (-src), negated abs (-|src|).
    - Output modifiers: clamp ([0.0, 1.0]).

Translated and verified through tools/rdna2spirv.py via tests/semantic/support.py.
Standard library Python only: unittest, struct, math. Zero game data.
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support


def run_cvt(
    src_h: int,
    old_dst: int = 0xDEADBEEF,
    opcode: str = "v_cvt_f32_f16",
    src_operand: str = "v1",
    mod_suffix: str = "",
) -> int:
    """Translate and run a fragment executing a single v_cvt_f32_f16 conversion."""
    suffix = f" {mod_suffix}" if mod_suffix else ""
    lines = [
        f"v_mov_b32 v0, {old_dst:#010x}",
        f"v_mov_b32 v1, {src_h:#010x}",
        f"{opcode} v0, {src_operand}{suffix}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_cvt_f32_f16", lines)
    return support.mrt0_words(spv)[0]


def run_cvt_quad(cases: list[tuple[int, int, str]]) -> list[int]:
    """Translate and run 4 simultaneous conversions exporting to mrt0 [v0, v1, v2, v3]."""
    assert len(cases) == 4
    lines = []
    for i, (src_h, old_dst, insn_tail) in enumerate(cases):
        lines.append(f"v_mov_b32 v{i}, {old_dst:#010x}")
        lines.append(f"v_mov_b32 v{10 + i}, {src_h:#010x}")
        lines.append(f"v_cvt_f32_f16_e64 v{i}, v{10 + i}{' ' + insn_tail if insn_tail else ''}")
    lines.append("exp mrt0 v0, v1, v2, v3 done vm")
    lines.append("s_endpgm")
    spv = support.translate_lines("ps_cvt_f32_f16_quad", lines)
    return support.mrt0_words(spv)


class TestValuCvtF32F16NormalConversions(unittest.TestCase):
    """Test normal IEEE-754 binary16 to binary32 conversions."""

    def test_normal_finite_values(self):
        cases = [
            (0x3C00, 0x3F800000),  # +1.0
            (0xBC00, 0xBF800000),  # -1.0
            (0x3800, 0x3F000000),  # +0.5
            (0xB800, 0xBF000000),  # -0.5
            (0x4000, 0x40000000),  # +2.0
            (0xC000, 0xC0000000),  # -2.0
            (0x3E00, 0x3FC00000),  # +1.5
            (0xBE00, 0xBFC00000),  # -1.5
            (0x4100, 0x40200000),  # +2.5
            (0x3000, 0x3E000000),  # +0.125
            (0x5140, 0x42280000),  # +42.0
            (0x7BFF, 0x477FE000),  # +65504.0 (max finite f16)
            (0xFBFF, 0xC77FE000),  # -65504.0
        ]
        for src_h, exp_f32 in cases:
            with self.subTest(src=hex(src_h), exp=hex(exp_f32)):
                got = run_cvt(src_h)
                self.assertEqual(got, exp_f32)

    def test_signed_zeros(self):
        # +0.0 (0x0000) -> +0.0 (0x00000000)
        got_pos = run_cvt(0x0000)
        self.assertEqual(got_pos, 0x00000000)

        # -0.0 (0x8000) -> -0.0 (0x80000000)
        got_neg = run_cvt(0x8000)
        self.assertEqual(got_neg, 0x80000000)

    def test_signed_infinities(self):
        # +inf (0x7C00) -> +inf (0x7F800000)
        got_pos = run_cvt(0x7C00)
        self.assertEqual(got_pos, 0x7F800000)

        # -inf (0xFC00) -> -inf (0xFF800000)
        got_neg = run_cvt(0xFC00)
        self.assertEqual(got_neg, 0xFF800000)

    def test_nans(self):
        # Quiet NaN (0x7E00) -> 0x7FC00000
        got_qnan = run_cvt(0x7E00)
        self.assertEqual(got_qnan, 0x7FC00000)

        # Negative quiet NaN (0xFE00) -> 0xFFC00000
        got_neg_qnan = run_cvt(0xFE00)
        self.assertEqual(got_neg_qnan, 0xFFC00000)

        # Signaling NaN with payload (0x7C01) -> 0x7FC00000
        got_snan = run_cvt(0x7C01)
        self.assertEqual(got_snan, 0x7FC00000)

    def test_subnormals(self):
        # Min normal f16: 2^-14 (0x0400) -> 0x38800000
        got_min_norm = run_cvt(0x0400)
        self.assertEqual(got_min_norm, 0x38800000)

        # Max subnormal f16: 0x03FF -> 0x387FC000
        got_max_sub = run_cvt(0x03FF)
        self.assertEqual(got_max_sub, 0x387FC000)

        # Smallest positive subnormal f16: 2^-24 (0x0001) -> 0x33800000
        got_min_sub = run_cvt(0x0001)
        self.assertEqual(got_min_sub, 0x33800000)

        # Smallest negative subnormal f16: -2^-24 (0x8001) -> 0xB3800000
        got_neg_min_sub = run_cvt(0x8001)
        self.assertEqual(got_neg_min_sub, 0xB3800000)

        # Largest negative subnormal f16: (0x83FF) -> 0xB87FC000
        got_neg_max_sub = run_cvt(0x83FF)
        self.assertEqual(got_neg_max_sub, 0xB87FC000)


class TestValuCvtF32F16Registers(unittest.TestCase):
    """Test register semantics: full destination overwrite and high-half masking."""

    def test_source_high_half_ignored(self):
        high_patterns = [0x00000000, 0x12340000, 0xABCD0000, 0xFFFF0000, 0xDEAD0000]
        for high in high_patterns:
            with self.subTest(high=hex(high)):
                got = run_cvt(high | 0x3C00)  # 1.0 in low half
                self.assertEqual(got, 0x3F800000)

    def test_destination_full_overwrite(self):
        dst_seeds = [0x00000000, 0xDEADBEEF, 0x12345678, 0xFFFFFFFF, 0x55AA55AA]
        for seed in dst_seeds:
            with self.subTest(seed=hex(seed)):
                got = run_cvt(0x3C00, old_dst=seed)  # 1.0 -> 0x3F800000
                self.assertEqual(got, 0x3F800000)


class TestValuCvtF32F16VOP3Modifiers(unittest.TestCase):
    """Test VOP3 source and output modifiers."""

    def test_source_abs(self):
        # |-1.0| -> +1.0 (0x3F800000)
        got = run_cvt(0xBC00, opcode="v_cvt_f32_f16_e64", src_operand="|v1|")
        self.assertEqual(got, 0x3F800000)

    def test_source_neg(self):
        # -(+1.0) -> -1.0 (0xBF800000)
        got = run_cvt(0x3C00, opcode="v_cvt_f32_f16_e64", src_operand="-v1")
        self.assertEqual(got, 0xBF800000)

        # -(-1.0) -> +1.0 (0x3F800000)
        got2 = run_cvt(0xBC00, opcode="v_cvt_f32_f16_e64", src_operand="-v1")
        self.assertEqual(got2, 0x3F800000)

    def test_source_neg_abs(self):
        # -|+1.0| -> -1.0 (0xBF800000)
        got = run_cvt(0x3C00, opcode="v_cvt_f32_f16_e64", src_operand="-|v1|")
        self.assertEqual(got, 0xBF800000)

    def test_output_clamp(self):
        # Value > 1.0 (2.0 = 0x4000) clamped to 1.0 (0x3F800000)
        got_pos = run_cvt(0x4000, opcode="v_cvt_f32_f16_e64", src_operand="v1", mod_suffix="clamp")
        self.assertEqual(got_pos, 0x3F800000)

        # Value < 0.0 (-1.0 = 0xBC00) clamped to 0.0 (0x00000000)
        got_neg = run_cvt(0xBC00, opcode="v_cvt_f32_f16_e64", src_operand="v1", mod_suffix="clamp")
        self.assertEqual(got_neg, 0x00000000)


class TestValuCvtF32F16InstructionForms(unittest.TestCase):
    """Test assembler forms, SGPR operands, inline constants, and lossless roundtrip."""

    def test_e32_vgpr_form(self):
        lines = [
            "v_mov_b32 v0, 0xDEADBEEF",
            "v_mov_b32 v1, 0x3C00",
            "v_cvt_f32_f16_e32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_e32", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x3F800000)

    def test_e32_sgpr_source(self):
        lines = [
            "v_mov_b32 v0, 0xDEADBEEF",
            "s_mov_b32 s4, 0x4000",  # 2.0
            "v_cvt_f32_f16_e32 v0, s4",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_sgpr", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x40000000)

    def test_inline_constants(self):
        cases = [
            ("1.0", 0x3F800000),
            ("0.5", 0x3F000000),
            ("-1.0", 0xBF800000),
            ("0", 0x00000000),
        ]
        for const_str, exp_f32 in cases:
            with self.subTest(const=const_str):
                lines = [
                    "v_mov_b32 v0, 0xDEADBEEF",
                    f"v_cvt_f32_f16 v0, {const_str}",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_cvt_const_{const_str.replace('.', '_')}", lines)
                self.assertEqual(support.mrt0_words(spv)[0], exp_f32)

    def test_lossless_roundtrip(self):
        # All finite f16 values survive roundtrip: v_cvt_f16_f32(v_cvt_f32_f16(h)) == h
        samples = [0x3C00, 0xBC00, 0x3800, 0xB800, 0x4000, 0xC000, 0x0001, 0x03FF, 0x7BFF, 0xFBFF]
        for h in samples:
            with self.subTest(h=hex(h)):
                lines = [
                    f"v_mov_b32 v1, {h:#010x}",
                    "v_mov_b32 v2, 0",
                    "v_cvt_f32_f16 v2, v1",
                    "v_mov_b32 v0, 0",
                    "v_cvt_f16_f32 v0, v2",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines("ps_roundtrip", lines)
                got_h = support.mrt0_words(spv)[0] & 0xFFFF
                self.assertEqual(got_h, h)

    def test_quad_pipeline(self):
        cases = [
            (0x3C00, 0xAAAA0000, ""),             # 1.0 -> 0x3F800000
            (0x4000, 0xBBBB0000, "clamp"),        # 2.0 clamped -> 0x3F800000
            (0x3800, 0xCCCC0000, ""),             # 0.5 -> 0x3F000000
            (0xBC00, 0xDDDD0000, ""),             # -1.0 -> 0xBF800000
        ]
        words = run_cvt_quad(cases)
        self.assertEqual(words, [0x3F800000, 0x3F800000, 0x3F000000, 0xBF800000])


if __name__ == "__main__":
    unittest.main()
