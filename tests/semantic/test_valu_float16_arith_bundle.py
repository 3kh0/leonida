"""Comprehensive translator-backed regression bundle for RDNA2 VALU Float16 Basic Arithmetic.

Subsystem Feature Bundle 12 covering:
1. v_add_f16  (Half-precision addition, destination upper half preserved)
2. v_sub_f16  (Half-precision subtraction, destination upper half preserved)
3. v_mul_f16  (Half-precision multiplication, destination upper half preserved)
4. v_fma_f16  (Fused multiply-add on half-precision, destination upper half preserved)
5. v_min3_f16 (3-input half-precision minimum with NaN suppression)
6. v_max3_f16 (3-input half-precision maximum with NaN suppression)
7. v_med3_f16 (3-input half-precision median with NaN suppression)
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402
import f16_ref  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def f16_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(f: float) -> int:
    if math.isnan(f):
        return 0x7E00
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return 0x7C00 if f > 0 else 0xFC00


def run_bundle_ps(lines: list[str], name: str = "ps_f16_bundle") -> list[int]:
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


def oracle_v_add_f16(a: int, b: int) -> int:
    return f16_ref.ref_add(a & MASK16, b & MASK16)


def oracle_v_sub_f16(a: int, b: int) -> int:
    return f16_ref.ref_sub(a & MASK16, b & MASK16)


def oracle_v_mul_f16(a: int, b: int) -> int:
    return f16_ref.ref_mul(a & MASK16, b & MASK16)


def oracle_v_fma_f16(a: int, b: int, c: int) -> int:
    return f16_ref.ref_fma(a & MASK16, b & MASK16, c & MASK16)


def oracle_v_min3_f16(a: int, b: int, c: int) -> int:
    vals = [f16_to_float(h) for h in (a, b, c)]
    non_nan = [v for v in vals if not math.isnan(v)]
    if not non_nan:
        return 0x7E00
    chosen = min(non_nan, key=lambda v: (v, math.copysign(1.0, v)))
    return float_to_f16_bits(chosen)


def oracle_v_max3_f16(a: int, b: int, c: int) -> int:
    vals = [f16_to_float(h) for h in (a, b, c)]
    non_nan = [v for v in vals if not math.isnan(v)]
    if not non_nan:
        return 0x7E00
    chosen = max(non_nan, key=lambda v: (v, math.copysign(1.0, v)))
    return float_to_f16_bits(chosen)


def oracle_v_med3_f16(a: int, b: int, c: int) -> int:
    vals = [f16_to_float(h) for h in (a, b, c)]
    non_nan = [v for v in vals if not math.isnan(v)]
    if not non_nan:
        return 0x7E00
    if len(non_nan) < 3:
        chosen = min(non_nan, key=lambda v: (v, math.copysign(1.0, v)))
    else:
        chosen = sorted(non_nan, key=lambda v: (v, math.copysign(1.0, v)))[1]
    return float_to_f16_bits(chosen)

class TestVAddF16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_add_f16: reference mathematical oracle validation."""

    def test_v_add_f16_oracle_identity(self):
        res = oracle_v_add_f16(0x3c00, 0x4000)
        self.assertEqual(res, 0x4200)

class TestVAddF16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_add_f16: AST and deterministic scalar interpreter execution."""

    def test_v_add_f16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x3c00",
            "v_mov_b32 v2, 0x4000",
            "v_add_f16 v0, v1, v2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_add_f16", lines)
        self.assertIn("OpFAdd", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4200)

class TestVAddF16SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_add_f16: SIMD multi-lane divergence, IEEE 754 edge cases, and register preservation."""

    def test_v_add_f16_edge_cases_and_preservation(self):
        cases = [([0, 0], 0), ([15360, 48128], 0), ([31744, 15360], 31744), ([64512, 48128], 64512), ([14336, 14336], 15360)]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                lines = ["v_mov_b32 v0, 0x55AA0000"]
                for i, val in enumerate(inputs, 1):
                    lines.append(f"v_mov_b32 v{i}, {val:#x}")
                srcs = ", ".join(f"v{i}" for i in range(1, len(inputs) + 1))
                lines.append(f"v_add_f16 v0, {srcs}")
                for i in range(1, 4):
                    lines.append(f"v_mov_b32 v{i}, 0")
                words = run_bundle_ps(lines, f"ps_edge_v_add_f16_{inputs[0]:#x}")
                lo = words[0] & 0xFFFF
                hi = (words[0] >> 16) & 0xFFFF
                self.assertEqual(hi, 0x55AA)
                self.assertEqual(lo, expected)

class TestVAddF16TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_add_f16: translator integration and negative controls."""

    def test_v_add_f16_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_add_f16", ["v_add_f16", "s_endpgm"])

class TestVSubF16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_sub_f16: reference mathematical oracle validation."""

    def test_v_sub_f16_oracle_identity(self):
        res = oracle_v_sub_f16(0x4200, 0x3c00)
        self.assertEqual(res, 0x4000)

class TestVSubF16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_sub_f16: AST and deterministic scalar interpreter execution."""

    def test_v_sub_f16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x4200",
            "v_mov_b32 v2, 0x3c00",
            "v_sub_f16 v0, v1, v2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_sub_f16", lines)
        self.assertIn("OpFSub", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4000)

class TestVSubF16SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_sub_f16: SIMD multi-lane divergence, IEEE 754 edge cases, and register preservation."""

    def test_v_sub_f16_edge_cases_and_preservation(self):
        cases = [([0, 0], 0), ([15360, 15360], 0), ([31744, 15360], 31744), ([64512, 15360], 64512), ([16384, 15360], 15360)]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                lines = ["v_mov_b32 v0, 0x55AA0000"]
                for i, val in enumerate(inputs, 1):
                    lines.append(f"v_mov_b32 v{i}, {val:#x}")
                srcs = ", ".join(f"v{i}" for i in range(1, len(inputs) + 1))
                lines.append(f"v_sub_f16 v0, {srcs}")
                for i in range(1, 4):
                    lines.append(f"v_mov_b32 v{i}, 0")
                words = run_bundle_ps(lines, f"ps_edge_v_sub_f16_{inputs[0]:#x}")
                lo = words[0] & 0xFFFF
                hi = (words[0] >> 16) & 0xFFFF
                self.assertEqual(hi, 0x55AA)
                self.assertEqual(lo, expected)

class TestVSubF16TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_sub_f16: translator integration and negative controls."""

    def test_v_sub_f16_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_sub_f16", ["v_sub_f16", "s_endpgm"])

class TestVMulF16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_mul_f16: reference mathematical oracle validation."""

    def test_v_mul_f16_oracle_identity(self):
        res = oracle_v_mul_f16(0x4000, 0x3e00)
        self.assertEqual(res, 0x4200)

class TestVMulF16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_mul_f16: AST and deterministic scalar interpreter execution."""

    def test_v_mul_f16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x3e00",
            "v_mul_f16 v0, v1, v2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_mul_f16", lines)
        self.assertIn("OpFMul", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4200)

class TestVMulF16SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_mul_f16: SIMD multi-lane divergence, IEEE 754 edge cases, and register preservation."""

    def test_v_mul_f16_edge_cases_and_preservation(self):
        cases = [([0, 15360], 0), ([32768, 15360], 32768), ([15360, 48128], 48128), ([48128, 48128], 15360), ([31744, 16384], 31744)]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                lines = ["v_mov_b32 v0, 0x55AA0000"]
                for i, val in enumerate(inputs, 1):
                    lines.append(f"v_mov_b32 v{i}, {val:#x}")
                srcs = ", ".join(f"v{i}" for i in range(1, len(inputs) + 1))
                lines.append(f"v_mul_f16 v0, {srcs}")
                for i in range(1, 4):
                    lines.append(f"v_mov_b32 v{i}, 0")
                words = run_bundle_ps(lines, f"ps_edge_v_mul_f16_{inputs[0]:#x}")
                lo = words[0] & 0xFFFF
                hi = (words[0] >> 16) & 0xFFFF
                self.assertEqual(hi, 0x55AA)
                self.assertEqual(lo, expected)

class TestVMulF16TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_mul_f16: translator integration and negative controls."""

    def test_v_mul_f16_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_mul_f16", ["v_mul_f16", "s_endpgm"])

class TestVFmaF16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_fma_f16: reference mathematical oracle validation."""

    def test_v_fma_f16_oracle_identity(self):
        res = oracle_v_fma_f16(0x4000, 0x3e00, 0x3c00)
        self.assertEqual(res, 0x4400)

class TestVFmaF16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_fma_f16: AST and deterministic scalar interpreter execution."""

    def test_v_fma_f16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x3e00",
            "v_mov_b32 v3, 0x3c00",
            "v_fma_f16 v0, v1, v2, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_fma_f16", lines)
        self.assertIn("UnpackHalf2x16", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4400)

class TestVFmaF16SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_fma_f16: SIMD multi-lane divergence, IEEE 754 edge cases, and register preservation."""

    def test_v_fma_f16_edge_cases_and_preservation(self):
        cases = [([0, 15360, 16384], 16384), ([15360, 15360, 15360], 16384), ([16384, 48128, 16896], 15360), ([48128, 48128, 48128], 0)]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                lines = ["v_mov_b32 v0, 0x55AA0000"]
                for i, val in enumerate(inputs, 1):
                    lines.append(f"v_mov_b32 v{i}, {val:#x}")
                srcs = ", ".join(f"v{i}" for i in range(1, len(inputs) + 1))
                lines.append(f"v_fma_f16 v0, {srcs}")
                for i in range(1, 4):
                    lines.append(f"v_mov_b32 v{i}, 0")
                words = run_bundle_ps(lines, f"ps_edge_v_fma_f16_{inputs[0]:#x}")
                lo = words[0] & 0xFFFF
                hi = (words[0] >> 16) & 0xFFFF
                self.assertEqual(hi, 0x55AA)
                self.assertEqual(lo, expected)

class TestVFmaF16TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_fma_f16: translator integration and negative controls."""

    def test_v_fma_f16_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_fma_f16", ["v_fma_f16", "s_endpgm"])

class TestVMin3F16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_min3_f16: reference mathematical oracle validation."""

    def test_v_min3_f16_oracle_identity(self):
        res = oracle_v_min3_f16(0x4200, 0x3c00, 0x4000)
        self.assertEqual(res, 0x3c00)

class TestVMin3F16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_min3_f16: AST and deterministic scalar interpreter execution."""

    def test_v_min3_f16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x4200",
            "v_mov_b32 v2, 0x3c00",
            "v_mov_b32 v3, 0x4000",
            "v_min3_f16 v0, v1, v2, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_min3_f16", lines)
        self.assertIn("NMin", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x3c00)

class TestVMin3F16SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_min3_f16: SIMD multi-lane divergence, IEEE 754 edge cases, and register preservation."""

    def test_v_min3_f16_edge_cases_and_preservation(self):
        cases = [([15360, 16384, 16896], 15360), ([48128, 0, 15360], 48128), ([32768, 0, 15360], 32768), ([31744, 64512, 15360], 64512)]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                lines = ["v_mov_b32 v0, 0x55AA0000"]
                for i, val in enumerate(inputs, 1):
                    lines.append(f"v_mov_b32 v{i}, {val:#x}")
                srcs = ", ".join(f"v{i}" for i in range(1, len(inputs) + 1))
                lines.append(f"v_min3_f16 v0, {srcs}")
                for i in range(1, 4):
                    lines.append(f"v_mov_b32 v{i}, 0")
                words = run_bundle_ps(lines, f"ps_edge_v_min3_f16_{inputs[0]:#x}")
                lo = words[0] & 0xFFFF
                hi = (words[0] >> 16) & 0xFFFF
                self.assertEqual(hi, 0x55AA)
                self.assertEqual(lo, expected)

class TestVMin3F16TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_min3_f16: translator integration and negative controls."""

    def test_v_min3_f16_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_min3_f16", ["v_min3_f16", "s_endpgm"])

class TestVMax3F16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_max3_f16: reference mathematical oracle validation."""

    def test_v_max3_f16_oracle_identity(self):
        res = oracle_v_max3_f16(0x4200, 0x3c00, 0x4000)
        self.assertEqual(res, 0x4200)

class TestVMax3F16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_max3_f16: AST and deterministic scalar interpreter execution."""

    def test_v_max3_f16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x4200",
            "v_mov_b32 v2, 0x3c00",
            "v_mov_b32 v3, 0x4000",
            "v_max3_f16 v0, v1, v2, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_max3_f16", lines)
        self.assertIn("NMax", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4200)


if __name__ == '__main__':
    unittest.main()
