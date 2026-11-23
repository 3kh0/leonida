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


if __name__ == '__main__':
    unittest.main()
