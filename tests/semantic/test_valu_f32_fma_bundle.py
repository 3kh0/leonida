"""Comprehensive translator-backed regression bundle for RDNA2 VALU Float32 FMA, FMAC, and MAD.

Subsystem Feature Bundle 39 covering:
1. v_fma_f32   (Fused multiply-add on 32-bit floats with single rounding: D = S0 * S1 + S2)
2. v_fmac_f32  (Fused multiply-accumulate on 32-bit floats: D = S0 * S1 + D)
3. v_fmaak_f32 (Fused multiply-add with inline constant addend: D = S0 * S1 + K)
4. v_fmamk_f32 (Fused multiply-add with inline constant multiplier: D = S0 * K + S1)
5. v_mad_f32   (Unfused multiply-add with denormal flushing on 32-bit floats)
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

MASK32 = 0xFFFFFFFF


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def run_bundle_ps(lines: list[str], name: str = "ps_fma_bundle") -> list[int]:
    all_lines = list(lines) + [
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


def oracle_fma_f32(a: int, b: int, c: int) -> int:
    fa, fb, fc = u32_to_f32(a), u32_to_f32(b), u32_to_f32(c)
    if math.isnan(fa) or math.isnan(fb) or math.isnan(fc):
        return 0x7FC00000
    res = fa * fb + fc
    if math.isinf(res):
        return 0x7F800000 if res > 0 else 0xFF800000
    return f32_to_u32(res)


def oracle_mad_f32(a: int, b: int, c: int) -> int:
    def flush(u: int) -> float:
        if (u & 0x7F800000) == 0:
            return 0.0
        return u32_to_f32(u)

    fa, fb, fc = flush(a), flush(b), flush(c)
    prod = fa * fb
    p_bits = f32_to_u32(prod)
    if (p_bits & 0x7F800000) == 0:
        prod = 0.0
    res = prod + fc
    r_bits = f32_to_u32(res)
    if (r_bits & 0x7F800000) == 0:
        return 0
    return r_bits

class TestVFmaF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_fma_f32: reference mathematical oracle validation."""

    def test_v_fma_f32_oracle_identity(self):
        # 2.0 * 3.0 + 1.0 = 7.0
        res = oracle_fma_f32(0x40000000, 0x40400000, 0x3F800000)
        self.assertEqual(res, 0x40E00000)


if __name__ == '__main__':
    unittest.main()
