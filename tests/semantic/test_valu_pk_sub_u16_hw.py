from __future__ import annotations

"""v_pk_sub_u16 (RDNA2 VOP3P) hardware-referenced semantic regression tests.

Covers:
- Packed 16-bit unsigned integer subtraction across low and high halves:
    D.u16[0] = (S0.u16[op_sel[0]] - S1.u16[op_sel[1]]) & 0xFFFF
    D.u16[1] = (S0.u16[op_sel_hi[0]] - S1.u16[op_sel_hi[1]]) & 0xFFFF
- Modifiers: op_sel, op_sel_hi, and unsigned clamp (saturates underflow to 0)
- Arithmetic wrapping modulo 2^16 (65536) when clamp is omitted
- Register operand variations: VGPRs, SGPRs, and in-place register aliasing
- Full translation to SPIR-V via tools/rdna2spirv.py and execution via spirv_mini_interp

Standard library only: python -I, unittest, struct, math. Strictly zero game data.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402


def P(lo: int, hi: int) -> int:
    """Pack two 16-bit integers into a single 32-bit dword."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def run_pk_sub(instruction: str, seeds: dict[str, int], old: int = 0xDEADBEEF) -> int:
    """Translate and run a minimal pixel shader exercising `v_pk_sub_u16`, returning v0 from mrt0."""
    prog = [f"v_mov_b32 v0, {old:#x}"]
    for reg, val in sorted(seeds.items()):
        prefix = "s" if reg.startswith("s") else "v"
        prog.append(f"{prefix}_mov_b32 {reg}, {val:#x}")
    prog.append(instruction)
    prog.append("exp mrt0 v0, v0, v0, v0 done vm")
    prog.append("s_endpgm")
    spv = support.translate_lines("ps_pk_sub_u16", prog)
    return support.mrt0_words(spv)[0]


class ValuPkSubU16HardwareTests(unittest.TestCase):
    def test_basic_subtraction_both_lanes(self) -> None:
        """Test independent lane unsigned subtraction without underflow."""
        cases = [
            # ((lo0, hi0), (lo1, hi1), (exp_lo, exp_hi))
            ((10, 20), (3, 5), (7, 15)),
            ((100, 500), (50, 200), (50, 300)),
            ((0xFFFF, 0x1000), (1, 0x1000), (0xFFFE, 0x0000)),
            ((5000, 12000), (4999, 11999), (1, 1)),
            ((0x8000, 0x7FFF), (0x4000, 0x1FFF), (0x4000, 0x6000)),
        ]
        for (lo0, hi0), (lo1, hi1), (exp_lo, exp_hi) in cases:
            s0 = P(lo0, hi0)
            s1 = P(lo1, hi1)
            expected = P(exp_lo, exp_hi)
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                got = run_pk_sub("v_pk_sub_u16 v0, v1, v2", {"v1": s0, "v2": s1})
                self.assertEqual(got, expected)

    def test_unsigned_wrap_modulo_65536(self) -> None:
        """Unsigned 16-bit underflow wraps modulo 65536 (2^16) when unclamped."""
        # 0 - 1 = 0xFFFF (65535)
        s0 = P(0, 5)
        s1 = P(1, 10)
        got = run_pk_sub("v_pk_sub_u16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0xFFFF, 0xFFFB))

        # 0 - 0xFFFF = 1
        s0 = P(0, 10)
        s1 = P(0xFFFF, 20)
        got = run_pk_sub("v_pk_sub_u16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0x0001, 0xFFF6))

    def test_clamp_modifier_unsigned_saturation(self) -> None:
        """Modifier clamp saturates unsigned underflow to 0."""
        # Unclamped would be (0xFFFF, 0xFFFB), but clamp forces underflows (S0 < S1) to 0
        s0 = P(0, 5)
        s1 = P(1, 10)
        got = run_pk_sub("v_pk_sub_u16 v0, v1, v2 clamp", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0, 0))

        # Mixed: low lane underflows (clamped to 0), high lane valid (preserved)
        s0 = P(5, 50)
        s1 = P(10, 20)
        got = run_pk_sub("v_pk_sub_u16 v0, v1, v2 clamp", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0, 30))

        # Valid differences unaffected by clamp
        s0 = P(100, 200)
        s1 = P(30, 80)
        got = run_pk_sub("v_pk_sub_u16 v0, v1, v2 clamp", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(70, 120))

    def test_modifiers_op_sel_and_op_sel_hi(self) -> None:
        """Modifier op_sel and op_sel_hi routing half words into destinations."""
        # S0 = (lo=10, hi=50), S1 = (lo=3, hi=20)
        s0 = P(10, 50)
        s1 = P(3, 20)

        # Default: lo = S0.lo (10) - S1.lo (3) = 7
        #          hi = S0.hi (50) - S1.hi (20) = 30
        got = run_pk_sub("v_pk_sub_u16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(7, 30))

        # op_sel:[1, 0] -> lo = S0.hi (50) - S1.lo (3) = 47
        #                  hi = S0.hi (50) - S1.hi (20) = 30
        got = run_pk_sub("v_pk_sub_u16 v0, v1, v2 op_sel:[1,0]", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(47, 30))

        # op_sel:[0, 1] op_sel_hi:[0, 0] ->
        # lo = S0.lo (10) - S1.hi (20) = -10 wrap = 0xFFF6
        # hi = S0.lo (10) - S1.lo (3) = 7
        got = run_pk_sub(
            "v_pk_sub_u16 v0, v1, v2 op_sel:[0,1] op_sel_hi:[0,0]",
            {"v1": s0, "v2": s1},
        )
        self.assertEqual(got, P(0xFFF6, 7))

        # op_sel:[1, 1] op_sel_hi:[0, 0] ->
        # lo = S0.hi (50) - S1.hi (20) = 30
        # hi = S0.lo (10) - S1.lo (3) = 7
        got = run_pk_sub(
            "v_pk_sub_u16 v0, v1, v2 op_sel:[1,1] op_sel_hi:[0,0]",
            {"v1": s0, "v2": s1},
        )
        self.assertEqual(got, P(30, 7))

    def test_sgpr_and_aliasing(self) -> None:
        """Register variations: scalar SGPR sources and self-subtraction aliasing."""
        # Both SGPR sources: s1 - s2 -> v0
        s0 = P(100, 200)
        s1 = P(25, 50)
        got = run_pk_sub("v_pk_sub_u16 v0, s1, s2", {"s1": s0, "s2": s1})
        self.assertEqual(got, P(75, 150))

        # Mixed SGPR src0 and VGPR src1: s1 - v2 -> v0
        got = run_pk_sub("v_pk_sub_u16 v0, s1, v2", {"s1": s0, "v2": s1})
        self.assertEqual(got, P(75, 150))

        # Register aliasing (subtracting self in-place): v0 = v0 - v0
        s0 = P(0x1234, 0x5678)
        got = run_pk_sub("v_pk_sub_u16 v0, v0, v0", {"v0": s0})
        self.assertEqual(got, P(0, 0))

    def test_spirv_structure(self) -> None:
        """Verify that translation produces proper SPIR-V integer subtraction and extraction."""
        prog = [
            "v_mov_b32 v1, 0x00320014",  # hi: 50, lo: 20
            "v_mov_b32 v2, 0x000a0005",  # hi: 10, lo: 5
            "v_pk_sub_u16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_sub_u16_struct", prog)
        self.assertIn("OpISub %uint", spv)
        self.assertIn("OpBitFieldUExtract", spv)
        words = support.mrt0_words(spv)
        # (20 - 5 = 15 = 0x000F, 50 - 10 = 40 = 0x0028)
        self.assertEqual(words[0], P(15, 40))


if __name__ == "__main__":
    unittest.main()
