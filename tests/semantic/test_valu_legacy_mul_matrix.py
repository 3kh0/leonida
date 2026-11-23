"""v_mul_legacy_f32 and v_mad_legacy_f32 over a matrix of special values, against a reference model.

The translator tests the legacy zero rule on the unflushed operands (|x| < FLT_MIN, see Gen.legacy_zero) instead of
flushing both operands first. This checks the bits it produces against an independent Python model of the RDNA2
mode-0 behaviour that test_valu_mul_legacy_denorm.py describes:

- an f32 operand below FLT_MIN reads as a zero with its sign (flush-in);
- the legacy multiply gives +0 when either flushed operand is zero, whatever the other is (inf and NaN included);
- a subnormal product flushes to a zero with its sign (flush-out), before the output modifiers;
- omod then clamp; div:2, the one omod that can make a normal value subnormal, flushes again;
- with IEEE off, omod makes -0 +0.

Every pair of 14 values (±0, ±the smallest subnormal, ±the largest subnormal, ±FLT_MIN, 1.5, -2.5, 2^100, ±inf,
a quiet NaN) under each output-modifier set. A NaN result is compared as "is a NaN", not by its payload, and a NaN under
clamp is skipped: GLSL's FClamp leaves it to the driver. Synthetic
only, no game files.

    python -I tests/semantic/test_valu_legacy_mul_matrix.py -v
"""
import math
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import support  # noqa: E402

VALUES = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
          0x3FC00000, 0xC0200000, 0x71800000, 0x7F800000, 0xFF800000, 0x7FC00000]
MODS = ["", "clamp", "mul:2", "mul:4", "div:2", "div:2 clamp"]
FLT_MIN = 2.0 ** -126


def f(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def to_f32(x):
    """Round a Python float to binary32 (overflow to inf), as the hardware's f32 multiply rounds."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def flush(x):
    return math.copysign(0.0, x) if not math.isnan(x) and abs(x) < FLT_MIN else x


def legacy_product(x, y):
    x, y = flush(x), flush(y)
    if x == 0.0 or y == 0.0:
        return 0.0
    return flush(to_f32(x * y))


def apply_mods(r, mods):
    omod = [m for m in mods.split() if m in ("mul:2", "mul:4", "div:2")]
    for m in omod:
        r = to_f32(r * {"mul:2": 2.0, "mul:4": 4.0, "div:2": 0.5}[m])
    if "clamp" in mods.split() and not math.isnan(r):
        r = min(max(r, 0.0), 1.0)
    if "div:2" in omod:
        r = flush(r)
    if omod and r == 0.0:
        r = 0.0
    return r


def same(got_bits, want, mods=""):
    if math.isnan(want):
        return math.isnan(f(got_bits))
    if want == 0.0 and "clamp" in mods.split():
        return got_bits in (0, 0x80000000)    # FClamp leaves the sign of a zero to the driver, as for a NaN
    return got_bits == bits(want)


def run(op, operands, mods):
    lines = [f"v_mov_b32 v{i}, {v:#x}" for i, v in enumerate(operands)]
    regs = ", ".join(f"v{i}" for i in range(len(operands)))
    ins = op if not mods and op == "v_mul_legacy_f32" else op + ("_e64" if op == "v_mul_legacy_f32" else "")
    lines += [f"{ins} v9, {regs}{(' ' + mods) if mods else ''}", "exp mrt0 v9, v9, v9, v9 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_legacy_matrix", lines))[0]


class LegacyMultiplyMatrixTests(unittest.TestCase):
    def test_mul_legacy_matches_the_model(self):
        bad = []
        for mods in MODS:
            for a in VALUES:
                for b in VALUES:
                    product = legacy_product(f(a), f(b))
                    if math.isnan(product) and "clamp" in mods:
                        continue    # FClamp of a NaN is left to the driver (Gen.clamp_mods_f64's note), not modelled
                    want = apply_mods(product, mods)
                    got = run("v_mul_legacy_f32", (a, b), mods)
                    if not same(got, want, mods):
                        bad.append(f"{a:#010x} * {b:#010x} {mods or '-'}: got {got:#010x}, want {bits(want):#010x}")
        self.assertEqual(bad, [], "\n".join(bad[:20]))

    def test_mad_legacy_matches_the_model(self):
        # The addend is flushed too, then the sum is rounded to f32 and flushed (flush_f32_mods); no omod here,
        # that path is shared with v_mad_f32 and unchanged.
        bad = []
        for c in (0x00000000, 0x80000001, 0x3F800000, 0x7F800000):
            for a in VALUES:
                for b in VALUES:
                    p = legacy_product(f(a), f(b))
                    s = p + flush(f(c))
                    want = flush(to_f32(s)) if not math.isnan(s) else s
                    got = run("v_mad_legacy_f32", (a, b, c), "")
                    if not same(got, want):
                        bad.append(f"{a:#010x} * {b:#010x} + {c:#010x}: got {got:#010x}, want {bits(want):#010x}")
        self.assertEqual(bad, [], "\n".join(bad[:20]))


if __name__ == "__main__":
    unittest.main()
