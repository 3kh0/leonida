from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_lines

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

def ref_s_cselect_b32(scc: int, s0: int, s1: int) -> int:
    return (s0 if (scc & 1) != 0 else s1) & MASK32


class TestSCSelectB32Reference(unittest.TestCase):
    def test_ref_scc_set_selects_first(self):
        self.assertEqual(ref_s_cselect_b32(1, 0x12345678, 0x87654321), 0x12345678)
        self.assertEqual(ref_s_cselect_b32(1, 0xFFFFFFFF, 0x00000000), 0xFFFFFFFF)

    def test_ref_scc_cleared_selects_second(self):
        self.assertEqual(ref_s_cselect_b32(0, 0x12345678, 0x87654321), 0x87654321)
        self.assertEqual(ref_s_cselect_b32(0, 0xFFFFFFFF, 0x00000000), 0x00000000)

class SaluInstruction:
    def __init__(self, op: str, dst: int, s0: int, s1: int = 0):
        self.op = op
        self.dst = dst
        self.s0 = s0
        self.s1 = s1


class SaluInterpreterState:
    HANDLERS = {}

    def __init__(self):
        self.sgpr: dict[int, int] = {i: 0 for i in range(104)}
        self.scc: int = 0

    @classmethod
    def register(cls, op: str):
        def decorator(fn):
            cls.HANDLERS[op] = fn
            return fn
        return decorator

    def step(self, inst: SaluInstruction):
        if inst.op not in self.HANDLERS:
            raise NotImplementedError(f"Unsupported opcode: {inst.op}")
        return self.HANDLERS[inst.op](self, inst)


@SaluInterpreterState.register("s_cselect_b32")
def _step_cselect_b32(state: SaluInterpreterState, inst: SaluInstruction) -> int:
    s0_val = inst.s0 if not isinstance(inst.s0, str) else state.sgpr[int(inst.s0.lstrip("s"))]
    s1_val = inst.s1 if not isinstance(inst.s1, str) else state.sgpr[int(inst.s1.lstrip("s"))]
    res = ref_s_cselect_b32(state.scc, s0_val, s1_val)
    state.sgpr[inst.dst] = res
    return res


class TestSCSelectB32Interpreter(unittest.TestCase):
    def test_interpreter_scc_true(self):
        state = SaluInterpreterState()
        state.scc = 1
        inst = SaluInstruction("s_cselect_b32", dst=0, s0=0xCAFE, s1=0xBEEF)
        res = state.step(inst)
        self.assertEqual(res, 0xCAFE)
        self.assertEqual(state.sgpr[0], 0xCAFE)
        self.assertEqual(state.scc, 1)

    def test_interpreter_scc_false(self):
        state = SaluInterpreterState()
        state.scc = 0
        inst = SaluInstruction("s_cselect_b32", dst=1, s0=0xCAFE, s1=0xBEEF)
        res = state.step(inst)
        self.assertEqual(res, 0xBEEF)
        self.assertEqual(state.sgpr[1], 0xBEEF)
        self.assertEqual(state.scc, 0)

class TestSCSelectB32EdgeCases(unittest.TestCase):
    def test_extremum_constants(self):
        self.assertEqual(ref_s_cselect_b32(1, 0x80000000, 0x7FFFFFFF), 0x80000000)
        self.assertEqual(ref_s_cselect_b32(0, 0x80000000, 0x7FFFFFFF), 0x7FFFFFFF)

    def test_identical_operands(self):
        self.assertEqual(ref_s_cselect_b32(1, 0xAAAA5555, 0xAAAA5555), 0xAAAA5555)
        self.assertEqual(ref_s_cselect_b32(0, 0xAAAA5555, 0xAAAA5555), 0xAAAA5555)

    def test_simd_mask_branch_selection(self):
        for scc in (0, 1):
            selected = ref_s_cselect_b32(scc, 0xFFFFFFFF, 0x00000000)
            lanes = [1 if (selected & (1 << (i % 32))) != 0 else 0 for i in range(64)]
            expected_lane = 1 if scc == 1 else 0
            self.assertTrue(all(bit == expected_lane for bit in lanes))

class TestSCSelectB32Translator(unittest.TestCase):
    def test_translate_s_cselect_b32_scc1_and_scc0(self):
        lines = [
            "s_cmp_eq_u32 5, 5",
            "s_cselect_b32 s0, 0x12345678, 0x87654321",
            "v_mov_b32 v0, s0",
            "s_cmp_lg_u32 5, 5",
            "s_cselect_b32 s1, 0x12345678, 0x87654321",
            "v_mov_b32 v1, s1",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_csel32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = mrt0_words(spv)
        self.assertEqual(words, [0x12345678, 0x87654321, 0x12345678, 0x87654321])

def ref_s_cselect_b64(scc: int, s0_lo: int, s0_hi: int, s1_lo: int, s1_hi: int) -> tuple[int, int]:
    if (scc & 1) != 0:
        return s0_lo & MASK32, s0_hi & MASK32
    return s1_lo & MASK32, s1_hi & MASK32


class TestSCSelectB64Reference(unittest.TestCase):
    def test_ref_scc1_selects_first_pair(self):
        r_lo, r_hi = ref_s_cselect_b64(1, 0x11112222, 0x33334444, 0x55556666, 0x77778888)
        self.assertEqual((r_lo, r_hi), (0x11112222, 0x33334444))

    def test_ref_scc0_selects_second_pair(self):
        r_lo, r_hi = ref_s_cselect_b64(0, 0x11112222, 0x33334444, 0x55556666, 0x77778888)
        self.assertEqual((r_lo, r_hi), (0x55556666, 0x77778888))

@SaluInterpreterState.register("s_cselect_b64")
def _step_cselect_b64(state: SaluInterpreterState, inst: SaluInstruction) -> tuple[int, int]:
    s0_lo = state.sgpr[inst.s0]
    s0_hi = state.sgpr[inst.s0 + 1]
    s1_lo = state.sgpr[inst.s1]
    s1_hi = state.sgpr[inst.s1 + 1]
    r_lo, r_hi = ref_s_cselect_b64(state.scc, s0_lo, s0_hi, s1_lo, s1_hi)
    state.sgpr[inst.dst] = r_lo
    state.sgpr[inst.dst + 1] = r_hi
    return r_lo, r_hi


class TestSCSelectB64Interpreter(unittest.TestCase):
    def test_interpreter_step_scc1(self):
        state = SaluInterpreterState()
        state.scc = 1
        state.sgpr[2] = 0xAAAA0000
        state.sgpr[3] = 0x0000BBBB
        state.sgpr[4] = 0xCCCC0000
        state.sgpr[5] = 0x0000DDDD
        inst = SaluInstruction("s_cselect_b64", dst=0, s0=2, s1=4)
        r_lo, r_hi = state.step(inst)
        self.assertEqual((r_lo, r_hi), (0xAAAA0000, 0x0000BBBB))
        self.assertEqual((state.sgpr[0], state.sgpr[1]), (0xAAAA0000, 0x0000BBBB))
        self.assertEqual(state.scc, 1)

    def test_interpreter_step_scc0(self):
        state = SaluInterpreterState()
        state.scc = 0
        state.sgpr[2] = 0xAAAA0000
        state.sgpr[3] = 0x0000BBBB
        state.sgpr[4] = 0xCCCC0000
        state.sgpr[5] = 0x0000DDDD
        inst = SaluInstruction("s_cselect_b64", dst=0, s0=2, s1=4)
        r_lo, r_hi = state.step(inst)
        self.assertEqual((r_lo, r_hi), (0xCCCC0000, 0x0000DDDD))
        self.assertEqual((state.sgpr[0], state.sgpr[1]), (0xCCCC0000, 0x0000DDDD))
        self.assertEqual(state.scc, 0)

class TestSCSelectB64EdgeCases(unittest.TestCase):
    def test_64bit_extremes(self):
        r_lo, r_hi = ref_s_cselect_b64(1, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0)
        self.assertEqual((r_lo, r_hi), (0xFFFFFFFF, 0xFFFFFFFF))
        r_lo, r_hi = ref_s_cselect_b64(0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0)
        self.assertEqual((r_lo, r_hi), (0, 0))

    def test_scc_preserved(self):
        state = SaluInterpreterState()
        for initial_scc in (0, 1):
            state.scc = initial_scc
            state.step(SaluInstruction("s_cselect_b64", dst=0, s0=2, s1=4))
            self.assertEqual(state.scc, initial_scc)

class TestSCSelectB64Translator(unittest.TestCase):
    def test_translate_s_cselect_b64(self):
        lines = [
            "s_mov_b32 s2, 0x11112222",
            "s_mov_b32 s3, 0x33334444",
            "s_mov_b32 s4, 0x55556666",
            "s_mov_b32 s5, 0x77778888",
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b64 s[0:1], s[2:3], s[4:5]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "s_cmp_lg_u32 1, 1",
            "s_cselect_b64 s[0:1], s[2:3], s[4:5]",
            "v_mov_b32 v2, s0",
            "v_mov_b32 v3, s1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_csel64", lines)
        self.assertIn("OpEntryPoint", spv)
        words = mrt0_words(spv)
        self.assertEqual(words, [0x11112222, 0x33334444, 0x55556666, 0x77778888])

def ref_s_quadmask_b32(s0: int) -> tuple[int, int]:
    val = s0 & MASK32
    res = 0
    for i in range(8):
        nibble = (val >> (4 * i)) & 0xF
        if nibble != 0:
            res |= (1 << i)
    scc = 1 if res != 0 else 0
    return res & MASK32, scc


class TestSQuadmaskB32Reference(unittest.TestCase):
    def test_ref_alternating_nibbles(self):
        res, scc = ref_s_quadmask_b32(0x01020408)
        self.assertEqual(res, 0x55)
        self.assertEqual(scc, 1)

    def test_ref_zero_input(self):
        res, scc = ref_s_quadmask_b32(0)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_ref_all_ones(self):
        res, scc = ref_s_quadmask_b32(0xFFFFFFFF)
        self.assertEqual(res, 0xFF)
        self.assertEqual(scc, 1)

@SaluInterpreterState.register("s_quadmask_b32")
def _step_quadmask_b32(state: SaluInterpreterState, inst: SaluInstruction) -> tuple[int, int]:
    s0_val = inst.s0 if not isinstance(inst.s0, str) else state.sgpr[int(inst.s0.lstrip("s"))]
    res, scc = ref_s_quadmask_b32(s0_val)
    state.sgpr[inst.dst] = res
    state.scc = scc
    return res, scc


class TestSQuadmaskB32Interpreter(unittest.TestCase):
    def test_interpreter_step_nonzero(self):
        state = SaluInterpreterState()
        inst = SaluInstruction("s_quadmask_b32", dst=0, s0=0x01020408)
        res, scc = state.step(inst)
        self.assertEqual(res, 0x55)
        self.assertEqual(scc, 1)
        self.assertEqual(state.sgpr[0], 0x55)
        self.assertEqual(state.scc, 1)

    def test_interpreter_step_zero(self):
        state = SaluInterpreterState()
        state.scc = 1
        inst = SaluInstruction("s_quadmask_b32", dst=0, s0=0)
        res, scc = state.step(inst)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(state.sgpr[0], 0)
        self.assertEqual(state.scc, 0)

class TestSQuadmaskB32EdgeCases(unittest.TestCase):
    def test_each_bit_in_nibble(self):
        for bit_offset in (0, 1, 2, 3):
            val = (1 << bit_offset)
            res, scc = ref_s_quadmask_b32(val)
            self.assertEqual(res, 1)
            self.assertEqual(scc, 1)

    def test_upper_bits_strictly_zero(self):
        res, _ = ref_s_quadmask_b32(0xFFFFFFFF)
        self.assertEqual(res & ~0xFF, 0)

class TestSQuadmaskB32Translator(unittest.TestCase):
    def test_translate_s_quadmask_b32(self):
        lines = [
            "s_mov_b32 s2, 0x01020408",
            "s_quadmask_b32 s0, s2",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_qm32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = mrt0_words(spv)
        self.assertEqual(words, [0x55, 0x1, 0x55, 0x1])

def ref_s_quadmask_b64(s0_lo: int, s0_hi: int) -> tuple[int, int, int]:
    val = ((s0_hi & MASK32) << 32) | (s0_lo & MASK32)
    res = 0
    for i in range(16):
        nibble = (val >> (4 * i)) & 0xF
        if nibble != 0:
            res |= (1 << i)
    scc = 1 if res != 0 else 0
    return res & 0xFFFF, 0, scc


class TestSQuadmaskB64Reference(unittest.TestCase):
    def test_ref_quadmask_b64_nonzero(self):
        r_lo, r_hi, scc = ref_s_quadmask_b64(0x01020408, 0x01020408)
        self.assertEqual(r_lo, 0x5555)
        self.assertEqual(r_hi, 0)
        self.assertEqual(scc, 1)

    def test_ref_quadmask_b64_zero(self):
        r_lo, r_hi, scc = ref_s_quadmask_b64(0, 0)
        self.assertEqual((r_lo, r_hi, scc), (0, 0, 0))

@SaluInterpreterState.register("s_quadmask_b64")
def _step_quadmask_b64(state: SaluInterpreterState, inst: SaluInstruction) -> tuple[int, int, int]:
    s0_lo = state.sgpr[inst.s0]
    s0_hi = state.sgpr[inst.s0 + 1]
    r_lo, r_hi, scc = ref_s_quadmask_b64(s0_lo, s0_hi)
    state.sgpr[inst.dst] = r_lo
    state.sgpr[inst.dst + 1] = r_hi
    state.scc = scc
    return r_lo, r_hi, scc


class TestSQuadmaskB64Interpreter(unittest.TestCase):
    def test_interpreter_step_quadmask_b64(self):
        state = SaluInterpreterState()
        state.sgpr[2] = 0x01020408
        state.sgpr[3] = 0x01020408
        inst = SaluInstruction("s_quadmask_b64", dst=0, s0=2)
        r_lo, r_hi, scc = state.step(inst)
        self.assertEqual((r_lo, r_hi, scc), (0x5555, 0, 1))
        self.assertEqual((state.sgpr[0], state.sgpr[1]), (0x5555, 0))
        self.assertEqual(state.scc, 1)

class TestSQuadmaskB64EdgeCases(unittest.TestCase):
    def test_quadmask_high_word_only(self):
        r_lo, r_hi, scc = ref_s_quadmask_b64(0, 0xF0000000)
        self.assertEqual(r_lo, 0x8000)
        self.assertEqual(r_hi, 0)
        self.assertEqual(scc, 1)

    def test_quadmask_low_word_only(self):
        r_lo, r_hi, scc = ref_s_quadmask_b64(0x0000000F, 0)
        self.assertEqual(r_lo, 0x0001)
        self.assertEqual(r_hi, 0)
        self.assertEqual(scc, 1)

class TestSQuadmaskB64Translator(unittest.TestCase):
    def test_translate_s_quadmask_b64(self):
        lines = [
            "s_mov_b32 s2, 0x01020408",
            "s_mov_b32 s3, 0x01020408",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "s_cselect_b32 s2, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "exp mrt0 v0, v1, v2, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_qm64", lines)
        self.assertIn("OpEntryPoint", spv)
        words = mrt0_words(spv)
        self.assertEqual(words, [0x5555, 0, 1, 0x5555])

def ref_s_bitset0_b32(dst: int, s0: int) -> int:
    bit = s0 & 31
    return (dst & ~(1 << bit)) & MASK32


class TestSBitset0B32Reference(unittest.TestCase):
    def test_ref_clear_bit(self):
        self.assertEqual(ref_s_bitset0_b32(0xFFFFFFFF, 4), 0xFFFFFFEF)
        self.assertEqual(ref_s_bitset0_b32(0xFFFFFFFF, 0), 0xFFFFFFFE)
        self.assertEqual(ref_s_bitset0_b32(0xFFFFFFFF, 31), 0x7FFFFFFF)

    def test_ref_clear_modulo(self):
        self.assertEqual(ref_s_bitset0_b32(0xFFFFFFFF, 36), 0xFFFFFFEF)


if __name__ == "__main__":
    unittest.main()
