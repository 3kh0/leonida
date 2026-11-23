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


if __name__ == "__main__":
    unittest.main()
