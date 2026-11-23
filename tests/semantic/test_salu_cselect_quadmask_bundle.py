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


if __name__ == "__main__":
    unittest.main()
