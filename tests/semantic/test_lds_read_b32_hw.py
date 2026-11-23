"""ds_read_b32: 32-bit LDS read into VGPR.

RDNA2 ISA Data Share Instructions:
  ds_read_b32 vdst, vaddr [offset:offset]
Effective byte address ADDR = (vaddr + offset) % LDS_SIZE.
Reads a 32-bit dword from LDS memory:
  D[31:0] = MEM[ADDR][31:0]
"""
from __future__ import annotations

import enum
import math
import os
import re
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L
import rdna2spirv
import support

LDS_DEFAULT_SIZE = 65536
M32 = 0xFFFFFFFF

def ds_read_b32_ref(lds_bytes: bytearray, vaddr: int, offset: int = 0, lds_size: int = LDS_DEFAULT_SIZE) -> int:
    eff_addr = (vaddr + offset) % lds_size
    b0 = lds_bytes[eff_addr]
    b1 = lds_bytes[(eff_addr + 1) % lds_size]
    b2 = lds_bytes[(eff_addr + 2) % lds_size]
    b3 = lds_bytes[(eff_addr + 3) % lds_size]
    return (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) & M32


class TestReferenceOracle(unittest.TestCase):
    def setUp(self) -> None:
        self.lds = bytearray(LDS_DEFAULT_SIZE)

    def test_basic_zero_offset(self) -> None:
        self.lds[0x100:0x104] = bytes([0x78, 0x56, 0x34, 0x12])
        got = ds_read_b32_ref(self.lds, 0x100)
        self.assertEqual(got, 0x12345678)

    def test_with_positive_offset(self) -> None:
        self.lds[0x210:0x214] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
        got = ds_read_b32_ref(self.lds, 0x200, offset=0x10)
        self.assertEqual(got, 0xDEADBEEF)

    def test_unaligned_byte_address(self) -> None:
        self.lds[0x301:0x305] = bytes([0x01, 0x02, 0x03, 0x04])
        got = ds_read_b32_ref(self.lds, 0x301)
        self.assertEqual(got, 0x04030201)

    def test_all_ones_pattern(self) -> None:
        self.lds[0x400:0x404] = bytes([0xFF, 0xFF, 0xFF, 0xFF])
        got = ds_read_b32_ref(self.lds, 0x400)
        self.assertEqual(got, 0xFFFFFFFF)

    def test_boundary_wrapping(self) -> None:
        # Wrap across boundary LDS_DEFAULT_SIZE - 2
        self.lds[LDS_DEFAULT_SIZE - 2] = 0x11
        self.lds[LDS_DEFAULT_SIZE - 1] = 0x22
        self.lds[0] = 0x33
        self.lds[1] = 0x44
        got = ds_read_b32_ref(self.lds, LDS_DEFAULT_SIZE - 2)
        self.assertEqual(got, 0x44332211)

    def test_large_offset_modulo(self) -> None:
        self.lds[0x20:0x24] = bytes([0xAA, 0xBB, 0xCC, 0xDD])
        got = ds_read_b32_ref(self.lds, 0x20, offset=LDS_DEFAULT_SIZE)
        self.assertEqual(got, 0xDDCCBBAA)

if __name__ == "__main__":
    unittest.main()
