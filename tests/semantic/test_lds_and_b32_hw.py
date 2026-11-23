"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_and_b32 and ds_and_rtn_b32.

AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
  - LDS is an on-chip, software-managed, high-throughput scratchpad memory shared across
    all work-items (lanes/threads) within a workgroup.
  - Single 32-bit Atomic Bitwise AND:
    - ds_and_b32 vaddr, vdata [offset:N]:
      Non-returning form:
        eff_byte = (vaddr + offset) & 0xFFFFFFFF
        dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
        LDS[dword_idx] = LDS[dword_idx] & (vdata & 0xFFFFFFFF)
      The prior value at LDS[dword_idx] is discarded.
    - ds_and_rtn_b32 vdst, vaddr, vdata [offset:N]:
      Returning form:
        eff_byte = (vaddr + offset) & 0xFFFFFFFF
        dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[dword_idx]
        LDS[dword_idx] = old_val & (vdata & 0xFFFFFFFF)
        vdst = old_val
      Preserves the pre-atomic memory contents in destination VGPR vdst.
  - Active lane masking:
    - Vector operations governed by EXEC mask (32-bit for Wave32, 64-bit for Wave64).
    - Inactive lanes do not mutate LDS storage and leave destination VGPRs unmodified.
  - Hardware wrapping & alignment:
    - Byte addresses are shifted right by 2, dropping unaligned low bits [1:0].
    - Resulting dword index is masked with (lds_dwords - 1).
  - Bitwise AND properties:
    - Identity: x & 0xFFFFFFFF == x
    - Annihilator: x & 0 == 0
    - Idempotence: x & x == x
    - Commutativity: x & y == y & x
    - Associativity: (x & y) & z == x & (y & z)
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import unittest
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lds_run as L  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB default LDS allocation in dwords


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, num_dwords: int = DEFAULT_LDS_DWORDS, bounds_check: bool = False) -> None:
        if num_dwords & (num_dwords - 1) != 0:
            raise ValueError("num_dwords must be a power of two")
        self.num_dwords: int = num_dwords
        self.size_bytes: int = num_dwords * 4
        self.dwords: List[int] = [0] * num_dwords
        self.bounds_check: bool = bounds_check

    def read_dword(self, byte_addr: int) -> int:
        eff = byte_addr & MASK32
        idx = eff >> 2
        if self.bounds_check and idx >= self.num_dwords:
            return 0
        idx &= self.num_dwords - 1
        return self.dwords[idx]

    def write_dword(self, byte_addr: int, val: int) -> None:
        eff = byte_addr & MASK32
        idx = eff >> 2
        if self.bounds_check and idx >= self.num_dwords:
            return
        idx &= self.num_dwords - 1
        self.dwords[idx] = val & MASK32

    def atomic_and(self, byte_addr: int, data: int) -> int:
        """Atomic bitwise AND. Returns pre-atomic dword value."""
        eff = byte_addr & MASK32
        idx = eff >> 2
        if self.bounds_check and idx >= self.num_dwords:
            return 0
        idx &= self.num_dwords - 1
        old = self.dwords[idx]
        self.dwords[idx] = (old & (data & MASK32)) & MASK32
        return old


def ds_and_b32_oracle(old_val: int, data: int) -> int:
    """Pure mathematical reference oracle for 32-bit bitwise AND."""
    return (old_val & data) & MASK32


def ds_and_b32_execute_model(
    lds: LDSMemory, vaddr: int, vdata: int, offset: int = 0
) -> Tuple[Optional[int], int]:
    """Executes non-returning ds_and_b32. Returns (None, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.read_dword(eff_byte)
    new_val = ds_and_b32_oracle(old_val, vdata)
    lds.write_dword(eff_byte, new_val)
    return None, new_val


def ds_and_rtn_b32_execute_model(
    lds: LDSMemory, vaddr: int, vdata: int, offset: int = 0
) -> Tuple[int, int]:
    """Executes returning ds_and_rtn_b32. Returns (old_val, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.atomic_and(eff_byte, vdata)
    new_val = lds.read_dword(eff_byte)
    return old_val, new_val


class TestDSAndB32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_identity(self) -> None:
        for val in (0, 1, 0x12345678, 0x80000000, 0x7FFFFFFF, 0xAAAAAAAA, 0x55555555, 0xFFFFFFFF):
            with self.subTest(val=hex(val)):
                self.assertEqual(ds_and_b32_oracle(val, MASK32), val)

    def test_oracle_annihilator(self) -> None:
        for val in (0, 1, 0x12345678, 0x80000000, 0xAAAAAAAA, 0xFFFFFFFF):
            with self.subTest(val=hex(val)):
                self.assertEqual(ds_and_b32_oracle(val, 0), 0)

    def test_oracle_idempotence(self) -> None:
        for val in (0, 1, 0x12345678, 0xDEADBEEF, 0x80000000, 0xFFFFFFFF):
            with self.subTest(val=hex(val)):
                self.assertEqual(ds_and_b32_oracle(val, val), val)

    def test_oracle_commutativity(self) -> None:
        pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0xF0F0F0F0, 0x0F0F0F0F),
            (0xAAAAAAAA, 0x55555555),
            (0xFF00FF00, 0x00FF00FF),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ds_and_b32_oracle(a, b), ds_and_b32_oracle(b, a))

    def test_oracle_associativity(self) -> None:
        triplets = [
            (0x12345678, 0x9ABCDEF0, 0x55555555),
            (0xFFFFFFFF, 0x0F0F0F0F, 0x00FF00FF),
            (0xAAAAAAAA, 0xCCCCCCCC, 0xEFEFEFEF),
        ]
        for a, b, c in triplets:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                left = ds_and_b32_oracle(ds_and_b32_oracle(a, b), c)
                right = ds_and_b32_oracle(a, ds_and_b32_oracle(b, c))
                self.assertEqual(left, right)

    def test_lds_memory_address_wrapping_and_alignment(self) -> None:
        lds = LDSMemory(num_dwords=1024)  # 4 KiB
        lds.write_dword(0x40, 0xCAFEBABE)
        for off in (0, 1, 2, 3):
            self.assertEqual(lds.read_dword(0x40 + off), 0xCAFEBABE)
        self.assertEqual(lds.read_dword(0x40 + 4096), 0xCAFEBABE)

    def test_model_non_returning_and_returning(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x100, 0xFFFFFFFF)

        old_val, new_val = ds_and_rtn_b32_execute_model(lds, 0x100, 0x0F0F0F0F)
        self.assertEqual(old_val, 0xFFFFFFFF)
        self.assertEqual(new_val, 0x0F0F0F0F)
        self.assertEqual(lds.read_dword(0x100), 0x0F0F0F0F)

        rtn, new_val2 = ds_and_b32_execute_model(lds, 0x0E0, 0x00FF00FF, offset=0x020)
        self.assertIsNone(rtn)
        self.assertEqual(new_val2, 0x000F000F)
        self.assertEqual(lds.read_dword(0x100), 0x000F000F)


if __name__ == "__main__":
    unittest.main()
