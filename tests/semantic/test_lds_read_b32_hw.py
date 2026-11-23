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

class RegType(enum.Enum):
    VGPR = "v"
    SGPR = "s"


class RegSingle:
    def __init__(self, reg_type: RegType, index: int) -> None:
        self.reg_type = reg_type
        self.index = index

    def __repr__(self) -> str:
        return f"{self.reg_type.value}{self.index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegSingle):
            return False
        return self.reg_type == other.reg_type and self.index == other.index


class DSReadB32Inst:
    def __init__(self, vdst: RegSingle, vaddr: RegSingle, offset: int = 0) -> None:
        self.vdst = vdst
        self.vaddr = vaddr
        self.offset = offset

    def emit_asm(self) -> str:
        if self.offset != 0:
            return f"ds_read_b32 {self.vdst}, {self.vaddr} offset:{self.offset}"
        return f"ds_read_b32 {self.vdst}, {self.vaddr}"


def parse_ds_read_b32(line: str) -> DSReadB32Inst:
    line = line.strip()
    match = re.match(r"^ds_read_b32\s+(v\d+),\s*(v\d+)(?:\s+offset:(\d+|0x[0-9a-fA-F]+))?$", line)
    if not match:
        raise ValueError(f"Invalid ds_read_b32 syntax: {line}")
    vdst_str, vaddr_str, off_str = match.groups()
    vdst = RegSingle(RegType.VGPR, int(vdst_str[1:]))
    vaddr = RegSingle(RegType.VGPR, int(vaddr_str[1:]))
    offset = int(off_str, 0) if off_str is not None else 0
    if offset < 0 or offset > 0xFFFF:
        raise ValueError(f"Offset out of 16-bit range: {offset}")
    return DSReadB32Inst(vdst, vaddr, offset)


class TestASTDefinitions(unittest.TestCase):
    def test_ast_construction_and_emit(self) -> None:
        inst = DSReadB32Inst(RegSingle(RegType.VGPR, 8), RegSingle(RegType.VGPR, 1), 64)
        self.assertEqual(inst.emit_asm(), "ds_read_b32 v8, v1 offset:64")

    def test_ast_construction_zero_offset(self) -> None:
        inst = DSReadB32Inst(RegSingle(RegType.VGPR, 4), RegSingle(RegType.VGPR, 2), 0)
        self.assertEqual(inst.emit_asm(), "ds_read_b32 v4, v2")

    def test_parse_valid_assembler(self) -> None:
        inst = parse_ds_read_b32("ds_read_b32 v12, v3 offset:0x80")
        self.assertEqual(inst.vdst, RegSingle(RegType.VGPR, 12))
        self.assertEqual(inst.vaddr, RegSingle(RegType.VGPR, 3))
        self.assertEqual(inst.offset, 128)

    def test_parse_invalid_registers(self) -> None:
        with self.assertRaises(ValueError):
            parse_ds_read_b32("ds_read_b32 s0, v1")
        with self.assertRaises(ValueError):
            parse_ds_read_b32("ds_read_b32 v0, s1")

class ScalarState:
    def __init__(self, lds_size: int = LDS_DEFAULT_SIZE) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.lds = bytearray(lds_size)
        self.lds_size = lds_size
        self.exec_lo = 0xFFFFFFFF
        self.exec_hi = 0xFFFFFFFF
        self.scc = 0
        self.vcc_lo = 0
        self.vcc_hi = 0

    def write_lds_u32(self, addr: int, val: int) -> None:
        eff = addr % self.lds_size
        for k in range(4):
            self.lds[(eff + k) % self.lds_size] = (val >> (k * 8)) & 0xFF

    def step(self, inst: DSReadB32Inst) -> None:
        if (self.exec_lo & 1) == 0:
            return
        addr = self.vgprs[inst.vaddr.index]
        res = ds_read_b32_ref(self.lds, addr, inst.offset, self.lds_size)
        self.vgprs[inst.vdst.index] = res


class TestScalarInterpreter(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ScalarState()

    def test_scalar_read_execution(self) -> None:
        self.state.write_lds_u32(0x80, 0xCAFEBABE)
        self.state.vgprs[1] = 0x80
        inst = DSReadB32Inst(RegSingle(RegType.VGPR, 4), RegSingle(RegType.VGPR, 1))
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[4], 0xCAFEBABE)

    def test_scalar_read_with_offset(self) -> None:
        self.state.write_lds_u32(0x88, 0x12345678)
        self.state.vgprs[2] = 0x80
        inst = DSReadB32Inst(RegSingle(RegType.VGPR, 5), RegSingle(RegType.VGPR, 2), offset=8)
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[5], 0x12345678)

    def test_scalar_inactive_exec(self) -> None:
        self.state.exec_lo = 0
        self.state.vgprs[1] = 0x80
        self.state.vgprs[4] = 0x11111111
        self.state.write_lds_u32(0x80, 0x22222222)
        inst = DSReadB32Inst(RegSingle(RegType.VGPR, 4), RegSingle(RegType.VGPR, 1))
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[4], 0x11111111)

class SimdState:
    def __init__(self, num_lanes: int = 32, lds_size: int = LDS_DEFAULT_SIZE) -> None:
        self.num_lanes = num_lanes
        self.vgprs = [[0] * num_lanes for _ in range(256)]
        self.lds = bytearray(lds_size)
        self.lds_size = lds_size
        self.exec_mask = (1 << num_lanes) - 1

    def step(self, inst: DSReadB32Inst) -> None:
        for lane in range(self.num_lanes):
            if (self.exec_mask >> lane) & 1:
                vaddr = self.vgprs[inst.vaddr.index][lane]
                val = ds_read_b32_ref(self.lds, vaddr, inst.offset, self.lds_size)
                self.vgprs[inst.vdst.index][lane] = val


class TestSimdDivergence(unittest.TestCase):
    def test_simd_independent_lane_addresses(self) -> None:
        simd = SimdState(num_lanes=32)
        for lane in range(32):
            simd.vgprs[1][lane] = lane * 4
            val = 0xA0000000 | (lane << 8) | lane
            for k in range(4):
                simd.lds[lane * 4 + k] = (val >> (k * 8)) & 0xFF
        inst = DSReadB32Inst(RegSingle(RegType.VGPR, 4), RegSingle(RegType.VGPR, 1))
        simd.step(inst)
        for lane in range(32):
            expected = 0xA0000000 | (lane << 8) | lane
            self.assertEqual(simd.vgprs[4][lane], expected, f"lane {lane}")

    def test_wave32_exec_masking(self) -> None:
        simd = SimdState(num_lanes=32)
        simd.exec_mask = 0x0000FFFF
        for lane in range(32):
            simd.vgprs[1][lane] = 0x100
            simd.vgprs[2][lane] = 0xFEEDFACE
        simd.lds[0x100:0x104] = bytes([0x11, 0x22, 0x33, 0x44])
        inst = DSReadB32Inst(RegSingle(RegType.VGPR, 2), RegSingle(RegType.VGPR, 1))
        simd.step(inst)
        for lane in range(32):
            if lane < 16:
                self.assertEqual(simd.vgprs[2][lane], 0x44332211)
            else:
                self.assertEqual(simd.vgprs[2][lane], 0xFEEDFACE)

    def test_wave64_divergence(self) -> None:
        simd = SimdState(num_lanes=64)
        simd.exec_mask = 0xFFFFFFFFFFFFFFFF
        for lane in range(64):
            simd.vgprs[0][lane] = 0x200 + lane * 4
            val = 0x70000000 + lane
            for k in range(4):
                simd.lds[0x200 + lane * 4 + k] = (val >> (k * 8)) & 0xFF
        inst = DSReadB32Inst(RegSingle(RegType.VGPR, 6), RegSingle(RegType.VGPR, 0))
        simd.step(inst)
        for lane in range(64):
            self.assertEqual(simd.vgprs[6][lane], 0x70000000 + lane)

if __name__ == "__main__":
    unittest.main()
