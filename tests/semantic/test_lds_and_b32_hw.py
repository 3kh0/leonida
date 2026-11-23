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


class DSOpcode(str, enum.Enum):
    DS_AND_B32 = "ds_and_b32"
    DS_AND_RTN_B32 = "ds_and_rtn_b32"


@dataclasses.dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclasses.dataclass(frozen=True)
class SGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    def __str__(self) -> str:
        return f"s{self.index}"


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    value: int

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 0xFFFF):
            raise ValueError(f"OffsetImmediate out of range [0, 0xFFFF]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}" if self.value != 0 else ""


@dataclasses.dataclass(frozen=True)
class DSAndInstruction:
    opcode: DSOpcode
    vaddr: VGPR
    vdata: VGPR
    vdst: Optional[VGPR] = None
    offset: OffsetImmediate = dataclasses.field(default_factory=lambda: OffsetImmediate(0))

    def __post_init__(self) -> None:
        if self.opcode == DSOpcode.DS_AND_B32 and self.vdst is not None:
            raise ValueError(f"{self.opcode} must not have a destination register")
        if self.opcode == DSOpcode.DS_AND_RTN_B32 and self.vdst is None:
            raise ValueError(f"{self.opcode} requires a destination register")

    def to_asm(self) -> str:
        off_str = f" {self.offset}" if self.offset.value != 0 else ""
        if self.opcode == DSOpcode.DS_AND_RTN_B32:
            return f"{self.opcode.value} {self.vdst}, {self.vaddr}, {self.vdata}{off_str}"
        return f"{self.opcode.value} {self.vaddr}, {self.vdata}{off_str}"


class TestDSAndB32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_non_returning(self) -> None:
        inst = DSAndInstruction(
            opcode=DSOpcode.DS_AND_B32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(16),
        )
        self.assertEqual(inst.opcode, DSOpcode.DS_AND_B32)
        self.assertEqual(inst.vaddr.index, 1)
        self.assertEqual(inst.vdata.index, 2)
        self.assertIsNone(inst.vdst)
        self.assertEqual(inst.offset.value, 16)
        self.assertEqual(inst.to_asm(), "ds_and_b32 v1, v2 offset:16")

    def test_ast_construction_returning(self) -> None:
        inst = DSAndInstruction(
            opcode=DSOpcode.DS_AND_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(0),
        )
        self.assertEqual(inst.opcode, DSOpcode.DS_AND_RTN_B32)
        self.assertEqual(inst.vdst.index, 5)
        self.assertEqual(inst.to_asm(), "ds_and_rtn_b32 v5, v1, v2")

    def test_ast_validation_rejects_dst_on_non_returning(self) -> None:
        with self.assertRaises(ValueError):
            DSAndInstruction(
                opcode=DSOpcode.DS_AND_B32,
                vdst=VGPR(5),
                vaddr=VGPR(1),
                vdata=VGPR(2),
            )

    def test_ast_validation_rejects_missing_dst_on_returning(self) -> None:
        with self.assertRaises(ValueError):
            DSAndInstruction(
                opcode=DSOpcode.DS_AND_RTN_B32,
                vdst=None,
                vaddr=VGPR(1),
                vdata=VGPR(2),
            )


class RegisterFile:
    """Architectural VGPR and SGPR register storage."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def get_vgpr(self, reg: VGPR) -> int:
        return self.vgprs[reg.index]

    def set_vgpr(self, reg: VGPR, val: int) -> None:
        self.vgprs[reg.index] = val & MASK32


class ScalarThreadState:
    """Thread state for scalar LDS execution."""

    def __init__(self, lds: LDSMemory) -> None:
        self.rf: RegisterFile = RegisterFile()
        self.lds: LDSMemory = lds


class ScalarDSInterpreter:
    """Deterministic scalar interpreter for DS AND instructions."""

    def execute(self, state: ScalarThreadState, inst: DSAndInstruction) -> Tuple[Optional[int], int]:
        addr = state.rf.get_vgpr(inst.vaddr)
        data = state.rf.get_vgpr(inst.vdata)
        eff_addr = (addr + inst.offset.value) & MASK32

        if inst.opcode == DSOpcode.DS_AND_RTN_B32:
            old_val = state.lds.atomic_and(eff_addr, data)
            new_val = state.lds.read_dword(eff_addr)
            assert inst.vdst is not None
            state.rf.set_vgpr(inst.vdst, old_val)
            return old_val, new_val
        else:
            old_val = state.lds.atomic_and(eff_addr, data)
            new_val = state.lds.read_dword(eff_addr)
            return None, new_val


class TestDSAndB32ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(num_dwords=1024)
        self.state = ScalarThreadState(self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_non_returning_execution(self) -> None:
        self.lds.write_dword(0x40, 0xFF00FF00)
        self.state.rf.set_vgpr(VGPR(1), 0x40)
        self.state.rf.set_vgpr(VGPR(2), 0x0F0F0F0F)

        inst = DSAndInstruction(opcode=DSOpcode.DS_AND_B32, vaddr=VGPR(1), vdata=VGPR(2))
        old, new = self.interp.execute(self.state, inst)
        self.assertIsNone(old)
        self.assertEqual(new, 0x0F000F00)
        self.assertEqual(self.lds.read_dword(0x40), 0x0F000F00)

    def test_scalar_returning_execution(self) -> None:
        self.lds.write_dword(0x80, 0x12345678)
        self.state.rf.set_vgpr(VGPR(1), 0x80)
        self.state.rf.set_vgpr(VGPR(2), 0x00FF00FF)
        self.state.rf.set_vgpr(VGPR(5), 0xDEADBEEF)

        inst = DSAndInstruction(
            opcode=DSOpcode.DS_AND_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0x00340078)
        self.assertEqual(self.state.rf.get_vgpr(VGPR(5)), 0x12345678)
        self.assertEqual(self.lds.read_dword(0x80), 0x00340078)

    def test_scalar_with_byte_offset(self) -> None:
        self.lds.write_dword(0x54, 0xFFFFFFFF)
        self.state.rf.set_vgpr(VGPR(1), 0x40)
        self.state.rf.set_vgpr(VGPR(2), 0xAAAAAAAA)

        inst = DSAndInstruction(
            opcode=DSOpcode.DS_AND_B32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(0x14),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(new, 0xAAAAAAAA)
        self.assertEqual(self.lds.read_dword(0x54), 0xAAAAAAAA)


class SIMDWaveState:
    """Wavefront state for SIMD execution (Wave32 or Wave64)."""

    def __init__(
        self,
        wave_size: int = 32,
        exec_mask: int = 0xFFFFFFFF,
        lds: Optional[LDSMemory] = None,
    ) -> None:
        if wave_size not in (32, 64):
            raise ValueError("wave_size must be 32 or 64")
        self.wave_size: int = wave_size
        self.exec_mask: int = exec_mask & ((1 << wave_size) - 1) if wave_size == 64 else exec_mask & MASK32
        self.lds: LDSMemory = lds or LDSMemory()
        self.lanes: List[RegisterFile] = [RegisterFile() for _ in range(wave_size)]

    def is_lane_active(self, lane_id: int) -> bool:
        return bool((self.exec_mask >> lane_id) & 1)


class SIMDDSInterpreter:
    """SIMD interpreter executing LDS operations with EXEC masking."""

    def execute(
        self, wave: SIMDWaveState, inst: DSAndInstruction
    ) -> List[Tuple[int, Optional[int], int]]:
        """Returns list of (lane_id, old_val, new_val) for executed active lanes."""
        results = []
        for lane_id in range(wave.wave_size):
            if not wave.is_lane_active(lane_id):
                continue
            rf = wave.lanes[lane_id]
            addr = rf.get_vgpr(inst.vaddr)
            data = rf.get_vgpr(inst.vdata)
            eff_addr = (addr + inst.offset.value) & MASK32

            if inst.opcode == DSOpcode.DS_AND_RTN_B32:
                old_val = wave.lds.atomic_and(eff_addr, data)
                new_val = wave.lds.read_dword(eff_addr)
                assert inst.vdst is not None
                rf.set_vgpr(inst.vdst, old_val)
                results.append((lane_id, old_val, new_val))
            else:
                old_val = wave.lds.atomic_and(eff_addr, data)
                new_val = wave.lds.read_dword(eff_addr)
                results.append((lane_id, None, new_val))
        return results


class TestDSAndB32SIMDDivergence(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_full_wave32(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        for l in range(32):
            lds.write_dword(l * 4, 0xFFFFFFFF)
            wave.lanes[l].set_vgpr(VGPR(1), l * 4)
            wave.lanes[l].set_vgpr(VGPR(2), 1 << l)

        inst = DSAndInstruction(
            opcode=DSOpcode.DS_AND_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(len(results), 32)
        for l in range(32):
            self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0xFFFFFFFF)
            self.assertEqual(lds.read_dword(l * 4), 1 << l)

    def test_simd_divergent_alternating_lanes(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x55555555, lds=lds)
        for l in range(32):
            lds.write_dword(l * 4, 0xFFFFFFFF)
            wave.lanes[l].set_vgpr(VGPR(1), l * 4)
            wave.lanes[l].set_vgpr(VGPR(2), 0x0F0F0F0F)
            wave.lanes[l].set_vgpr(VGPR(5), 0xCAFEBABE)

        inst = DSAndInstruction(
            opcode=DSOpcode.DS_AND_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(len(results), 16)
        for l in range(32):
            if l % 2 == 0:
                self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0xFFFFFFFF)
                self.assertEqual(lds.read_dword(l * 4), 0x0F0F0F0F)
            else:
                self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0xCAFEBABE)
                self.assertEqual(lds.read_dword(l * 4), 0xFFFFFFFF)

    def test_simd_exec_zero_neutrality(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x40, 0xFFFFFFFF)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x0, lds=lds)
        for l in range(32):
            wave.lanes[l].set_vgpr(VGPR(1), 0x40)
            wave.lanes[l].set_vgpr(VGPR(2), 0x0)
            wave.lanes[l].set_vgpr(VGPR(5), 0xDEADBEEF)

        inst = DSAndInstruction(
            opcode=DSOpcode.DS_AND_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(len(results), 0)
        self.assertEqual(lds.read_dword(0x40), 0xFFFFFFFF)
        for l in range(32):
            self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0xDEADBEEF)

    def test_simd_wave64_reduction_to_single_dword(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x20, 0xFFFFFFFF)
        wave = SIMDWaveState(wave_size=64, exec_mask=0xFFFFFFFF_FFFFFFFF, lds=lds)
        for l in range(64):
            wave.lanes[l].set_vgpr(VGPR(1), 0x20)
            wave.lanes[l].set_vgpr(VGPR(2), (~(1 << (l % 32))) & MASK32)

        inst = DSAndInstruction(opcode=DSOpcode.DS_AND_B32, vaddr=VGPR(1), vdata=VGPR(2))
        SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(lds.read_dword(0x20), 0)


if __name__ == "__main__":
    unittest.main()
