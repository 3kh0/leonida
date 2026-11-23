"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_cmpst_b32.

Models AMD RDNA2 / GFX10 Local Data Share (LDS) ds_cmpst_b32 / ds_cmpst_rtn_b32 atomic compare-and-swap semantics:
  - Atomically computes:
      addr = (vaddr + offset) & 0xFFFFFFFF
      idx = (addr >> 2) & (lds_dwords - 1)
      old_val = LDS[idx]
      if old_val == vcmp:
          LDS[idx] = vsrc
  - In returning form (ds_cmpst_rtn_b32): writes old_val to vdst VGPR.
  - In non-returning form (ds_cmpst_b32): prior value is discarded.
  - 32-bit word equality comparison:
      * Matches when exact 32-bit pattern matches.
      * Unchanged when old_val != vcmp.
  - Pure memory atomic: condition codes (SCC / VCC) are not modified.

Synthesizes test listings, invokes tools/rdna2spirv.py via tests/semantic/support.py,
executes translated SPIR-V shaders in tests/semantic/spirv_mini_interp.py, and compares output buffers.

Standard library Python only (unittest, struct, math, dataclasses, enum), strictly zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Optional, Sequence, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF

# Compute FSSM header: RSRC2 0x8088 (4 user SGPRs, 64x1x1 threads)
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 64, 1, 1)
CS_FSSM_BYTES = bytes(CS_FSSM)

OUT_VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28


def ref_cmpst_b32(current_val: int, cmp_val: int, swap_val: int) -> Tuple[int, int]:
    """Architectural reference oracle for RDNA2 ds_cmpst_b32.
    Returns: (new_val, old_val)
    """
    curr = current_val & MASK32
    cmp_ = cmp_val & MASK32
    swap_ = swap_val & MASK32
    if curr == cmp_:
        return (swap_, curr)
    return (curr, curr)


def run_cs_lds(instructions: Sequence[str]) -> dict[int, int]:
    """Translate and execute a synthetic compute shader, returning output memory."""
    lines = list(instructions) + ["s_endpgm"]
    spvasm = support.translate_lines("cs_lds_cmpst_b32_test", lines, fssm=CS_FSSM_BYTES)
    memory: dict[int, int] = {}
    spirv_mini_interp.run_fragment(
        spvasm,
        inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
        push_constants=OUT_VSHARP,
        memory=memory,
    )
    return memory

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_compare_and_swap_matching(self):
        # Current is 0x11111111, cmp is 0x11111111, swap is 0x22222222 -> new is 0x22222222
        new_val, old_val = ref_cmpst_b32(0x11111111, 0x11111111, 0x22222222)
        self.assertEqual(new_val, 0x22222222)
        self.assertEqual(old_val, 0x11111111)

    def test_compare_and_swap_mismatch(self):
        # Current is 0x11111111, cmp is 0x99999999, swap is 0x22222222 -> new remains 0x11111111
        new_val, old_val = ref_cmpst_b32(0x11111111, 0x99999999, 0x22222222)
        self.assertEqual(new_val, 0x11111111)
        self.assertEqual(old_val, 0x11111111)

    def test_translator_basic_execution_matching(self):
        instructions = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0x11111111",
            "ds_write_b32 v0, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 0x22222222",
            "ds_cmpst_b32 v0, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), 0x22222222)

    def test_translator_basic_execution_mismatch(self):
        instructions = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0x11111111",
            "ds_write_b32 v0, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 0x99999999",  # mismatch cmp
            "v_mov_b32_e32 v3, 0x33333333",  # swap candidate
            "ds_cmpst_b32 v0, v2, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v4, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), 0x11111111)

class RegType(Enum):
    VGPR = auto()


@dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0

    def __post_init__(self):
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of bounds [0, 255]")

    @classmethod
    def vgpr(cls, idx: int) -> Operand:
        return cls(RegType.VGPR, index=idx)

    def to_asm(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class DSCmpStB32Instruction:
    vaddr: Operand
    vcmp: Operand
    vsrc: Operand
    offset: int = 0
    rtn: bool = False
    vdst: Optional[Operand] = None

    def __post_init__(self):
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"Offset {self.offset} out of bounds [0, 65535]")
        if self.rtn and self.vdst is None:
            raise ValueError("Returning form requires vdst operand")
        if not self.rtn and self.vdst is not None:
            raise ValueError("Non-returning form must not have vdst operand")

    def to_asm(self) -> str:
        off_str = f" offset:{self.offset}" if self.offset != 0 else ""
        if self.rtn:
            return f"ds_cmpst_rtn_b32 {self.vdst.to_asm()}, {self.vaddr.to_asm()}, {self.vcmp.to_asm()}, {self.vsrc.to_asm()}{off_str}"
        return f"ds_cmpst_b32 {self.vaddr.to_asm()}, {self.vcmp.to_asm()}, {self.vsrc.to_asm()}{off_str}"


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn1 = DSCmpStB32Instruction(
            vaddr=Operand.vgpr(0),
            vcmp=Operand.vgpr(1),
            vsrc=Operand.vgpr(2),
            offset=16,
        )
        self.assertEqual(insn1.to_asm(), "ds_cmpst_b32 v0, v1, v2 offset:16")

        insn2 = DSCmpStB32Instruction(
            vaddr=Operand.vgpr(0),
            vcmp=Operand.vgpr(1),
            vsrc=Operand.vgpr(2),
            offset=8,
            rtn=True,
            vdst=Operand.vgpr(3),
        )
        self.assertEqual(insn2.to_asm(), "ds_cmpst_rtn_b32 v3, v0, v1, v2 offset:8")

    def test_ast_immutability(self):
        insn = DSCmpStB32Instruction(
            vaddr=Operand.vgpr(0),
            vcmp=Operand.vgpr(1),
            vsrc=Operand.vgpr(2),
        )
        with self.assertRaises(Exception):
            insn.offset = 4  # type: ignore

    def test_ast_validation_errors(self):
        with self.assertRaises(ValueError):
            DSCmpStB32Instruction(
                vaddr=Operand.vgpr(0),
                vcmp=Operand.vgpr(1),
                vsrc=Operand.vgpr(2),
                offset=70000,
            )
        with self.assertRaises(ValueError):
            DSCmpStB32Instruction(
                vaddr=Operand.vgpr(0),
                vcmp=Operand.vgpr(1),
                vsrc=Operand.vgpr(2),
                rtn=True,
                vdst=None,
            )

class LDSState:
    def __init__(self, size_dwords: int = 1024):
        self.size_dwords = size_dwords
        self.data = [0] * size_dwords

    def atomic_cmpst_b32(self, byte_addr: int, cmp_val: int, swap_val: int) -> int:
        idx = ((byte_addr & MASK32) >> 2) & (self.size_dwords - 1)
        new_val, old_val = ref_cmpst_b32(self.data[idx], cmp_val, swap_val)
        self.data[idx] = new_val
        return old_val


class ScalarRegisterState:
    def __init__(self):
        self.vgprs = [0] * 256
        self.scc = 0
        self.vcc = 0


class ScalarLDSInterpreter:
    def __init__(self, lds: Optional[LDSState] = None, regs: Optional[ScalarRegisterState] = None):
        self.lds = lds if lds is not None else LDSState()
        self.regs = regs if regs is not None else ScalarRegisterState()

    def step(self, insn: DSCmpStB32Instruction) -> int:
        base_addr = self.regs.vgprs[insn.vaddr.index]
        eff_addr = (base_addr + insn.offset) & MASK32
        cmp_val = self.regs.vgprs[insn.vcmp.index]
        swap_val = self.regs.vgprs[insn.vsrc.index]
        old = self.lds.atomic_cmpst_b32(eff_addr, cmp_val, swap_val)
        if insn.rtn and insn.vdst is not None:
            self.regs.vgprs[insn.vdst.index] = old
        return old


class TestScalarInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step_and_chaining(self):
        interp = ScalarLDSInterpreter()
        interp.lds.data[0] = 0xAAAAAAAA

        interp.regs.vgprs[0] = 0  # addr
        interp.regs.vgprs[1] = 0xAAAAAAAA  # cmp
        interp.regs.vgprs[2] = 0xBBBBBBBB  # swap
        interp.regs.vgprs[4] = 0xCCCCCCCC  # second swap

        # Step 1: ds_cmpst_rtn_b32 v3, v0, v1, v2 -> match, writes BBBBBBBB, returns AAAAAAAA
        insn1 = DSCmpStB32Instruction(
            vaddr=Operand.vgpr(0),
            vcmp=Operand.vgpr(1),
            vsrc=Operand.vgpr(2),
            rtn=True,
            vdst=Operand.vgpr(3),
        )
        old1 = interp.step(insn1)
        self.assertEqual(old1, 0xAAAAAAAA)
        self.assertEqual(interp.regs.vgprs[3], 0xAAAAAAAA)
        self.assertEqual(interp.lds.data[0], 0xBBBBBBBB)

        # Step 2: ds_cmpst_rtn_b32 v5, v0, v2, v4 -> match, writes CCCCCCCC, returns BBBBBBBB
        insn2 = DSCmpStB32Instruction(
            vaddr=Operand.vgpr(0),
            vcmp=Operand.vgpr(2),
            vsrc=Operand.vgpr(4),
            rtn=True,
            vdst=Operand.vgpr(5),
        )
        old2 = interp.step(insn2)
        self.assertEqual(old2, 0xBBBBBBBB)
        self.assertEqual(interp.regs.vgprs[5], 0xBBBBBBBB)
        self.assertEqual(interp.lds.data[0], 0xCCCCCCCC)

class WaveSize(Enum):
    WAVE32 = 32
    WAVE64 = 64


class SIMDWaveState:
    def __init__(self, size: WaveSize = WaveSize.WAVE32):
        self.size = size
        self.num_lanes = size.value
        self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        self.exec_mask = (1 << self.num_lanes) - 1

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)


class SIMDLDSInterpreter:
    @staticmethod
    def step(insn: DSCmpStB32Instruction, wave: SIMDWaveState, lds: LDSState):
        for lane in range(wave.num_lanes):
            if not wave.is_lane_active(lane):
                continue
            base_addr = wave.vgprs[lane][insn.vaddr.index]
            eff_addr = (base_addr + insn.offset) & MASK32
            cmp_val = wave.vgprs[lane][insn.vcmp.index]
            swap_val = wave.vgprs[lane][insn.vsrc.index]
            old = lds.atomic_cmpst_b32(eff_addr, cmp_val, swap_val)
            if insn.rtn and insn.vdst is not None:
                wave.vgprs[lane][insn.vdst.index] = old


class TestSIMDDivergence(unittest.TestCase):
    def test_wave32_divergence_and_exec_masking(self):
        lds = LDSState()
        wave = SIMDWaveState(WaveSize.WAVE32)
        wave.exec_mask = 0x55555555  # Alternating lanes

        # Initialize LDS slots with lane index
        for i in range(32):
            lds.data[i] = i

        sentinel = 0xDEADBEEF
        for lane in range(32):
            wave.vgprs[lane][0] = lane * 4  # byte address
            wave.vgprs[lane][1] = lane       # cmp val
            wave.vgprs[lane][2] = 0x100 + lane  # swap val
            wave.vgprs[lane][3] = sentinel   # vdst

        insn = DSCmpStB32Instruction(
            vaddr=Operand.vgpr(0),
            vcmp=Operand.vgpr(1),
            vsrc=Operand.vgpr(2),
            rtn=True,
            vdst=Operand.vgpr(3),
        )
        SIMDLDSInterpreter.step(insn, wave, lds)

        for lane in range(32):
            if lane % 2 == 0:  # Active
                self.assertEqual(wave.vgprs[lane][3], lane)
                self.assertEqual(lds.data[lane], 0x100 + lane)
            else:  # Inactive
                self.assertEqual(wave.vgprs[lane][3], sentinel)
                self.assertEqual(lds.data[lane], lane)

    def test_wave64_divergence(self):
        lds = LDSState()
        wave = SIMDWaveState(WaveSize.WAVE64)
        wave.exec_mask = 0x00000000FFFFFFFF  # Lower 32 lanes active

        for i in range(64):
            lds.data[i] = 0x55

        sentinel = 0xCAFEBABE
        for lane in range(64):
            wave.vgprs[lane][0] = lane * 4
            wave.vgprs[lane][1] = 0x55
            wave.vgprs[lane][2] = 0xAA
            wave.vgprs[lane][3] = sentinel

        insn = DSCmpStB32Instruction(
            vaddr=Operand.vgpr(0),
            vcmp=Operand.vgpr(1),
            vsrc=Operand.vgpr(2),
            rtn=True,
            vdst=Operand.vgpr(3),
        )
        SIMDLDSInterpreter.step(insn, wave, lds)

        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][3], 0x55)
            self.assertEqual(lds.data[lane], 0xAA)
        for lane in range(32, 64):
            self.assertEqual(wave.vgprs[lane][3], sentinel)
            self.assertEqual(lds.data[lane], 0x55)

class TestRegisterAliasing(unittest.TestCase):
    def test_alias_vcmp_equals_vsrc(self):
        # cmp val and swap val are identical: no-op write
        instructions = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0x12345678",
            "ds_write_b32 v0, v1",
            "s_waitcnt lgkmcnt(0)",
            "ds_cmpst_b32 v0, v1, v1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), 0x12345678)

    def test_alias_vdst_equals_vcmp(self):
        # In returning form: vdst and vcmp share register
        instructions = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0x55555555",
            "ds_write_b32 v0, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 0xAAAAAAAA",
            "ds_cmpst_rtn_b32 v1, v0, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v1, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), 0x55555555)

@dataclass
class IntegerStatusFlags:
    zero: bool = False
    negative: bool = False


def classify_integer_status(val: int) -> IntegerStatusFlags:
    val &= MASK32
    return IntegerStatusFlags(
        zero=(val == 0),
        negative=bool(val & 0x80000000),
    )


class TestConditionCodesAndStatus(unittest.TestCase):
    def test_vcc_and_scc_preservation(self):
        interp = ScalarLDSInterpreter()
        interp.regs.scc = 1
        interp.regs.vcc = 0x123456789ABCDEF0
        interp.regs.vgprs[0] = 0
        interp.regs.vgprs[1] = 0
        interp.regs.vgprs[2] = 1

        insn = DSCmpStB32Instruction(
            vaddr=Operand.vgpr(0),
            vcmp=Operand.vgpr(1),
            vsrc=Operand.vgpr(2),
        )
        interp.step(insn)

        self.assertEqual(interp.regs.scc, 1)
        self.assertEqual(interp.regs.vcc, 0x123456789ABCDEF0)

    def test_status_flag_classification(self):
        s_zero = classify_integer_status(0)
        self.assertTrue(s_zero.zero)
        self.assertFalse(s_zero.negative)

        s_neg = classify_integer_status(0x80000000)
        self.assertFalse(s_neg.zero)
        self.assertTrue(s_neg.negative)


if __name__ == "__main__":
    unittest.main()
