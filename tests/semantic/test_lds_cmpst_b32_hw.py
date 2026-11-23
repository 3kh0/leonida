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


if __name__ == "__main__":
    unittest.main()
