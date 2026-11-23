from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_orn2_saveexec_b64.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_orn2_saveexec_b64 instruction belongs to SOP1 format.
  Instruction:
    s_orn2_saveexec_b64 sdst, ssrc0

  Operands:
    sdst  = 64-bit destination SGPR pair (or EXEC) to receive the previous EXEC mask
    ssrc0 = 64-bit source operand: SGPR pair s[2k:2k+1], literal 64-bit constant, VCC, or EXEC

  Semantics:
    sdst = EXEC
    EXEC = ssrc0 | ~EXEC
    SCC = 1 if EXEC != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
ALL_LANES_ACTIVE = 0xFFFFFFFFFFFFFFFF
ALL_LANES_INACTIVE = 0x0000000000000000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit integer range [0, 2^64 - 1]."""
    return val & MASK64


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_orn2_saveexec_b64(ssrc0: int, old_exec: int) -> tuple[int, int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_orn2_saveexec_b64 instruction.

    Args:
        ssrc0: 64-bit source operand (uint64)
        old_exec: 64-bit current EXEC mask before instruction execution (uint64)

    Returns:
        tuple (new_exec, sdst, scc):
          - new_exec: updated 64-bit EXEC mask = ssrc0 | ~old_exec
          - sdst: saved previous EXEC mask = old_exec
          - scc: scalar condition code = 1 if new_exec != 0 else 0
    """
    s0 = u64(ssrc0)
    old_e = u64(old_exec)
    sdst = old_e
    new_exec = u64(s0 | (~old_e & MASK64))
    scc = 1 if new_exec != 0 else 0
    return new_exec, sdst, scc


class TestSaluOrn2SaveexecB64Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_orn2_saveexec_b64."""

    def test_oracle_all_ones_exec(self):
        # If old_exec = all 1s, ~old_exec = 0, so new_exec = ssrc0 | 0 = ssrc0
        s0 = 0x123456789ABCDEF0
        old_exec = ALL_LANES_ACTIVE
        new_exec, sdst, scc = ref_orn2_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, s0)
        self.assertEqual(sdst, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_all_zeros_exec(self):
        # If old_exec = 0, ~old_exec = all 1s, so new_exec = ssrc0 | all 1s = all 1s
        s0 = 0x0
        old_exec = ALL_LANES_INACTIVE
        new_exec, sdst, scc = ref_orn2_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, ALL_LANES_ACTIVE)
        self.assertEqual(sdst, ALL_LANES_INACTIVE)
        self.assertEqual(scc, 1)

    def test_oracle_zero_s0_all_ones_exec_produces_zero_exec(self):
        # The unique condition where new_exec becomes 0: s0 == 0 and old_exec == ~0
        s0 = 0x0
        old_exec = ALL_LANES_ACTIVE
        new_exec, sdst, scc = ref_orn2_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, ALL_LANES_ACTIVE)
        self.assertEqual(scc, 0)  # SCC is 0 when new_exec == 0




class Opcode(Enum):
    S_ORN2_SAVEEXEC_B64 = auto()


class OperandKind(Enum):
    SGPR = auto()
    EXEC = auto()
    VCC = auto()
    LITERAL64 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    count: int = 2
    value: int = 0

    @classmethod
    def sgpr(cls, index: int, count: int = 2) -> ASTOperand:
        if count != 2:
            raise ValueError(f"s_orn2_saveexec_b64 requires 64-bit register pair (count=2), got {count}")
        if index < 0 or index > 104 or (index + count) > 106:
            raise ValueError(f"SGPR index out of range: s[{index}:{index + count - 1}]")
        if index % 2 != 0:
            raise ValueError(f"SGPR pair must be 64-bit aligned (even index), got s{index}")
        return cls(kind=OperandKind.SGPR, index=index, count=count)

    @classmethod
    def exec_mask(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC, count=2)

    @classmethod
    def vcc(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC, count=2)

    @classmethod
    def literal64(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL64, count=2, value=u64(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, count=2, value=u64(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.S_ORN2_SAVEEXEC_B64:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR, OperandKind.EXEC):
            raise TypeError(f"Destination must be SGPR pair or EXEC, got {self.dst.kind}")
        if self.dst.count != 2:
            raise ValueError(f"Destination must be 64-bit (2 registers), got {self.dst.count}")
        if self.src0.count != 2:
            raise ValueError(f"Source must be 64-bit operand, got {self.src0.count}")


class TestSaluOrn2SaveexecB64AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ORN2_SAVEEXEC_B64,
            dst=ASTOperand.sgpr(4, 2),
            src0=ASTOperand.sgpr(6, 2),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.S_ORN2_SAVEEXEC_B64)
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.index, 4)
        self.assertEqual(inst.src0.index, 6)

    def test_ast_rejects_unaligned_sgpr_pair(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(5, 2)  # Odd index rejected for 64-bit pair

    def test_ast_rejects_single_sgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(4, 1)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ORN2_SAVEEXEC_B64,
            dst=ASTOperand.literal64(0x123),
            src0=ASTOperand.sgpr(6, 2),
        )
        with self.assertRaises(TypeError):
            inst.validate()




class ScalarRegisterState:
    """Simulated state of GCN/RDNA Scalar Registers (SGPRs, EXEC, and SCC)."""

    def __init__(self, exec_mask: int = ALL_LANES_ACTIVE):
        self.sgpr = [0] * 106
        self.exec_mask = u64(exec_mask)
        self.scc = 0

    def get_sgpr_64(self, index: int) -> int:
        if index % 2 != 0 or index + 1 >= len(self.sgpr):
            raise ValueError(f"Invalid SGPR pair index: s{index}")
        lo = self.sgpr[index] & MASK32
        hi = self.sgpr[index + 1] & MASK32
        return (hi << 32) | lo

    def set_sgpr_64(self, index: int, val: int) -> None:
        if index % 2 != 0 or index + 1 >= len(self.sgpr):
            raise ValueError(f"Invalid SGPR pair index: s{index}")
        v = u64(val)
        self.sgpr[index] = v & MASK32
        self.sgpr[index + 1] = (v >> 32) & MASK32

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        # Read source operand
        if inst.src0.kind == OperandKind.SGPR:
            src_val = self.get_sgpr_64(inst.src0.index)
        elif inst.src0.kind == OperandKind.EXEC:
            src_val = self.exec_mask
        elif inst.src0.kind in (OperandKind.LITERAL64, OperandKind.INLINE_CONST):
            src_val = inst.src0.value
        else:
            raise ValueError(f"Unsupported operand kind for scalar execution: {inst.src0.kind}")

        new_exec, sdst, scc = ref_orn2_saveexec_b64(src_val, self.exec_mask)
        if inst.dst.kind == OperandKind.SGPR:
            self.set_sgpr_64(inst.dst.index, sdst)

        self.exec_mask = new_exec
        self.scc = scc


class TestSaluOrn2SaveexecB64ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_basic_execution(self):
        state = ScalarRegisterState(exec_mask=0x00000000FFFFFFFF)
        state.set_sgpr_64(6, 0x1122334455667788)
        inst = ASTInstruction(
            opcode=Opcode.S_ORN2_SAVEEXEC_B64,
            dst=ASTOperand.sgpr(4, 2),
            src0=ASTOperand.sgpr(6, 2),
        )
        state.execute(inst)
        self.assertEqual(state.get_sgpr_64(4), 0x00000000FFFFFFFF)
        expected_new_exec = 0x1122334455667788 | (~0x00000000FFFFFFFF & MASK64)
        self.assertEqual(state.exec_mask, expected_new_exec)
        self.assertEqual(state.scc, 1)

    def test_scalar_interpreter_literal_source(self):
        state = ScalarRegisterState(exec_mask=0xF0F0F0F00F0F0F0F)
        inst = ASTInstruction(
            opcode=Opcode.S_ORN2_SAVEEXEC_B64,
            dst=ASTOperand.sgpr(0, 2),
            src0=ASTOperand.literal64(0x0000FFFF0000FFFF),
        )
        state.execute(inst)
        self.assertEqual(state.get_sgpr_64(0), 0xF0F0F0F00F0F0F0F)
        expected = 0x0000FFFF0000FFFF | (~0xF0F0F0F00F0F0F0F & MASK64)
        self.assertEqual(state.exec_mask, expected)
        self.assertEqual(state.scc, 1)




class Wave64SIMDState:
    """Simulated Wave64 multi-lane divergence and active lane state tracking."""

    def __init__(self, exec_mask: int = ALL_LANES_ACTIVE):
        self.exec_mask = u64(exec_mask)

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < 64):
            raise IndexError(f"Lane {lane} out of range [0, 63]")
        return bool((self.exec_mask >> lane) & 1)

    def active_lanes_mask(self) -> int:
        return self.exec_mask

    def execute(self, inst: ASTInstruction, scalar_state: ScalarRegisterState) -> None:
        scalar_state.exec_mask = self.exec_mask
        scalar_state.execute(inst)
        self.exec_mask = scalar_state.exec_mask


class TestSaluOrn2SaveexecB64SIMDDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence, lane activation, and translator verification."""

    def test_inactive_lanes_turn_active_under_orn2(self):
        # Under s_orn2_saveexec_b64, a lane that was inactive in old EXEC (0) has ~EXEC = 1,
        # so it becomes active in new EXEC regardless of ssrc0!
        initial_exec = 0x00000000FFFFFFFF  # lanes 0..31 active, lanes 32..63 inactive
        wave = Wave64SIMDState(exec_mask=initial_exec)
        scalar = ScalarRegisterState(exec_mask=initial_exec)
        scalar.set_sgpr_64(8, 0x0)  # ssrc0 = 0

        inst = ASTInstruction(
            opcode=Opcode.S_ORN2_SAVEEXEC_B64,
            dst=ASTOperand.sgpr(2, 2),
            src0=ASTOperand.sgpr(8, 2),
        )
        wave.execute(inst, scalar)

        # Inactive lanes 32..63 should now be active!
        for lane in range(32, 64):
            self.assertTrue(wave.is_lane_active(lane), f"Lane {lane} should have turned active")
        # Lanes 0..31 had ~EXEC = 0 and s0 = 0, so they turn off
        for lane in range(0, 32):
            self.assertFalse(wave.is_lane_active(lane), f"Lane {lane} should have turned off")

    def test_translator_ps_translate_lines(self):
        # Exercises tools/rdna2spirv.py to verify s_orn2_saveexec_b64 produces correct SPIR-V instructions
        lines = [
            "s_mov_b64 s[6:7], 0x1234567887654321",
            "s_orn2_saveexec_b64 s[4:5], s[6:7]",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_salu_orn2_saveexec_hw", lines)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpBitwiseOr", spv)
        self.assertIn("OpNot", spv)


if __name__ == "__main__":
    unittest.main()
