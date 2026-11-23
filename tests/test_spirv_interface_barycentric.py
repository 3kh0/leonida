"""Member-level PerVertexKHR reflection and linking, with hand-written SPIR-V.

Run with the standard library only:
    python -I tests/test_spirv_interface_barycentric.py
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from spirv_asm import assemble  # noqa: E402
import shader_link_check as link  # noqa: E402
import spirv_interface as spv  # noqa: E402


def block_module(stage, members, *, per_vertex=(), storage=None, base_location=None):
    """Build a complete module; members are (type ID, location) pairs."""
    storage = storage or ("Output" if stage == "vs" else "Input")
    header = "OpCapability Shader\n"
    if per_vertex:
        header += ('OpCapability FragmentBarycentricKHR\n'
                   'OpExtension "SPV_KHR_fragment_shader_barycentric"\n')
    header += "OpMemoryModel Logical GLSL450\n"
    model = "Vertex" if stage == "vs" else "Fragment"
    header += f'OpEntryPoint {model} %main "main" %varying\n'
    if stage == "ps":
        header += "OpExecutionMode %main OriginUpperLeft\n"
    header += "OpDecorate %block Block\n"
    if base_location is not None:
        header += f"OpDecorate %varying Location {base_location}\n"
    for index, (_, location) in enumerate(members):
        if base_location is None:
            header += f"OpMemberDecorate %block {index} Location {location}\n"
        if index in per_vertex:
            header += f"OpMemberDecorate %block {index} PerVertexKHR\n"
    body = """%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%vec = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%two = OpConstant %uint 2
%three = OpConstant %uint 3
%payload = OpTypeArray %vec %two
%vertices = OpTypeArray %vec %three
%payload_vertices = OpTypeArray %payload %three
"""
    body += "%block = OpTypeStruct " + " ".join(shape for shape, _ in members) + "\n"
    body += f"%ptr = OpTypePointer {storage} %block\n%varying = OpVariable %ptr {storage}\n"
    body += """%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""
    return assemble(header + body, target_env="vulkan1.2")


class PerVertexMemberTest(unittest.TestCase):
    def test_member_vertex_array_does_not_consume_locations(self):
        binary = block_module("ps", (("%vertices", 0), ("%vec", 1)), per_vertex=(0,))
        declaration, = spv.reflect_module(binary).inputs
        self.assertTrue(declaration.structure)
        self.assertTrue(declaration.block)
        self.assertTrue(declaration.indexed_by_vertex_or_primitive)
        self.assertEqual([v.type for v in declaration.varyings], [spv.NumericType("f", 32, 4)] * 2)
        self.assertEqual([v.address for v in declaration.varyings], [spv.Address(0), spv.Address(1)])

    def test_strips_only_vertex_dimension_and_keeps_ordinary_arrays(self):
        binary = block_module(
            "ps", (("%payload_vertices", 0), ("%payload", 2)), per_vertex=(0,))
        declaration, = spv.reflect_module(binary).inputs
        expected = spv.AggregateType("array", (spv.NumericType("f", 32, 4),), 2)
        self.assertEqual([v.type for v in declaration.varyings], [expected, expected])
        self.assertEqual([v.address for v in declaration.varyings], [spv.Address(0), spv.Address(2)])
        self.assertEqual(
            sorted({address.location for varying in declaration.varyings for address in varying.occupied()}),
            [0, 1, 2, 3],
        )

    def test_variable_location_advances_by_payload_size(self):
        binary = block_module(
            "ps", (("%payload_vertices", 0), ("%vec", 0)), per_vertex=(0,), base_location=4)
        declaration, = spv.reflect_module(binary).inputs
        self.assertEqual([v.address for v in declaration.varyings], [spv.Address(4), spv.Address(6)])

    def test_per_vertex_members_require_fragment_input_arrays(self):
        for stage, storage in (("vs", "Input"), ("vs", "Output"), ("ps", "Output")):
            with self.subTest(stage=stage, storage=storage):
                binary = block_module(stage, (("%vertices", 0),), per_vertex=(0,), storage=storage)
                with self.assertRaisesRegex(spv.InterfaceError, "PerVertexKHR is only supported on fragment inputs"):
                    spv.reflect_module(binary)
        binary = block_module("ps", (("%vec", 0),), per_vertex=(0,))
        with self.assertRaisesRegex(spv.InterfaceError, "ps requires an outer per-vertex or per-primitive array"):
            spv.reflect_module(binary)

    def test_stage_link_matches_payload_and_preserves_array_mismatches(self):
        for output_type, input_type in (("%vec", "%vertices"), ("%payload", "%payload_vertices")):
            with self.subTest(payload=output_type):
                producer = spv.reflect_module(block_module("vs", ((output_type, 0),)))
                consumer = spv.reflect_module(block_module("ps", ((input_type, 0),), per_vertex=(0,)))
                self.assertEqual(link.compare_stages(producer, consumer), [])

        producer = spv.reflect_module(block_module("vs", (("%vec", 0),)))
        consumer = spv.reflect_module(block_module("ps", (("%vertices", 0),)))
        self.assertEqual([problem.kind for problem in link.compare_stages(producer, consumer)], ["type-mismatch"])
        self.assertFalse(consumer.inputs[0].indexed_by_vertex_or_primitive)

    def test_cli_accepts_barycentric_block_members(self):
        with tempfile.TemporaryDirectory() as directory:
            vertex = Path(directory) / "vertex.spv"
            pixel = Path(directory) / "pixel.spv"
            vertex.write_bytes(block_module("vs", (("%payload", 0), ("%vec", 2))))
            pixel.write_bytes(block_module(
                "ps", (("%payload_vertices", 0), ("%vec", 2)), per_vertex=(0,)))
            result = subprocess.run(
                [sys.executable, "-I", str(ROOT / "tools" / "shader_link_check.py"),
                 "check", "--vs", str(vertex), "--ps", str(pixel)],
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("compatible (1 links checked)", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
