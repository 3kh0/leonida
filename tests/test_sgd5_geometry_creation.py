"""Synthetic coverage of weighted Vulkan creation outcomes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sgd5_geometry_creation import checker_results, measure


def stored(gs):
    return {"programs": {"0": "v", "1": "p", "2": gs},
            "stages": {"0": "vs", "1": "ps", "2": "gs"}}


class GeometryCreationTests(unittest.TestCase):
    def test_weighted_results(self):
        def fake_checker(argv, **kwargs):
            rows = json.loads(Path(argv[3]).read_text())["pipelines"]
            self.assertEqual([r["gs"] for r in rows], ["a", "b", "c", "d"])
            output = ("PIPELINE FAIL 1 (graphics, -3)\nSKIPPED 2 (gs: unsupported feature)\n"
                      "GRAPHICS OK 0\nMODULE FAIL d (gs in 3: invalid SPIR-V size)\n")
            return subprocess.CompletedProcess(argv, 1, stdout=output)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for stem in "abcdvpz":
                (directory / (stem + ".spv")).touch()
            (directory / "index.json").write_text(json.dumps(dict.fromkeys("abcdvpx", {})),
                                                  encoding="utf-8", newline="\n")
            rows = [stored(gs) for gs in "aabbcddxxz"]
            rows.append({"programs": {"0": "v"}, "stages": {"0": "vs"}})
            with patch("sgd5_geometry_creation.subprocess.run", side_effect=fake_checker):
                summary, triples = measure(rows, directory, "checker")
        self.assertEqual(summary["geometry_occurrences"], 10)
        self.assertEqual(summary["pass_rows"], 11)
        self.assertEqual(summary["distinct_triples"], 6)
        expected = {"created": 2, "pipeline-failed": 2, "skipped": 1, "module-failed": 2,
                    "missing-module": 2, "missing-index-entry": 1}
        self.assertEqual(summary["outcomes"], {key: {"distinct": 1, "occurrences": count}
                                               for key, count in expected.items()})
        self.assertEqual(triples[2]["reason"], "gs: unsupported feature")
        self.assertEqual(triples[4]["stages"], ["gs"])

    def test_untrustworthy_results(self):
        manifest = [{"name": "0", "vs": "v", "gs": "g", "ps": "p"}]
        for output in ("", "GRAPHICS OK 1", "GRAPHICS OK 0\nGRAPHICS OK 0",
                       "MODULE FAIL v (gs in 0: invalid SPIR-V size)"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                checker_results(output, manifest)


if __name__ == "__main__":
    unittest.main()
