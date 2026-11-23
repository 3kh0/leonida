"""Synthetic stored-pass and batch-report joins. Run with python -I."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "sgd5_pass_link_summary.py"


def stored(vertex="v", pixel="p", geometry=None, mesh=False):
    programs = {"1": pixel, "0": vertex}
    stages = {"1": "ps", "0": "ms" if mesh else "vs"}
    if geometry:
        programs["2"] = geometry
        stages["2"] = "gs"
    return {"programs": programs, "stages": stages}


class JoinTests(unittest.TestCase):
    def run_join(self, rows, pipelines, results):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, data in [("passes", rows), ("manifest", {"pipelines": pipelines}),
                               ("report", {"version": 1, "pipelines": results})]:
                path = Path(directory) / name
                path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
                paths.append(path)
            return subprocess.run([sys.executable, "-I", TOOL, *paths], capture_output=True, text=True)

    def test_stage_roles_order_multiplicity_and_coverage(self):
        triple = stored(geometry="g")
        rows = [triple, dict(triple, raster=3), stored(vertex="g", geometry="v"), stored(),
                stored(mesh=True), stored(mesh=True, geometry="g"), stored(pixel="q"),
                {"programs": {"5": "c"}, "stages": {"5": "cs"}},
                {"programs": {"0": "v"}, "stages": {"0": "vs"}}]
        pipelines = [{"name": "pair", "ps": "p", "vs": "v"},
                     {"name": "original", "ps": "p", "gs": {"shader": "g"}, "vs": "v"},
                     {"name": "swap", "vs": "g", "gs": "v", "ps": "p"},
                     {"name": "mesh", "ms": "v", "ps": "p"},
                     {"name": "other-entry", "vs": {"shader": "v", "entry": "other"}, "ps": "q"}]
        results = [{"name": "mesh", "status": "invalid", "problems": [{"kind": "module-error"}]},
                   {"name": "swap", "status": "compatible", "problems": []},
                   {"name": "original", "status": "incompatible", "problems": [{"kind": "missing-output"}] * 2}]
        run = self.run_join(rows, pipelines, results)
        self.assertEqual(run.returncode, 0, run.stderr)
        summary = json.loads(run.stdout)
        outcomes = summary["outcomes"]
        self.assertEqual(outcomes.pop("incompatible"), {"distinct": 1, "occurrences": 2})
        self.assertEqual(outcomes, dict.fromkeys(
            ["compatible", "invalid", "missing-result", "missing-manifest", "compute", "no-ps", "unsupported"],
            {"distinct": 1, "occurrences": 1}))
        self.assertEqual(summary["diagnostics"], {"missing-output": {"distinct": 2, "occurrences": 4},
                                                  "module-error": {"distinct": 1, "occurrences": 1}})
        self.assertEqual((summary["distinct_chains"], summary["pass_rows"]), (8, 9))

    def test_ambiguous_or_stale_inputs(self):
        pipeline = {"name": "pair", "vs": "v", "ps": "p"}
        result = {"name": "pair", "status": "compatible", "problems": []}
        for pipelines, results, rows in [
            ([pipeline, dict(pipeline, name="alias")], [], []),
            ([pipeline], [result, result], []),
            ([pipeline], [dict(result, name="unknown")], []),
            ([pipeline], [result], [{"programs": {"0": "v"}, "stages": {}}]),
        ]:
            run = self.run_join(rows, pipelines, results)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(run.stdout)


if __name__ == "__main__":
    unittest.main()
