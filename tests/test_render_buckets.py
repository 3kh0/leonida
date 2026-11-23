"""Synthetic tests for tools/render_buckets.py (no game data)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import render_buckets as rb  # noqa: E402


def model(name, buckets, verts=10, tris=4):
    return {"Name": name, "Verts": verts, "Tris": tris,
            "Materials": [{"Shader": f"{i:08x}", "RenderBucket": b} for i, b in enumerate(buckets)]}


def write_manifest(folder, items, count=None, extra_before=True):
    os.makedirs(folder, exist_ok=True)
    doc = {"src": "synthetic"}
    if extra_before:
        doc["Count"] = len(items) if count is None else count
    doc["Items"] = items
    doc["ModelDecodeFailures"] = []
    path = os.path.join(folder, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return path


class StreamTests(unittest.TestCase):
    def test_items_are_read_across_small_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = [model(f"m{i}", [i % 7, 0x80 | 2]) for i in range(25)]
            path = write_manifest(tmp, items)
            for chunk in (5, 7, 64, rb.CHUNK):
                gen = rb.iter_items(path, chunk)
                got = []
                while True:
                    try:
                        got.append(next(gen))
                    except StopIteration as stop:
                        count = stop.value
                        break
                self.assertEqual(got, items, chunk)
                self.assertEqual(count, 25)

    def test_empty_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(tmp, [])
            self.assertEqual(list(rb.iter_items(path, 3)), [])

    def test_missing_items_array_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manifest.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"Count": 0}, f)
            with self.assertRaises(ValueError):
                list(rb.iter_items(path))


class CountTests(unittest.TestCase):
    def test_buckets_flag_outside_and_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = model("a", [0, 1, 0x80 | 2, 6, 7])
            m["Materials"].append({"Shader": "x", "RenderBucket": "3"})   # not an integer
            m["Materials"].append({"Shader": "y"})                         # missing
            path = write_manifest(os.path.join(tmp, "arc", "models"), [m])
            per, total, problems = rb.count([("arc", path)])
            self.assertEqual(problems, [])
            self.assertEqual(total["materials"], 5)
            self.assertEqual(total["bad"], 2)
            self.assertEqual(dict(total["low"]), {0: 1, 1: 1, 2: 1, 6: 1, 7: 1})
            self.assertEqual(total["flag"], 1)
            self.assertEqual(dict(total["flag_by_low"]), {2: 1})
            self.assertEqual(total["outside"], 1)

    def test_unique_models_count_once_across_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = model("shared", [0, 2])
            p1 = write_manifest(os.path.join(tmp, "one", "models"), [shared, model("only1", [1])])
            p2 = write_manifest(os.path.join(tmp, "two", "models"), [shared, shared, model("only2", [6])])
            per, total, _ = rb.count([("one", p1), ("two", p2)])
            self.assertEqual(total["models"], 5)
            self.assertEqual(total["materials"], 8)
            self.assertEqual(total["unique_models"], 3)
            self.assertEqual(total["unique_materials"], 4)
            self.assertEqual(dict(total["unique_low"]), {0: 1, 2: 1, 1: 1, 6: 1})
            # same name, different geometry: a different model
            p3 = write_manifest(os.path.join(tmp, "three", "models"), [model("shared", [0, 2], verts=11)])
            _, total, _ = rb.count([("one", p1), ("three", p3)])
            self.assertEqual(total["unique_models"], 3)

    def test_distinct_shaders_per_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = [{"Name": "a", "Materials": [{"Shader": "AAAA0001", "RenderBucket": 4},
                                                 {"Shader": "aaaa0001", "RenderBucket": 0x80 | 4},
                                                 {"Shader": "bbbb0002", "RenderBucket": 0}]}]
            path = write_manifest(os.path.join(tmp, "arc", "models"), items)
            _, total, _ = rb.count([("arc", path)])
            out = rb.plain(total)
            self.assertEqual(out["distinct_shaders"], {0: 1, 4: 1})
            self.assertEqual(out["shaders_if_few"][4], ["aaaa0001"])

    def test_count_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(os.path.join(tmp, "arc", "models"), [model("a", [0])], count=2)
            _, _, problems = rb.count([("arc", path)])
            self.assertEqual(len(problems), 1)
            self.assertIn("1 items read", problems[0])


class CliTests(unittest.TestCase):
    def run_cli(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = rb.main(args)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_directory_finds_every_archive_including_dot_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(os.path.join(tmp, "plain", "models"), [model("a", [0])])
            write_manifest(os.path.join(tmp, ".dotted", "models"), [model("b", [6])])
            os.makedirs(os.path.join(tmp, "nomodels"))
            labels = [label for label, _ in rb.manifests(tmp)]
            self.assertEqual(sorted(labels), [".dotted", "plain"])
            code, out, _ = self.run_cli([tmp])
            self.assertEqual(code, 0)
            self.assertIn("total: 2 models, 2 materials", out)

    def test_json_report_and_refusal_to_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(os.path.join(tmp, "arc", "models"), [model("a", [0x80 | 1])])
            report = os.path.join(tmp, "report.json")
            code, _, _ = self.run_cli([path, "--json", report])
            self.assertEqual(code, 0)
            with open(report, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["total"]["flag"], 1)
            self.assertEqual(data["total"]["low"], {"1": 1})
            code, _, err = self.run_cli([path, "--json", path])
            self.assertNotEqual(code, 0)
            self.assertIn("overwrite", err)

    def test_mismatch_sets_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(os.path.join(tmp, "arc", "models"), [model("a", [0])], count=5)
            code, _, err = self.run_cli([tmp])
            self.assertEqual(code, 1)
            self.assertIn("Count says 5", err)


if __name__ == "__main__":
    unittest.main()
