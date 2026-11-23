"""A carve ``index.json`` saved with a UTF-8 BOM is read, not rejected.

A work folder is someone's own copy: re-saving ``index.json`` with Notepad or PowerShell's
``Out-File`` puts a UTF-8 BOM in front of the JSON, and ``json.load`` then refuses the whole
folder with ``Unexpected UTF-8 BOM``. The gamedata tools read that index through
``gamedata_index.DirectorySource``/``ArchiveSource`` and ``gamedata_maps._read_index_rows``.

Run with ``python -I tests/test_gamedata_index_bom.py``. All inputs are hand-built; no game data.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
XML = (b'<?xml version="1.0" encoding="utf-8"?>\n'
       b'<CDataFileMgr__ContentChangeSetDef version="1"><id>fixture</id></CDataFileMgr__ContentChangeSetDef>')


class GamedataIndexBomTest(unittest.TestCase):
    def run_tool(self, tool, carve):
        return subprocess.run([sys.executable, "-I", str(TOOLS / tool), str(carve)],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    def make_carve(self, folder, bom):
        carve = pathlib.Path(folder) / "x.rpf"
        carve.mkdir()
        doc = {"files": [{"off": 0, "size": len(XML), "stored": len(XML), "head": XML[:16].hex()}],
               "summary": {}}
        (carve / "index.json").write_text(json.dumps(doc), encoding="utf-8-sig" if bom else "utf-8")
        (carve / "000000000000.bin").write_bytes(XML)
        return carve

    def assert_read(self, result, expected):
        self.assertNotIn("Traceback", result.stderr, result.stderr)
        self.assertNotIn("UTF-8 BOM", result.stderr, result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(expected, result.stdout, result.stdout)

    def test_index_saved_with_a_bom(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td, bom=True)
            self.assert_read(self.run_tool("gamedata_index.py", carve), "1 payloads")

    def test_xml_saved_with_a_bom(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td, bom=True)
            self.assert_read(self.run_tool("gamedata_xml.py", carve), "1 well-formed")

    def test_graph_saved_with_a_bom(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td, bom=True)
            self.assert_read(self.run_tool("gamedata_graph.py", carve), "1 manifest documents")

    def test_maps_saved_with_a_bom(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td, bom=True)
            self.assert_read(self.run_tool("gamedata_maps.py", carve), "no MHIU or PAMH payloads")

    def test_a_bomless_index_still_reads(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td, bom=False)
            self.assert_read(self.run_tool("gamedata_xml.py", carve), "1 well-formed")
            self.assert_read(self.run_tool("gamedata_maps.py", carve), "no MHIU or PAMH payloads")


if __name__ == "__main__":
    unittest.main()
