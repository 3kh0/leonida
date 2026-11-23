"""tools/bink_play.py: name matching and the name-table cache. Synthetic names only, no game files."""
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bink_play  # noqa: E402

NAMES = [
    ("Stinger/INTRO_A.bk2", 4096, "3840", "2160", "450"),
    ("Stinger/INTRO_B.bk2", 8192, "3840", "2160", "300"),
    ("TV/News_01.bk2", 12288, "1280", "720", "40"),
]


class ListTest(unittest.TestCase):
    def test_filter_accepts_both_path_separators_and_ignores_case(self):
        from io import StringIO

        names = [("Clips/Demo.bk2", 4096, "640", "480", "24"), *NAMES]
        with mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(bink_play, "load_names", return_value=names):
            for query in ("Clips/Demo", r"Clips\Demo", "cLiPs/dEmO", r"cLiPs\dEmO"):
                with self.subTest(query=query), mock.patch.object(sys, "stdout", new_callable=StringIO) as stdout:
                    rc = bink_play.main(["--game-dir", "synthetic", "--list", query])
                    self.assertEqual(rc, 0)
                    self.assertEqual(stdout.getvalue(), "Clips/Demo.bk2  640x480, 24 frames\n")


class PickTest(unittest.TestCase):
    def test_exact_path_wins_case_insensitively_and_with_backslashes(self):
        hit, _ = bink_play.pick(NAMES, "stinger\\intro_a.BK2")
        self.assertEqual(hit[1], 4096)

    def test_a_unique_part_of_the_name_is_enough(self):
        hit, _ = bink_play.pick(NAMES, "news")
        self.assertEqual(hit[0], "TV/News_01.bk2")

    def test_an_ambiguous_part_returns_the_candidates(self):
        hit, candidates = bink_play.pick(NAMES, "intro")
        self.assertIsNone(hit)
        self.assertEqual([c[0] for c in candidates], ["Stinger/INTRO_A.bk2", "Stinger/INTRO_B.bk2"])

    def test_no_match(self):
        self.assertEqual(bink_play.pick(NAMES, "credits"), (None, []))


class CacheTest(unittest.TestCase):
    def test_the_table_is_named_once_and_rebuilt_when_the_archive_changes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            archive, mount = d / "movies.rpf", d / "common.rpf"
            archive.write_bytes(b"x" * 16)
            mount.write_bytes(b"y")
            calls = []

            def fake_names(argv):
                calls.append(argv)
                out = Path(argv[argv.index("--out") + 1])
                out.write_text("source,offset,size,path,confidence,width,height,frames\n"
                               "movies.rpf,4096,10,Stinger/INTRO_A.bk2,anchored,3840,2160,450\n", encoding="utf-8")
                return 0

            with mock.patch.object(bink_play, "cache_path", return_value=d / "cache" / "bink_names.csv"), \
                    mock.patch.object(bink_play.bink_names, "main", side_effect=fake_names):
                first = bink_play.load_names(archive, mount)
                second = bink_play.load_names(archive, mount)
                self.assertEqual(len(calls), 1, "the second call reads the cache")
                self.assertEqual(first, second)
                self.assertEqual(first[0][:2], ("Stinger/INTRO_A.bk2", 4096))
                archive.write_bytes(b"x" * 32)  # a different archive size: the cache is stale
                bink_play.load_names(archive, mount)
                self.assertEqual(len(calls), 2)
                bink_play.load_names(archive, mount, use_cache=False)
                self.assertEqual(len(calls), 3, "--no-cache names again")

    def test_a_different_mount_source_names_the_streams_again(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            archive, old, new = d / "movies.rpf", d / "old.xml", d / "new.xml"
            archive.write_bytes(b"x" * 16)
            old.write_bytes(b"<old/>")
            new.write_bytes(b"<new/>")
            cache = d / "cache" / "bink_names.csv"

            def fake_names(argv):
                name = "OLD.bk2" if Path(argv[argv.index("--mount") + 1]).name == "old.xml" else "NEW.bk2"
                Path(argv[argv.index("--out") + 1]).write_text(
                    "source,offset,size,path,confidence,width,height,frames\n"
                    f"movies.rpf,4096,10,Stinger/{name},anchored,3840,2160,450\n", encoding="utf-8", newline="\n")
                return 0

            with mock.patch.object(bink_play, "cache_path", return_value=cache), \
                    mock.patch.object(bink_play.bink_names, "main", side_effect=fake_names):
                self.assertEqual(bink_play.load_names(archive, old)[0][0], "Stinger/OLD.bk2")
                self.assertEqual(bink_play.load_names(archive, new)[0][0], "Stinger/NEW.bk2",
                                 "names from another mount source are not served from the cache")
                # a table written before the mount source was part of the key is a miss, not a crash
                cache.write_text("# " + bink_play.stamp(archive) + "\npath,offset,width,height,frames\n"
                                 "Stinger/STALE.bk2,4096,3840,2160,450\n", encoding="utf-8", newline="\n")
                self.assertEqual(bink_play.load_names(archive, new)[0][0], "Stinger/NEW.bk2")

    def test_two_lookups_at_once_each_get_their_own_names(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            archive, first, second = d / "movies.rpf", d / "first.xml", d / "second.xml"
            archive.write_bytes(b"x" * 16)
            first.write_bytes(b"<a/>")
            second.write_bytes(b"<b/>")
            ready, done = threading.Event(), threading.Event()
            got = {}

            def fake_names(argv):
                mount = Path(argv[argv.index("--mount") + 1])
                Path(argv[argv.index("--out") + 1]).write_text(
                    "source,offset,size,path,confidence,width,height,frames\n"
                    f"movies.rpf,4096,10,Stinger/{mount.stem}.bk2,anchored,3840,2160,450\n",
                    encoding="utf-8", newline="\n")
                if mount == first:  # hold the first lookup between writing its report and reading it
                    ready.set()
                    done.wait(10)
                return 0

            def look_up(mount):
                try:
                    got[mount.stem] = bink_play.load_names(archive, mount)[0][0]
                except SystemExit as e:
                    got[mount.stem] = str(e)

            with mock.patch.object(bink_play, "cache_path", return_value=d / "cache" / "bink_names.csv"), \
                    mock.patch.object(bink_play.bink_names, "main", side_effect=fake_names):
                worker = threading.Thread(target=look_up, args=(first,))
                worker.start()
                self.assertTrue(ready.wait(10))
                try:
                    look_up(second)
                finally:
                    done.set()
                    worker.join()
            self.assertEqual(got, {"first": "Stinger/first.bk2", "second": "Stinger/second.bk2"})
            self.assertEqual(sorted(p.name for p in (d / "cache").iterdir()), ["bink_names.csv"])


class FindExeTest(unittest.TestCase):
    """--exe is a path someone typed. A wrong one should read like the tool's own hint, not a traceback."""

    def test_an_existing_file_is_used_as_given(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "bink_play.exe"
            exe.write_bytes(b"MZ\x90\x00")
            self.assertEqual(bink_play.find_exe(str(exe)), exe)

    def test_a_missing_path_names_itself_and_gives_the_hint(self):
        with tempfile.TemporaryDirectory() as d:
            gone = Path(d) / "nope" / "bink_play.exe"
            with self.assertRaises(SystemExit) as cm:
                bink_play.find_exe(str(gone))
            self.assertIn(str(gone), str(cm.exception))
            self.assertIn("build the runtime", str(cm.exception))

    def test_a_folder_is_not_an_executable(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                bink_play.find_exe(d)
            self.assertIn("is not a file", str(cm.exception))

    def test_the_default_search_keeps_its_message(self):
        with mock.patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(SystemExit) as cm:
                bink_play.find_exe(None)
        self.assertIn("bink_play.exe not found", str(cm.exception))

    def test_a_bare_name_on_path_is_resolved(self):
        # The merge base handed a bare name to subprocess, which finds it on PATH; keep that.
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "bink_play.exe"
            exe.write_bytes(b"MZ\x90\x00")
            exe.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": d}):
                self.assertEqual(bink_play.find_exe("bink_play.exe"), exe)

    def test_a_bare_name_not_on_path_gives_the_hint(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"PATH": d}):
                with self.assertRaises(SystemExit) as cm:
                    bink_play.find_exe("bink_play.exe")
            self.assertIn("is not a file", str(cm.exception))

    def test_an_explicit_relative_path_is_not_a_path_lookup(self):
        # "./name" is a typed path, not a bare command; a missing one keeps the diagnostic even when
        # the same name sits on PATH.
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "bink_play.exe"
            exe.write_bytes(b"MZ\x90\x00")
            exe.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": d}):
                with self.assertRaises(SystemExit) as cm:
                    bink_play.find_exe("./bink_play.exe")
            self.assertIn("is not a file", str(cm.exception))


class PlayTest(unittest.TestCase):
    """The run itself: a bad --exe reaches subprocess.call, which is where the traceback came from."""

    def test_a_bad_exe_exits_with_the_hint_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as d:
            game = Path(d) / "game"
            game.mkdir()
            (game / "movies.rpf").write_bytes(b"\x00" * 32)
            (game / "common.rpf").write_bytes(b"\x00" * 32)
            with mock.patch.object(bink_play, "load_names", return_value=NAMES), \
                    mock.patch.object(bink_play.subprocess, "call") as called:
                with self.assertRaises(SystemExit):
                    bink_play.main(["INTRO_A", "--game-dir", str(game), "--exe", str(Path(d) / "gone.exe")])
                called.assert_not_called()


if __name__ == "__main__":
    unittest.main()
