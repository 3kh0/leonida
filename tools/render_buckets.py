"""Count the material render buckets over every ``rpf9dump models`` manifest, streaming each manifest's ``Items``.

``docs/models.md`` ("Shader render buckets") describes ``RenderBucket`` (byte at shader +0x39) from a sample of
3,000 models. This tool counts it over whole exports: for every material, ``RenderBucket & 0x7F`` and bit 7 (a
separate flag), any value outside 0-6, and the number of distinct material shader hashes per bucket (a material
without a shader hash is counted separately, not as a hash). Counts are given per material row and per unique model.
A unique model is the model name together with its vertex and triangle counts and the ordered list of (shader hash,
render bucket) of its materials: a model repeated with the same list in several dictionaries or archives counts once,
the same model with its materials in another order counts again, and two different models that agree on all of it
would merge. Per archive, the unique figures count the models first seen in that archive (archives in listing order),
so they add up to the overall unique figures.

Input is one or more ``manifest.json`` files, or a directory: ``<dir>/<archive>/models/manifest.json`` is used for
every archive folder (found with ``os.listdir``, so no name is skipped), and ``<dir>`` itself if it holds a
``manifest.json``. The manifests are read incrementally (the levels manifest is several GB), item by item, and each
archive's item total is checked against the manifest's ``Count``. A material whose ``RenderBucket`` is missing or not
an integer is counted on its own line rather than ending the run.

    python -I tools/render_buckets.py <export dir | manifest.json> [...] [--json FILE]

``--json`` writes the totals (per archive and overall); it may not overwrite an input. Prints counts only: no model,
texture or shader names. Standard library only.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

BUCKET_NAMES = {0: "opaque", 1: "alpha", 2: "decal", 3: "cutout", 4: "no splash", 5: "no water", 6: "water"}
CHUNK = 8 << 20
_WS = " \t\r\n"


def _count_after_items(f, buf, pos, chunk, dec):
    """The top-level ``Count`` among the keys that follow the ``Items`` array, or None.

    Parses the remaining members of the top-level object one at a time (a nested ``Count`` inside a later value is
    not a top-level count), reading more of the file as needed."""
    def value_at(pos):
        nonlocal buf
        while True:
            try:
                val, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                val, end = None, None
            if end is not None and end < len(buf):
                return val, end
            # A value that reaches the end of the buffer may go on in the next read (the digits of a number split
            # between reads were accepted as a shorter number): read on before accepting it.
            more = f.read(chunk)
            if not more:
                if end is None:
                    raise ValueError("truncated JSON after the Items array")
                return val, end
            buf, pos = buf[pos:] + more, 0

    def skip(pos):
        nonlocal buf
        while True:
            while pos < len(buf) and buf[pos] in _WS:
                pos += 1
            if pos < len(buf):
                return pos
            more = f.read(chunk)
            if not more:
                return pos
            buf, pos = buf[pos:] + more, 0

    count = None
    while True:
        pos = skip(pos)
        if pos >= len(buf) or buf[pos] == "}":
            return count
        if buf[pos] == ",":
            pos += 1
            continue
        key, pos = value_at(pos)
        pos = skip(pos)
        if pos >= len(buf) or buf[pos] != ":":
            return count
        val, pos = value_at(skip(pos + 1))
        if key == "Count" and isinstance(val, int) and not isinstance(val, bool):
            count = val


def iter_items(path, chunk=CHUNK):
    """Yield the objects of the top-level ``"Items": [...]`` array of a JSON file without loading it whole.

    Also returns (as the generator's value) the top-level ``Count`` when it appears before ``Items``."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        start = -1
        while start < 0:
            more = f.read(chunk)
            if not more:
                raise ValueError(f"{path}: no top-level \"Items\" array")
            buf += more
            m = re.search(r'"Items"\s*:\s*\[', buf)
            if m:
                start = m.end()
        head = buf[:start]
        cm = re.search(r'"Count"\s*:\s*(\d+)', head)
        count = int(cm.group(1)) if cm else None
        pos = start
        while True:
            while True:  # skip whitespace and the separating comma
                while pos < len(buf) and buf[pos] in _WS:
                    pos += 1
                if pos < len(buf) and buf[pos] == ",":
                    pos += 1
                    continue
                if pos < len(buf):
                    break
                more = f.read(chunk)
                if not more:
                    raise ValueError(f"{path}: \"Items\" array is not closed")
                buf, pos = buf[pos:] + more, 0
            if buf[pos] == "]":
                if count is None:  # Count may follow Items: read the rest of the top-level object, key by key
                    count = _count_after_items(f, buf, pos + 1, chunk, dec)
                return count
            while True:
                try:
                    obj, end = dec.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    more = f.read(chunk)
                    if not more:
                        raise
                    buf, pos = buf[pos:] + more, 0
            yield obj
            pos = end
            if pos > chunk:  # keep the buffer small
                buf, pos = buf[pos:], 0


def manifests(target):
    """(archive label, manifest path) pairs for a manifest file or an export directory."""
    if os.path.isfile(target):
        return [(os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(target)))) or target, target)]
    found = []
    own = os.path.join(target, "manifest.json")
    if os.path.isfile(own):
        found.append((os.path.basename(os.path.abspath(target)), own))
    for name in sorted(os.listdir(target)):
        p = os.path.join(target, name, "models", "manifest.json")
        if os.path.isfile(p):
            found.append((name, p))
    return found


def new_totals():
    return {"models": 0, "materials": 0, "low": Counter(), "flag": 0, "flag_by_low": Counter(), "outside": 0,
            "bad": 0, "unique_models": 0, "unique_materials": 0, "unique_low": Counter(), "unique_flag": 0,
            "shaders": {}, "no_shader": 0}


def count(paths, chunk=CHUNK):
    """Totals per archive and overall. Returns (per_archive, overall, problems)."""
    per, problems = {}, []
    overall = new_totals()
    seen = set()
    for label, path in paths:
        t = new_totals()
        gen = iter_items(path, chunk)
        declared = None
        while True:
            try:
                item = next(gen)
            except StopIteration as stop:
                declared = stop.value
                break
            t["models"] += 1
            sig = []
            for m in item.get("Materials") or []:
                b = m.get("RenderBucket")
                if not isinstance(b, int) or isinstance(b, bool):
                    t["bad"] += 1
                    sig.append((m.get("Shader"), None))
                    continue
                t["materials"] += 1
                low = b & 0x7F
                t["low"][low] += 1
                if b & 0x80:
                    t["flag"] += 1
                    t["flag_by_low"][low] += 1
                if low > 6 or b > 0xFF or b < 0:
                    t["outside"] += 1
                shader = m.get("Shader")
                if isinstance(shader, str) and shader.strip():
                    t["shaders"].setdefault(low, set()).add(shader.strip().lower())
                else:
                    t["no_shader"] += 1
                sig.append((m.get("Shader"), b))
            key = (item.get("Name"), item.get("Verts"), item.get("Tris"), tuple(sig))
            if key not in seen:
                seen.add(key)
                for agg in (t, overall):
                    agg["unique_models"] += 1
                for _, b in sig:
                    if isinstance(b, int):
                        for agg in (t, overall):
                            agg["unique_materials"] += 1
                            agg["unique_low"][b & 0x7F] += 1
                            if b & 0x80:
                                agg["unique_flag"] += 1
        if declared is not None and declared != t["models"]:
            problems.append(f"{label}: {t['models']} items read, manifest Count says {declared}")
        t["declared_count"] = declared
        per[label] = t
        for k in ("models", "materials", "flag", "outside", "bad", "no_shader"):
            overall[k] += t[k]
        overall["low"].update(t["low"])
        for low, hashes in t["shaders"].items():
            overall["shaders"].setdefault(low, set()).update(hashes)
        overall["flag_by_low"].update(t["flag_by_low"])
    return per, overall, problems


def plain(t):
    out = {k: (dict(sorted(v.items())) if isinstance(v, Counter) else v) for k, v in t.items() if k != "shaders"}
    # distinct shader hashes per bucket; the hashes themselves only where a bucket has a handful
    out["distinct_shaders"] = {low: len(h) for low, h in sorted(t["shaders"].items())}
    out["shaders_if_few"] = {low: sorted(h) for low, h in sorted(t["shaders"].items()) if len(h) <= 5}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="+", help="an export directory or rpf9dump models manifest.json file(s)")
    ap.add_argument("--json", metavar="FILE", help="write the totals as JSON")
    a = ap.parse_args(argv)
    paths = []
    for t in a.targets:
        if not os.path.exists(t):
            ap.error(f"{t}: not found")
        paths += manifests(t)
    if not paths:
        ap.error("no rpf9dump models manifest.json found")
    if a.json and any(same_report_path(a.json, p) for _, p in paths):
        ap.error(f"--json {a.json} would overwrite an input")
    per, overall, problems = count(paths)
    for label, t in per.items():
        print(f"{label}: {t['models']} models (Count {t['declared_count']}), {t['materials']} materials, "
              f"flag {t['flag']}, outside 0-6 {t['outside']}, unreadable bucket {t['bad']}")
    print(f"total: {overall['models']} models, {overall['materials']} materials; "
          f"unique models {overall['unique_models']} with {overall['unique_materials']} materials")
    print("bucket  name        materials   unique-model materials   bit 7 set   distinct shaders")
    for low in sorted(set(overall["low"]) | set(overall["unique_low"])):
        print(f"{low:>6}  {BUCKET_NAMES.get(low, '?'):<10}  {overall['low'][low]:>9}   {overall['unique_low'][low]:>22}   "
              f"{overall['flag_by_low'][low]:>9}   {len(overall['shaders'].get(low, ())):>16}")
    print(f"bit 7 set: {overall['flag']} materials ({overall['unique_flag']} in unique models); "
          f"outside 0-6: {overall['outside']}; unreadable bucket: {overall['bad']}; "
          f"materials without a shader hash: {overall['no_shader']}")
    for p in problems:
        print("problem:", p, file=sys.stderr)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"archives": {k: plain(v) for k, v in per.items()}, "total": plain(overall),
                       "problems": problems}, f, indent=1)
        print("wrote", a.json)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
