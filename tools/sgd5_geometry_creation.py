"""Measure Vulkan graphics pipeline creation for the stored SGD5 VS/GS/PS triples."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sgd5_pass_link_summary import add_counts, pass_chain, read_json


def checker_results(stdout, manifest):
    """Accept one explicit terminal record per chain, in any worker order."""
    results = {}
    expected = {row["name"]: row for row in manifest}
    patterns = (
        (r"GRAPHICS OK (\d+)", "created"),
        (r"PIPELINE FAIL (\d+) \(graphics, (-?\d+)\)", "pipeline-failed"),
        (r"SKIPPED (\d+) \((.+)\)", "skipped"),
        (r"MODULE FAIL (.+) \((vs|gs|ps) in (\d+): (.+)\)", "module-failed"),
    )
    for line in stdout.splitlines():
        for pattern, status in patterns:
            match = re.fullmatch(pattern, line)
            if not match:
                continue
            name = match[3] if status == "module-failed" else match[1]
            if name not in expected or name in results:
                raise ValueError("unknown or repeated checker outcome")
            if status == "module-failed" and expected[name][match[2]] != match[1]:
                raise ValueError("module failure disagrees with the manifest")
            results[name] = {"status": status}
            if status == "pipeline-failed":
                results[name]["vk_result"] = int(match[2])
            elif status in ("skipped", "module-failed"):
                results[name]["reason"] = match.groups()[-1]
            break
        else:
            if line.startswith(("GRAPHICS OK ", "PIPELINE FAIL ", "SKIPPED ", "MODULE FAIL ")):
                raise ValueError("unrecognized checker outcome")
    if results.keys() != expected.keys():
        raise ValueError(f"incomplete checker output ({len(results)}/{len(expected)} chains)")
    return results


def measure(rows, directory, checker):
    if not directory.is_dir():
        raise ValueError("shader directory does not exist")
    index = read_json(directory / "index.json")
    if not isinstance(index, dict) or not isinstance(rows, list):
        raise ValueError("expected a shader index object and a list of stored passes")
    counts = Counter()
    for row in rows:
        chain = pass_chain(row)
        if tuple(stage for stage, _, _ in chain) == ("vs", "gs", "ps"):
            counts[tuple(stem for _, stem, _ in chain)] += 1
    triples = []
    manifest = []
    for number, (chain, occurrences) in enumerate(sorted(counts.items())):
        stages = dict(zip(("vs", "gs", "ps"), chain))
        triple = {**stages, "occurrences": occurrences}
        triples.append(triple)
        missing = [stage for stage, stem in stages.items() if not (directory / (stem + ".spv")).is_file()]
        unindexed = [stage for stage, stem in stages.items() if stem not in index]
        if missing:
            triple.update(status="missing-module", stages=missing)
        elif unindexed:
            triple.update(status="missing-index-entry", stages=unindexed)
        else:
            manifest.append({"name": str(number), **stages})

    if manifest:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"pipelines": manifest}), encoding="utf-8", newline="\n")
            run = subprocess.run([checker, directory, "--graphics-manifest", path, "--jobs", "1"],
                                 stdout=subprocess.PIPE, encoding="utf-8", errors="replace")
        if run.returncode not in (0, 1):
            raise ValueError(f"checker exited with code {run.returncode}")
        results = checker_results(run.stdout, manifest)
        failed = any(r["status"] in ("module-failed", "pipeline-failed") for r in results.values())
        if run.returncode != int(failed):
            raise ValueError("checker exit status disagrees with its per-chain outcomes")
        for name, result in results.items():
            triples[int(name)].update(result)

    outcomes = {}
    for triple in triples:
        add_counts(outcomes, triple["status"], triple["occurrences"])
    return {"pass_rows": len(rows), "geometry_occurrences": sum(counts.values()),
            "distinct_triples": len(triples), "outcomes": outcomes}, triples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("passes", type=Path)
    parser.add_argument("shader_dir", type=Path)
    parser.add_argument("--checker", default="vk_pipeline_check")
    parser.add_argument("--report", type=Path, help="write per-triple results for private use")
    args = parser.parse_args()
    try:
        summary, triples = measure(read_json(args.passes), args.shader_dir, args.checker)
        if args.report:
            args.report.write_text(json.dumps({"summary": summary, "triples": triples}, indent=2) + "\n",
                                   encoding="utf-8", newline="\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
