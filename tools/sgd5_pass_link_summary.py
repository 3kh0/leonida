"""Join SGD5 --passes rows to a shader_link_check batch manifest and its --report."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shader_link_check import CHAINS, STAGES, is_shader_stem, load_pipeline_manifest, reject_duplicate_keys


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=reject_duplicate_keys)


def pass_chain(row):
    programs = row.get("programs") if isinstance(row, dict) else None
    stages = row.get("stages") if isinstance(row, dict) else None
    if not isinstance(programs, dict) or not isinstance(stages, dict) or programs.keys() != stages.keys():
        raise ValueError("expected matching programs and stages group maps")
    refs = {}
    for group, stem in programs.items():
        if not is_shader_stem(stem) or not isinstance(stages[group], str):
            raise ValueError("invalid program stem or stage")
        # SGD5 group 0 selects a mesh or vertex program. Other selectors have fixed roles.
        stage = {"0": "ms" if stages[group] == "ms" else "vs", "1": "ps",
                 "2": "gs", "5": "cs"}.get(group, "g" + group)
        refs[stage] = stem
    order = [*STAGES, *sorted(refs.keys() - set(STAGES))]
    return tuple((stage, refs[stage], "main") for stage in order if stage in refs)


def report_index(pipelines, report):
    if (not isinstance(report, dict) or type(report.get("version")) is not int or report["version"] != 1
            or not isinstance(report.get("pipelines"), list)):
        raise ValueError("expected a version 1 batch report")
    chains = {p.name: tuple((m.stage, m.path.name.removesuffix(".spv"), m.entry) for m in p.modules)
              for p in pipelines}
    results = dict.fromkeys(chains.values())
    if len(results) != len(chains):
        raise ValueError("ambiguous manifest: multiple pipeline names for one chain")
    for row in report["pipelines"]:
        name = row.get("name") if isinstance(row, dict) else None
        if not isinstance(name, str) or name not in chains or results[chains[name]] is not None:
            raise ValueError("unknown or duplicate report pipeline name")
        if row.get("status") not in ("compatible", "incompatible", "invalid"):
            raise ValueError("invalid batch status")
        problems = row.get("problems")
        if not isinstance(problems, list) or any(not isinstance(p, dict) or not isinstance(p.get("kind"), str)
                                                 for p in problems):
            raise ValueError("expected batch problems with a kind")
        results[chains[name]] = row
    return results


def summarize(rows, pipelines, report):
    results = report_index(pipelines, report)
    if not isinstance(rows, list):
        raise ValueError("expected a list of pass rows")
    counts = Counter()
    for index, row in enumerate(rows):
        try:
            counts[pass_chain(row)] += 1
        except ValueError as exc:
            raise ValueError(f"passes[{index}]: {exc}") from exc

    outcomes, diagnostics = {}, {}
    for chain, occurrences in counts.items():
        stages = tuple(stage for stage, _, _ in chain)
        result = results.get(chain)
        if not stages:
            status = "no-program"
        elif "cs" in stages:
            status = "compute"
        elif "ps" not in stages:
            status = "no-ps"
        elif stages not in CHAINS:
            status = "unsupported"
        elif chain not in results:
            status = "missing-manifest"
        elif result is None:
            status = "missing-result"
        else:
            status = result["status"]
            for problem in result["problems"]:
                add_counts(diagnostics, problem["kind"], occurrences)
        add_counts(outcomes, status, occurrences)
    return {"pass_rows": len(rows), "distinct_chains": len(counts),
            "outcomes": outcomes, "diagnostics": diagnostics}


def add_counts(table, key, occurrences):
    counts = table.setdefault(key, {"distinct": 0, "occurrences": 0})
    counts["distinct"] += 1
    counts["occurrences"] += occurrences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("passes", "manifest", "report"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args(argv)
    try:
        pipelines = load_pipeline_manifest(args.manifest, Path("."))
        summary = summarize(read_json(args.passes), pipelines, read_json(args.report))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    main()
