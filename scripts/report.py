#!/usr/bin/env python3
"""Writes the report of one job: one ctest test, one correctness batch or one
performance dataset, upstream against rvjit. run.sh calls it after each job;
it needs Python 3 only (no third-party package), so it runs on the board.

Usage: report.py --method M --id ID --name NAME [--results DIR] [--machine M]
                 [--base upstream] [--cand rvjit]

  --method   ctest, correctness, perf (performance-extended) or perfci
             (performance-onednn)
  --id       the logs' id (<id>.<driver>.<dt>.t<threads>), or the test name
             for ctest
  --name     the report's name, <task>.<batch>

Reads results/<machine>/<flavor>/<method>/ and writes
results/<machine>/reports/<name>.md (and <name>.csv, one row per problem, for
performance), then prints the markdown.

  ctest        each flavor's result, and whether they agree
  correctness  per log: benchdnn's counts per flavor; the implementations
               each flavor ran; the failed problems' repro lines
  performance  per problem, rvjit against upstream (speedup: GFLOPS ratio, or
               time ratio for perf-CI medians; > 1 is faster), summarized
               over all problems, per data type x weights tag, and per
               (upstream impl, rvjit impl); the slowest problems
"""

import argparse
import csv
import glob
import math
import os
import re
import socket
import statistics
import sys
from collections import Counter, defaultdict

COUNTS = re.compile(r"(\w+):(\d+)")
IMPL = re.compile(r"^\|\s*(\S+)\s*:\s*(\d+)\s*\(")
LOG = re.compile(r"^(?P<id>[^.]+)\.(?P<driver>[^.]+)\.(?P<dt>[^.]+)"
                 r"\.t(?P<threads>\d+)\.(?:csv|log)$")
MEAS = {"engine", "impl", "Gops", "max_ctime", "min_time", "min_Gflops",
        "avg_time", "avg_Gflops"}


def gmean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def fmt(x, spec=".3f"):
    return format(x, spec) if isinstance(x, float) and math.isfinite(x) else (
        "-" if isinstance(x, float) else str(x))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return out


def logs(root, flavor, method, id_):
    folder = os.path.join(root, flavor, method)
    found = {}
    for path in sorted(glob.glob(os.path.join(folder, id_ + ".*"))):
        m = LOG.match(os.path.basename(path))
        if m and m.group("id") == id_:
            found[(m.group("dt").replace("-", ":"), m.group("threads"))] = path
    return found


def prb_option(prb, name, default):
    m = re.search(rf"(?:^|\s)--{name}=(\S+)", prb)
    return m.group(1) if m else default


# ctest

def report_ctest(root, args):
    rows, ok = [], True
    results = {}
    for flavor in (args.base, args.cand):
        path = os.path.join(root, flavor, "ctest", args.id + ".result")
        line = open(path).read().split(None, 1) if os.path.isfile(path) else []
        results[flavor] = line[1].strip() if len(line) == 2 else "not run"
    same = results[args.base] == results[args.cand]
    ok = results[args.cand] == "Passed"
    rows.append([args.id, results[args.base], results[args.cand],
                 "same" if same else "DIFFERS"])
    md = table(["test", args.base, args.cand, "agreement"], rows)
    return md, ("PASS" if ok else "FAIL"), []


# correctness

def read_correctness(path):
    counts, impls, failed = {}, {}, []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("tests:"):
                counts = {k: int(v) for k, v in COUNTS.findall(line)}
            m = IMPL.match(line)
            if m:
                impls[m.group(1)] = impls.get(m.group(1), 0) + int(m.group(2))
            if "FAILED" in line and "__REPRO" in line:
                failed.append(line.split("__REPRO:", 1)[1].strip())
    return counts, impls, failed


def report_correctness(root, args):
    md, ok = [], True
    keys = ["tests", "passed", "skipped", "unimplemented", "failed"]
    found = {f: logs(root, f, "correctness", args.id) for f in (args.base, args.cand)}
    runs = sorted(set(found[args.base]) | set(found[args.cand]))
    if not runs:
        return ["no correctness log for " + args.id], "FAIL", []
    count_rows, impl_rows, failures = [], [], []
    for dt, thr in runs:
        impls = {}
        for flavor in (args.base, args.cand):
            path = found[flavor].get((dt, thr))
            counts, imp, failed = read_correctness(path) if path else ({}, {}, [])
            crashed = path is not None and not counts
            if flavor == args.cand and (crashed or counts.get("failed", 0)
                                        or path is None):
                ok = False
            count_rows.append([dt, f"t{thr}", flavor]
                              + [counts.get(k, "-") for k in keys]
                              + ["crashed" if crashed else ""])
            impls[flavor] = imp
            failures += [(flavor, dt, r) for r in failed]
        for impl in sorted(set(impls[args.base]) | set(impls[args.cand])):
            impl_rows.append([dt, impl, impls[args.base].get(impl, 0),
                              impls[args.cand].get(impl, 0)])
    md += ["### Problems", ""] + table(["dt", "threads", "flavor"] + keys + ["note"],
                                      count_rows)
    md += ["", "### Implementations (problems run)", ""] + table(
        ["dt", "implementation", args.base, args.cand], impl_rows)
    if failures:
        md += ["", f"### Failed problems ({len(failures)}, first 20)", ""]
        md += [f"- {fl} {dt}: `{r}`" for fl, dt, r in failures[:20]]
    return md, ("PASS" if ok else "FAIL"), []


# performance

def read_perf(path):
    """{problem key: (impl, gflops, time_ms, ctime_ms, dt, wtag)} of a
    performance.sh log (benchdnn's csv template)."""
    header, out = None, {}
    with open(path, newline="", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0] != "perf":
                continue
            if row[1] == "engine":
                header = row[1:]
                continue
            if not header or len(row) - 1 != len(header):
                continue
            r = dict(zip(header, row[1:]))
            if "warmup" in r.get("name", ""):
                continue
            key = tuple((k, v) for k, v in r.items() if k not in MEAS)
            try:
                out[key] = (r["impl"], float(r["avg_Gflops"]), float(r["avg_time"]),
                            float(r["max_ctime"]), r.get("wtag", ""), r.get("desc", ""))
            except (KeyError, ValueError):
                continue
    return out


def read_perfci(path):
    """{prb: (impl, gflops, time_ms, ctime_ms, wtag, desc)}, medians over the
    repetitions of a perfci.sh log."""
    samples = defaultdict(list)
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("--mode=P") or "warmup" in line:
                continue
            parts = line.rstrip("\n").rsplit(",", 4)
            if len(parts) != 5:
                continue
            try:
                samples[parts[0].strip()].append(
                    (parts[1].strip(), float(parts[2]), float(parts[3]), float(parts[4])))
            except ValueError:
                continue
    out = {}
    for prb, s in samples.items():
        impl = Counter(x[0] for x in s).most_common(1)[0][0]
        out[prb] = (impl, statistics.median(x[1] for x in s),
                    statistics.median(x[2] for x in s),
                    statistics.median(x[3] for x in s),
                    prb_option(prb, "wtag", "any"), prb.split()[-1])
    return out


def summary_row(label, ps):
    sp = [p["speedup"] for p in ps]
    fin = [x for x in sp if math.isfinite(x)]
    return [*label, len(ps), gmean(sp), statistics.median(fin) if fin else float("nan"),
            min(fin, default=float("nan")), max(fin, default=float("nan")),
            sum(x > 1 for x in fin), gmean(p["ctime_ratio"] for p in ps)]


def report_performance(root, args):
    read = read_perfci if args.method == "perfci" else read_perf
    found = {f: logs(root, f, args.method, args.id) for f in (args.base, args.cand)}
    problems = []
    for run in sorted(set(found[args.base]) & set(found[args.cand])):
        dt, thr = run
        base, cand = read(found[args.base][run]), read(found[args.cand][run])
        for key in base.keys() & cand.keys():
            b, c = base[key], cand[key]
            pdt = prb_option(key, "dt", "f32") if args.method == "perfci" else (
                dict(key).get("sdt", dt) if dt == "harness" else dt)
            if args.method == "perfci":
                speedup = b[2] / c[2] if c[2] > 0 else float("nan")
            else:
                speedup = c[1] / b[1] if b[1] > 0 else float("nan")
            problems.append({
                "dt": pdt, "wtag": c[4], "threads": thr, "desc": c[5],
                "base_impl": b[0], "cand_impl": c[0],
                "base_gflops": b[1], "cand_gflops": c[1],
                "base_time_ms": b[2], "cand_time_ms": c[2],
                "base_ctime_ms": b[3], "cand_ctime_ms": c[3],
                "speedup": speedup,
                "ctime_ratio": c[3] / b[3] if b[3] > 0 else float("nan")})
    # Problems a build skipped (e.g. a data type the CPU lacks) have no
    # implementation or time: they are counted apart, not in the summaries
    measured = [p for p in problems
                if p["base_impl"] and p["cand_impl"] and math.isfinite(p["speedup"])]
    skipped = len(problems) - len(measured)
    if not measured:
        return [f"no problem of {args.id} ran on both flavors"
                + (f" ({skipped} skipped)" if skipped else "")], "FAIL", problems
    problems = measured
    head = ["problems", "speedup geomean", "median", "min", "max", "faster",
            "ctime ratio geomean"]
    md = ["### All problems", ""] + table(head, [summary_row([], problems)])
    by = defaultdict(list)
    for p in problems:
        by[(p["dt"], p["wtag"], "t" + p["threads"])].append(p)
    md += ["", "### Per data type and weights tag", ""] + table(
        ["dt", "wtag", "threads"] + head,
        [summary_row(k, v) for k, v in sorted(by.items())])
    by = defaultdict(list)
    for p in problems:
        by[(p["base_impl"], p["cand_impl"])].append(p)
    md += ["", "### Per implementation", ""] + table(
        [f"{args.base} impl", f"{args.cand} impl"] + head,
        [summary_row(k, v) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))])
    slow = sorted((p for p in problems if math.isfinite(p["speedup"])),
                  key=lambda p: p["speedup"])[:10]
    md += ["", "### Slowest problems", ""] + table(
        ["problem", "dt", "wtag", f"{args.cand} impl", "speedup", "ctime ratio"],
        [[f"`{p['desc']}`", p["dt"], p["wtag"], p["cand_impl"], p["speedup"],
          p["ctime_ratio"]] for p in slow])
    kind = "time ratio of perf-CI medians" if args.method == "perfci" else "GFLOPS ratio"
    md = [f"Speedup is {args.cand} over {args.base} ({kind}; > 1 is faster); "
          f"the creation-time ratio is {args.cand}'s over {args.base}'s (< 1 is "
          "faster).", ""] + (
        [f"{skipped} problems were not measured on both builds (skipped) and "
         "are left out.", ""] if skipped else []) + md
    return md, f"{len(problems)} problems, geomean speedup {gmean(p['speedup'] for p in problems):.3f}", problems


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--method", required=True,
                        choices=["ctest", "correctness", "perf", "perfci"])
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--results", default=os.environ.get(
        "RESULTS_DIR", os.path.join(here, "..", "results")))
    parser.add_argument("--machine", default=os.environ.get(
        "SLURM_JOB_PARTITION", socket.gethostname()))
    parser.add_argument("--base", default="upstream")
    parser.add_argument("--cand", default="rvjit")
    args = parser.parse_args()

    root = os.path.join(args.results, args.machine)
    if args.method == "ctest":
        body, verdict, problems = report_ctest(root, args)
    elif args.method == "correctness":
        body, verdict, problems = report_correctness(root, args)
    else:
        body, verdict, problems = report_performance(root, args)
    md = [f"## {args.name}: {verdict}", "",
          f"{args.cand} against {args.base} on {args.machine}; logs "
          f"`{args.method}/{args.id}.*`.", ""] + body
    out = os.path.join(root, "reports")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, args.name + ".md"), "w") as f:
        f.write("\n".join(md) + "\n")
    if problems:
        with open(os.path.join(out, args.name + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, list(problems[0]))
            w.writeheader()
            w.writerows(problems)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
