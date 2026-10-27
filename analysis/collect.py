#!/usr/bin/env python3
"""Merge the logs under a results folder into CSVs.

Usage: collect.py [results] [-o DIR]

Walks <results>/<machine>/<flavor>/<kind>/ for the logs the scripts write,
named <id>.<driver>.<dt>.t<threads>.{csv,log}, and writes to DIR (default
csv/):

  perf.csv         one row per problem of performance.sh: the run (machine,
                   host, flavor, revision, CPU), the experiment (id, driver,
                   dt, threads), then benchdnn's columns (impl, name, desc,
                   Gops, max_ctime, min_time, avg_Gflops, ...)
  perfci.csv       one row per sample of perfci.sh: the run, the experiment,
                   prb (the problem's repro line), impl, gflops, time_ms,
                   ctime_ms
  correctness.csv  one row per correctness log: benchdnn's counts (tests,
                   passed, skipped, unimplemented, failed, ...)
  dispatch.csv     one row per implementation of each correctness log: impl,
                   hits (benchdnn's implementation statistics)
  ctest.csv        one row per test of ctest.sh (its <test>.result): result
  timings.csv      the timings.csv of every machine, with its machine

Warm-up problems (named warmup:*) are dropped.
"""

import argparse
import csv
import os
import re
import sys

RUN = ["machine", "host", "flavor", "revision", "cpu_isa", "cpu_model"]
EXP = ["id", "driver", "dt", "threads"]
LOG = re.compile(r"^(?P<id>[^.]+)\.(?P<driver>[^.]+)\.(?P<dt>[^.]+)"
                 r"\.t(?P<threads>\d+)\.(?:csv|log)$")
COUNTS = re.compile(r"(\w+):(\d+)")
IMPL = re.compile(r"^\|\s*(\S+)\s*:\s*(\d+)\s*\(")


def read_env(folder):
    env = {}
    path = os.path.join(folder, "run.env")
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                key, _, value = line.rstrip("\n").partition("=")
                env[key] = value
    return env


def experiment(name):
    m = LOG.match(name)
    if not m:
        return None
    exp = m.groupdict()
    exp["dt"] = exp["dt"].replace("-", ":")
    return exp


def is_warmup(text):
    return "warmup" in text


def read_perf(path):
    header, rows = None, []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0] != "perf":
                continue
            if row[1] == "engine":
                header = row[1:]
            elif header and len(row) - 1 == len(header):
                rec = dict(zip(header, row[1:]))
                if not is_warmup(rec.get("name", "")):
                    rows.append(rec)
    return rows


def read_perfci(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("--mode=P") or is_warmup(line):
                continue
            parts = line.rstrip("\n").rsplit(",", 4)
            if len(parts) != 5:
                continue
            prb, impl, gflops, time, ctime = parts
            try:
                rows.append({"prb": prb.strip(), "impl": impl.strip(),
                             "gflops": float(gflops), "time_ms": float(time),
                             "ctime_ms": float(ctime)})
            except ValueError:
                continue
    return rows


def read_correctness(path):
    counts, impls = {}, []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("tests:"):
                counts = {k: int(v) for k, v in COUNTS.findall(line)}
            m = IMPL.match(line)
            if m:
                impls.append({"impl": m.group(1), "hits": int(m.group(2))})
    if not counts:
        counts = {"crashed": 1}
    return counts, impls


def read_ctest(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) == 2:
                rows.append({"test": parts[0], "result": parts[1].strip()})
    return rows


def write(path, rows, columns):
    columns += [c for r in rows for c in r if c not in columns]
    seen = []
    for c in columns:
        if c not in seen:
            seen.append(c)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, seen, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows):6d} rows in {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("results", nargs="?",
                        default=os.path.join(here, "..", "results"))
    parser.add_argument("-o", "--output", default=os.path.join(here, "..", "csv"))
    args = parser.parse_args()

    out = {k: [] for k in ["perf", "perfci", "correctness", "dispatch",
                           "ctest", "timings"]}
    for machine in sorted(os.listdir(args.results)):
        mdir = os.path.join(args.results, machine)
        if machine == "slurm" or not os.path.isdir(mdir):
            continue
        tpath = os.path.join(mdir, "timings.csv")
        if os.path.isfile(tpath):
            with open(tpath, newline="") as f:
                out["timings"] += [{"machine": machine, **r}
                                   for r in csv.DictReader(f)]
        for flavor in sorted(os.listdir(mdir)):
            fdir = os.path.join(mdir, flavor)
            if not os.path.isdir(fdir):
                continue
            for kind in sorted(os.listdir(fdir)):
                kdir = os.path.join(fdir, kind)
                if not os.path.isdir(kdir):
                    continue
                env = read_env(kdir)
                run = {k: env.get(k, "") for k in RUN}
                run["machine"], run["flavor"] = machine, flavor
                if kind == "ctest":
                    for name in sorted(os.listdir(kdir)):
                        if name.endswith(".result"):
                            out["ctest"] += [{**run, **r} for r in
                                             read_ctest(os.path.join(kdir, name))]
                    continue
                for name in sorted(os.listdir(kdir)):
                    exp = experiment(name)
                    if exp is None:
                        continue
                    path = os.path.join(kdir, name)
                    meta = {**run, **exp}
                    if kind == "perf":
                        out["perf"] += [{**meta, **r} for r in read_perf(path)]
                    elif kind == "perfci":
                        out["perfci"] += [{**meta, **r} for r in read_perfci(path)]
                    elif kind == "correctness":
                        counts, impls = read_correctness(path)
                        out["correctness"].append({**meta, **counts})
                        out["dispatch"] += [{**meta, **i} for i in impls]

    if not any(out.values()):
        sys.exit(f"no logs under {args.results}")
    os.makedirs(args.output, exist_ok=True)
    for kind, rows in out.items():
        first = (["machine"] if kind == "timings" else RUN
                 + ([] if kind == "ctest" else EXP))
        write(os.path.join(args.output, f"{kind}.csv"), rows, list(first))


if __name__ == "__main__":
    main()
