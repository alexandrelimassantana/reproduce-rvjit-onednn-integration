#!/usr/bin/env python3
"""Compare the flavors problem by problem, and summarize per implementation.

Usage: compare.py [csv-dir] [--base upstream] [--cand rvjit] [--figures DIR]

Reads <csv-dir>/perf.csv and perfci.csv (collect.py) and pairs each problem
the baseline and the candidate both ran (same machine, experiment, data
type, thread count and problem). A perf-CI problem counts with the median of
its repetitions; its data type and weights tag come from its repro line. A
comparison is one experiment x data type x weights tag x thread count, e.g.
`extended-conv-ipdps26 f16:f16:f32 wtag=any t8`. A dataset is one
performance batch (its id, e.g. `onednn-ip-models`); a primitive is a
benchdnn driver (matmul, ip, conv), whose ensemble is every dataset of it.
Writes:

  compare.csv                     one row per problem: both implementations,
                                  GFLOPS, times and creation times; speedup
                                  (candidate GFLOPS over baseline's) and
                                  ctime_ratio (candidate's creation time over
                                  baseline's; < 1 is faster)
  summary_by_impl.csv             one row per comparison x (baseline impl,
                                  candidate impl): problem count, geomean,
                                  median, min and max speedup, problems faster
                                  than the baseline, geomean ctime ratio,
                                  geomean GFLOPS of each flavor
  summary_by_dataset.csv          per dataset: one row over all its problems
                                  (impls `all`), then one per (baseline impl,
                                  candidate impl)
  summary_by_primitive.csv        per primitive, the ensemble of its datasets:
                                  all problems, per data type, and per
                                  candidate impl (all data types and each),
                                  the impl rows over same-impl problems
  report.md                       per machine: the ctest and correctness
                                  verdicts (from ctest.csv, correctness.csv
                                  and dispatch.csv, when present), then the
                                  primitive and dataset summaries as
                                  markdown tables

  duplicates.csv                  every case measured more than once: in the
                                  same dataset under another name, or in
                                  several datasets

Problems that ran on different implementations (`impl differs`) keep their
own (baseline impl, candidate impl) rows and are left out of the overall
summary, so an implementation's numbers never mix in another's.

A case is a unique problem: driver, data type, thread count, direction,
tags, bias, algorithm, attributes and shape, ignoring its name and how the
batch spelled defaults, run on the same pair of implementations (a batch
that forces --impl measures another case). Each summary counts a case once: the comparison and
dataset summaries once per dataset, the primitive summary once per
primitive (its first occurrence, in dataset order). compare.csv keeps every
row and marks the repeats (dup_in_dataset, dup_in_primitive).
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

RUN = ["machine", "host", "flavor", "revision", "cpu_isa", "cpu_model"]
EXP = ["machine", "id", "driver", "dt", "threads"]
MEAS = ["Gops", "max_ctime", "min_time", "min_Gflops", "avg_time", "avg_Gflops"]
COMPARISON = ["machine", "driver", "id", "dt", "wtag", "threads"]


def gmean(x):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def pair(perf, base, cand):
    bench = [c for c in perf.columns
             if c not in RUN + EXP + MEAS + ["engine", "impl"]]
    key = EXP + bench
    perf = perf.copy()
    perf[bench] = perf[bench].fillna("")
    perf = perf[pd.to_numeric(perf["avg_Gflops"], errors="coerce") > 0]
    sides = []
    for flavor in (base, cand):
        side = perf[perf["flavor"] == flavor].drop_duplicates(key, keep="last")
        side = side[key + ["impl", "avg_Gflops", "avg_time", "max_ctime"]]
        sides.append(side.rename(columns={
            "impl": f"{flavor}_impl", "avg_Gflops": f"{flavor}_gflops",
            "avg_time": f"{flavor}_time_ms", "max_ctime": f"{flavor}_ctime_ms"}))
    both = sides[0].merge(sides[1], on=key)
    for c in both.columns:
        if c.endswith(("_gflops", "_time_ms", "_ctime_ms")):
            both[c] = pd.to_numeric(both[c], errors="coerce")
    both["speedup"] = both[f"{cand}_gflops"] / both[f"{base}_gflops"]
    both["ctime_ratio"] = both[f"{cand}_ctime_ms"] / both[f"{base}_ctime_ms"]
    both["same_impl"] = both[f"{base}_impl"] == both[f"{cand}_impl"]
    both["note"] = np.where(both["same_impl"], "", "impl differs")
    if "wtag" not in both:
        both["wtag"] = ""
    both["comparison"] = (both["id"] + " " + both["dt"] + " wtag="
                          + both["wtag"].astype(str) + " t"
                          + both["threads"].astype(str))
    return both


def prb_option(prb, name, default):
    m = re.search(rf"(?:^|\s)--{name}=(\S+)", prb)
    return m.group(1) if m else default


def pair_perfci(ci, base, cand):
    """The perf-CI samples as pair() rows: per problem and flavor, the median
    of the repetitions and the implementation that ran most often."""
    ci = ci.copy()
    for c in ("gflops", "time_ms", "ctime_ms"):
        ci[c] = pd.to_numeric(ci[c], errors="coerce")
    # A problem the build skipped (e.g. a data type the CPU lacks) has no
    # implementation or time
    ci = ci[ci["impl"].notna() & (ci["impl"] != "") & (ci["time_ms"] > 0)]
    ci["dt"] = ci["prb"].apply(prb_option, args=("dt", "f32"))
    ci["wtag"] = ci["prb"].apply(prb_option, args=("wtag", "any"))
    ci["desc"] = ci["prb"].str.split().str[-1]
    key = EXP + ["wtag", "desc", "prb"]
    sides = []
    for flavor in (base, cand):
        g = ci[ci["flavor"] == flavor].groupby(key, sort=False)
        side = g.agg(impl=("impl", lambda x: x.mode().iat[0]),
                     gflops=("gflops", "median"), time_ms=("time_ms", "median"),
                     ctime_ms=("ctime_ms", "median")).reset_index()
        sides.append(side.rename(columns={
            "impl": f"{flavor}_impl", "gflops": f"{flavor}_gflops",
            "time_ms": f"{flavor}_time_ms", "ctime_ms": f"{flavor}_ctime_ms"}))
    both = sides[0].merge(sides[1], on=key)
    both["speedup"] = both[f"{base}_time_ms"] / both[f"{cand}_time_ms"]
    both["ctime_ratio"] = both[f"{cand}_ctime_ms"] / both[f"{base}_ctime_ms"]
    both["same_impl"] = both[f"{base}_impl"] == both[f"{cand}_impl"]
    both["note"] = np.where(both["same_impl"], "", "impl differs")
    both["comparison"] = (both["id"] + " " + both["dt"] + " wtag="
                          + both["wtag"] + " t" + both["threads"].astype(str))
    return both


def summarize(df, by, base, cand):
    rows = []
    for k, g in df.groupby(by, dropna=False, sort=False):
        rows.append({**dict(zip(by, k if isinstance(k, tuple) else (k,))),
                     "problems": len(g),
                     "speedup_geomean": gmean(g["speedup"]),
                     "speedup_median": g["speedup"].median(),
                     "speedup_min": g["speedup"].min(),
                     "speedup_max": g["speedup"].max(),
                     "faster": int((g["speedup"] > 1).sum()),
                     "ctime_ratio_geomean": gmean(g["ctime_ratio"]),
                     f"{base}_gflops_geomean": gmean(g[f"{base}_gflops"]),
                     f"{cand}_gflops_geomean": gmean(g[f"{cand}_gflops"])})
    return pd.DataFrame(rows)


def md_table(df, columns):
    def cell(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    return (["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
            + ["| " + " | ".join(cell(v) for v in row) + " |"
               for row in df[columns].itertuples(index=False)])


# Defaults benchdnn assumes when a batch leaves an option out, per driver
DEFAULTS = {"conv": {"dir": "FWD_B", "bia-dt": "f32", "alg": "direct"},
            "ip": {"dir": "FWD_B", "bia-dt": "f32"},
            "matmul": {"bia-dt": "undef"}}
TAGS = {"acb": "axb", "acdb": "axb", "acdeb": "axb",
        "ab": "abx", "abc": "abx", "abcd": "abx", "": "any"}
IGNORED = {"mode", "max-ms-per-prb", "dt", "wtag", "impl", "mb", "sdt"}


def strip_name(desc):
    return re.split(r"_?n[\"a-zA-Z]", str(desc), maxsplit=1)[0]


def case_key(row):
    """A problem's identity, the same for perf and perf-CI rows: options
    with their defaults filled in and tags in one spelling, and the shape
    without its name."""
    opts = dict(DEFAULTS.get(row["driver"], {}))
    if row["source"] == "perfci":
        for name, value in re.findall(r"--([a-z-]+)=(\S*)", str(row["prb"])):
            opts[name] = value
    else:
        for col, name in (("dir", "dir"), ("bia_dt", "bia-dt"), ("stag", "stag"),
                          ("dtag", "dtag"), ("alg", "alg"), ("attr", "attr")):
            value = row.get(col, "")
            if isinstance(value, str) and value:
                opts[name] = value
    for name in IGNORED:
        opts.pop(name, None)
    for name in ("stag", "dtag"):
        opts[name] = TAGS.get(opts.get(name, ""), opts.get(name, ""))
    wtag = TAGS.get(str(row["wtag"]) if isinstance(row["wtag"], str) else "",
                    row["wtag"])
    attrs = " ".join(f"--{k}={v}" for k, v in sorted(opts.items()) if v != "")
    impls = " ".join(str(v) for k, v in row.items() if k.endswith("_impl"))
    return (f"{row['driver']} --dt={row['dt']} --wtag={wtag} t{row['threads']} "
            f"{attrs} {strip_name(row['desc'])} on {impls}")


def mark_duplicates(both):
    both["case"] = both.apply(case_key, axis=1)
    both["dup_in_dataset"] = both.duplicated(["machine", "id", "case"])
    both["dup_in_primitive"] = both.duplicated(["machine", "driver", "case"])
    return both


def duplicates_table(both):
    """One row per occurrence of a case measured more than once."""
    rep = both[both.duplicated(["machine", "driver", "case"], keep=False)]
    cols = ["machine", "driver", "case", "id", "desc", "speedup", "dup_in_dataset",
            "dup_in_primitive"]
    return rep.sort_values(["machine", "driver", "case"], kind="stable")[cols]


def cases_table(both):
    rows = []
    for (machine, driver), g in both.groupby(["machine", "driver"], sort=True):
        rows.append({"machine": machine, "driver": driver, "measured": len(g),
                     "unique_cases": int((~g["dup_in_primitive"]).sum()),
                     "repeated_in_dataset": int(g["dup_in_dataset"].sum()),
                     "repeated_across_datasets": int(
                         (g["dup_in_primitive"] & ~g["dup_in_dataset"]).sum())})
    return pd.DataFrame(rows)


def cases_section(machine, cases, dups):
    c = cases[cases["machine"] == machine]
    if c.empty:
        return []
    cols = ["driver", "measured", "unique_cases", "repeated_in_dataset",
            "repeated_across_datasets"]
    md = ["", "### Cases", "",
          f"{int(c['unique_cases'].sum())} unique cases evaluated "
          f"({int(c['measured'].sum())} measurements). A case is counted once: "
          "the performance tables and figures leave out its repeats, within a "
          "dataset (the same problem under another name) and across datasets "
          "(the primitive tables keep its first dataset).", ""]
    md += md_table(c, cols)
    d = dups[dups["machine"] == machine]
    if not d.empty:
        rows = []
        for (driver, case), g in d.groupby(["driver", "case"], sort=False):
            rows.append([driver, f"`{case}`", len(g),
                         "<br>".join(f"{i}: `{desc}`" for i, desc
                                     in zip(g["id"], g["desc"])),
                         " / ".join(f"{v:.2f}" for v in g["speedup"])])
        md += ["", "<details><summary>Repeated cases "
               f"({len(rows)}, with each measurement's speedup)</summary>", ""]
        md += md_table(pd.DataFrame(rows, columns=["driver", "case", "times",
                                                   "occurrences", "speedups"]),
                       ["driver", "case", "times", "occurrences", "speedups"])
        md += ["", "</details>"]
    return md


def job_name(dataset):
    """onednn-conv-test_ci -> correctness-onednn.conv.test_ci"""
    suite, driver, batch = (dataset.split("-", 2) + ["", ""])[:3]
    return f"correctness-{suite}.{driver}.{batch}"


def read_csv(csv_dir, name):
    path = os.path.join(csv_dir, name)
    df = pd.read_csv(path, dtype=str).fillna("") if os.path.isfile(path) else None
    return df if df is not None and not df.empty else None


def correctness_section(machine, ctest, correctness, dispatch, base, cand):
    """The verdicts of report.py, one row per ctest test and correctness run:
    PASS when the candidate passed (ctest) or finished without failures."""
    md = []
    if ctest is not None and machine in set(ctest["machine"]):
        c = ctest[ctest["machine"] == machine]
        res = {(r.flavor, r.test): r.result for r in c.itertuples()}
        rows = []
        for test in sorted(set(c["test"])):
            b = res.get((base, test), "not run")
            k = res.get((cand, test), "not run")
            rows.append([test, b, k, "same" if b == k else "DIFFERS",
                         "PASS" if k == "Passed" else "FAIL"])
        n = sum(r[-1] == "PASS" for r in rows)
        md += ["", "### ctest", "", f"{n} of {len(rows)} tests pass on {cand}.", ""]
        md += md_table(pd.DataFrame(rows, columns=["test", base, cand,
                                                   "agreement", "verdict"]),
                       ["test", base, cand, "agreement", "verdict"])
    if correctness is not None and machine in set(correctness["machine"]):
        c = correctness[correctness["machine"] == machine]
        key = ["id", "dt", "threads"]
        counts = {(r["flavor"],) + tuple(r[k] for k in key): r
                  for _, r in c.iterrows()}
        hits = {}
        if dispatch is not None:
            for _, r in dispatch[dispatch["machine"] == machine].iterrows():
                hits.setdefault(tuple(r[k] for k in key), {}).setdefault(
                    r["impl"], {})[r["flavor"]] = r["hits"]

        def cell(r):
            if r is None:
                return "not run"
            if r.get("crashed") or not r.get("tests"):
                return "no summary (running, killed or crashed)"
            out = [f"{r['passed']} passed", f"{r['skipped']} skipped"]
            out += [f"{r[k]} {k.replace('_', ' ')}"
                    for k in ("mistrusted", "unimplemented", "invalid_arguments",
                              "failed", "listed")
                    if r.get(k) not in ("", "0", None)]
            return ", ".join(out)

        rows = []
        for run in sorted({tuple(r[k] for k in key) for _, r in c.iterrows()}):
            b, k = counts.get((base,) + run), counts.get((cand,) + run)
            tests = next((r["tests"] for r in (b, k)
                          if r is not None and r.get("tests")), "-")
            ok = (k is not None and k.get("tests") and not k.get("crashed")
                  and k.get("failed", "0") in ("", "0"))
            changed = [i for i, f in hits.get(run, {}).items()
                       if f.get(base, "0") != f.get(cand, "0")]
            rows.append([job_name(run[0]), run[1], f"t{run[2]}", tests,
                         cell(b), cell(k),
                         f"{len(changed)} impls differ" if changed else "same",
                         "PASS" if ok else "FAIL"])
        columns = ["job", "dt", "threads", "tests", base, cand, "dispatch",
                   "verdict"]
        n = sum(r[-1] == "PASS" for r in rows)
        md += ["", "### Correctness", "",
               f"{n} of {len(rows)} runs pass on {cand}. `dispatch` compares "
               "the problems each implementation ran (details in the job's "
               "report).", ""]
        md += md_table(pd.DataFrame(rows, columns=columns), columns)
    return md


def write_report(path, datasets, primitives, base, cand, checks=(None,) * 3,
                 cases=None, dups=None, figures=None):
    """Per machine: the ctest and correctness verdicts, then each primitive's
    ensemble and its datasets."""
    bi, ci = f"{base}_impl", f"{cand}_impl"
    stats = ["problems", "speedup_geomean", "speedup_median", "speedup_min",
             "speedup_max", "faster", "ctime_ratio_geomean"]
    md = [f"# {cand} against {base}", "",
          f"Speedup is {cand} over {base} (GFLOPS, or time for perf-CI medians; "
          f"> 1 is faster). The creation-time ratio is {cand}'s over {base}'s "
          "(< 1 is faster). Rows with `all` cover every problem; the others, "
          "one pair of implementations."]
    machines = set(datasets["machine"]) if datasets is not None else set()
    for df in checks:
        if df is not None:
            machines |= set(df["machine"])
    for machine in sorted(machines):
        md += ["", f"## {machine}"]
        md += correctness_section(machine, *checks, base, cand)
        if datasets is None:
            continue
        md += cases_section(machine, cases, dups)
        d = datasets[datasets["machine"] == machine]
        p = primitives[primitives["machine"] == machine]
        if not d.empty:
            md += ["", "### Performance"]
        for driver in dict.fromkeys(d["driver"]):
            md += ["", f"#### {driver}", ""]
            if figures:
                img = os.path.join(figures, re.sub(r"[^A-Za-z0-9.=-]+", "_", machine),
                                   f"report.{driver}.png")
                md += [f"![{driver} on {machine}]({img})", "",
                       "Per data type: the speedup of every unique case per "
                       "dataset; its heatmap over the problem's dimensions; the "
                       "heatmap of the cases slower than the baseline. Crosses "
                       "mark cases that ran on different implementations.", ""]
            md += ["Ensemble of its datasets:", ""]
            md += md_table(p[p["driver"] == driver], [ci, "dt"] + stats)
            for dataset in dict.fromkeys(d[d["driver"] == driver]["id"]):
                md += ["", f"##### {dataset}", ""]
                md += md_table(d[(d["driver"] == driver) & (d["id"] == dataset)],
                               [bi, ci] + stats)
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("csv_dir", nargs="?", default=os.path.join(here, "..", "csv"))
    parser.add_argument("--base", default="upstream")
    parser.add_argument("--cand", default="rvjit")
    parser.add_argument("--figures", default=None,
                        help="plots.py's output directory, linked from report.md "
                             "(default: figures/ next to csv-dir)")
    args = parser.parse_args()
    figures = args.figures or os.path.join(os.path.dirname(
        os.path.abspath(args.csv_dir)), "figures")
    figures = os.path.relpath(figures, os.path.abspath(args.csv_dir))

    parts = []
    for name, pairing in (("perf.csv", pair), ("perfci.csv", pair_perfci)):
        path = os.path.join(args.csv_dir, name)
        if os.path.isfile(path):
            df = pd.read_csv(path, dtype=str)
            if df.empty:
                continue
            parts.append(pairing(df, args.base, args.cand)
                         .assign(source=name[:-4]))
    both = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    checks = tuple(read_csv(args.csv_dir, name)
                   for name in ("ctest.csv", "correctness.csv", "dispatch.csv"))
    report = os.path.join(args.csv_dir, "report.md")
    if both.empty:
        if all(c is None for c in checks):
            sys.exit(f"no problem ran on both {args.base} and {args.cand}")
        write_report(report, None, None, args.base, args.cand, checks)
        print(f"no performance problem to compare; report: {report}",
              file=sys.stderr)
        return
    base_impl, cand_impl = f"{args.base}_impl", f"{args.cand}_impl"

    both = mark_duplicates(both)
    both.to_csv(os.path.join(args.csv_dir, "compare.csv"), index=False)
    dups = duplicates_table(both)
    dups.to_csv(os.path.join(args.csv_dir, "duplicates.csv"), index=False)
    cases = cases_table(both)
    every = both
    both = every[~every["dup_in_dataset"]]
    by_impl = summarize(both, ["comparison"] + COMPARISON
                        + [base_impl, cand_impl, "same_impl"], args.base, args.cand)
    by_impl.to_csv(os.path.join(args.csv_dir, "summary_by_impl.csv"), index=False)
    unique = every[~every["dup_in_primitive"]]
    same = unique[unique["same_impl"]]
    impls = [base_impl, cand_impl]
    dataset_by = ["machine", "driver", "id"] + impls
    datasets = pd.concat([
        summarize(both.assign(**{base_impl: "all", cand_impl: "all"}), dataset_by,
                  args.base, args.cand),
        summarize(both, dataset_by, args.base, args.cand)])
    datasets = datasets.sort_values(["machine", "driver", "id"], kind="stable")
    datasets.to_csv(os.path.join(args.csv_dir, "summary_by_dataset.csv"), index=False)
    prim_by = ["machine", "driver", cand_impl, "dt"]
    overall = pd.concat([
        summarize(unique.assign(**{cand_impl: "all", "dt": "all"}), prim_by,
                  args.base, args.cand),
        summarize(unique.assign(**{cand_impl: "all"}), prim_by, args.base, args.cand),
        summarize(same.assign(dt="all"), prim_by, args.base, args.cand),
        summarize(same, prim_by, args.base, args.cand)])
    overall = overall.sort_values(["machine", "driver"], kind="stable")
    overall.to_csv(os.path.join(args.csv_dir, "summary_by_primitive.csv"),
                   index=False)
    write_report(report, datasets, overall, args.base, args.cand, checks,
                 cases, dups, figures)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    cols = ["comparison", base_impl, cand_impl, "problems", "speedup_geomean",
            "speedup_min", "speedup_max", "faster", "ctime_ratio_geomean"]
    print(by_impl[cols].to_string(index=False, float_format="%.3f"))
    print()
    print(overall.drop(columns=["machine"]).to_string(index=False, float_format="%.3f"))
    print(f"\nreport: {report}", file=sys.stderr)
    print(f"\n{len(every)} problems compared, {len(unique)} unique cases; "
          f"tables in {args.csv_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
