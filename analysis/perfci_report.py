#!/usr/bin/env python3
"""Apply oneDNN's performance-CI test to the perfci.sh samples.

Usage: perfci_report.py [csv-dir] [--base upstream] [--cand rvjit]

Reads <csv-dir>/perfci.csv (collect.py) and, for each problem of each
experiment, compares the candidate's samples with the baseline's as
.github/automation/performance/benchdnn_comparison.py does:

  exec regressed  t-test (candidate slower) p <= 0.05, and the median or the
                  minimum time >= 10% slower
  exec improved   t-test (candidate faster) p <= 0.05, and the median or the
                  minimum time >= 1.1x faster
  ctime regressed t-test p <= 0.05, and the median or the minimum creation
                  time >= 10% slower

(with fewer than 3 samples per side the t-test is skipped, as in the CI).
Writes perfci_report.csv (one row per problem: medians, speedup, p-values,
verdicts) and perfci_report.md (per experiment: counts, geomean speedup, and
the CI's table of the problems that changed).
"""

import argparse
import os
import statistics
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

KEY = ["machine", "id", "driver", "dt", "threads", "prb"]


def gmean(x):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def verdict(e1, e2, c1, c2):
    use_ttest = len(e1) >= 3 and len(e2) >= 3
    m1, m2 = statistics.median(e1), statistics.median(e2)
    cm1, cm2 = statistics.median(c1), statistics.median(c2)
    if 0 in (m1, m2, min(e1), min(e2), cm1, min(c1)):
        return None
    p_reg = ttest_ind(e2, e1, alternative="greater").pvalue if use_ttest else 0.0
    p_imp = ttest_ind(e2, e1, alternative="less").pvalue if use_ttest else 0.0
    p_ct = ttest_ind(c2, c1, alternative="greater").pvalue if use_ttest else 0.0
    return {
        "base_median_ms": m1, "cand_median_ms": m2,
        "base_min_ms": min(e1), "cand_min_ms": min(e2),
        "speedup": m1 / m2,
        "base_ctime_median_ms": cm1, "cand_ctime_median_ms": cm2,
        "ctime_ratio": cm2 / cm1,
        "samples": f"{len(e1)}/{len(e2)}",
        "p_regressed": p_reg, "p_improved": p_imp, "p_ctime": p_ct,
        "exec_regressed": p_reg <= 0.05 and ((m2 - m1) / m1 >= 0.1
                                            or (min(e2) - min(e1)) / min(e1) >= 0.1),
        "exec_improved": p_imp <= 0.05 and (m1 / m2 >= 1.1 or min(e1) / min(e2) >= 1.1),
        "ctime_regressed": p_ct <= 0.05 and ((cm2 - cm1) / cm1 >= 0.1
                                             or (min(c2) - min(c1)) / min(c1) >= 0.1),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("csv_dir", nargs="?", default=os.path.join(here, "..", "csv"))
    parser.add_argument("--base", default="upstream")
    parser.add_argument("--cand", default="rvjit")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.csv_dir, "perfci.csv"),
                     dtype={"threads": str, "dt": str})
    rows = []
    for key, g in df.groupby(KEY, sort=False):
        b = g[g["flavor"] == args.base]
        c = g[g["flavor"] == args.cand]
        if b.empty or c.empty:
            continue
        v = verdict(list(b["time_ms"]), list(c["time_ms"]),
                    list(b["ctime_ms"]), list(c["ctime_ms"]))
        if v:
            rows.append({**dict(zip(KEY, key)), **v})
    if not rows:
        sys.exit(f"no problem has samples of both {args.base} and {args.cand}")
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(args.csv_dir, "perfci_report.csv"), index=False)

    md = [f"# Performance CI: {args.cand} against {args.base}", "",
          "Speedup is the baseline's median time over the candidate's (> 1 is "
          "faster); ctime ratio is the candidate's median creation time over the "
          "baseline's (> 1 is slower). Verdicts follow oneDNN's "
          "`benchdnn_comparison.py`.", "",
          "| machine | experiment | dt | threads | problems | improved | regressed "
          "| ctime regressed | geomean speedup | geomean ctime ratio |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    by = ["machine", "id", "dt", "threads"]
    for k, g in rep.groupby(by, sort=False):
        md.append("| " + " | ".join(map(str, k)) + f" | {len(g)} | "
                  f"{int(g['exec_improved'].sum())} | {int(g['exec_regressed'].sum())} | "
                  f"{int(g['ctime_regressed'].sum())} | {gmean(g['speedup']):.3f} | "
                  f"{gmean(g['ctime_ratio']):.3f} |")
    changed = rep[rep["exec_improved"] | rep["exec_regressed"] | rep["ctime_regressed"]]
    if not changed.empty:
        md += ["", "## Problems that changed", "",
               f"| experiment | problem | {args.base} time (ms) | {args.cand} time (ms) "
               "| speedup | ctime ratio |", "|---|---|---:|---:|---:|---:|"]
        for _, r in changed.iterrows():
            tag = ("improved" if r["exec_improved"] else
                   "regressed" if r["exec_regressed"] else "")
            tag += " ctime regressed" if r["ctime_regressed"] else ""
            md.append(f"| {r['id']} {r['dt']} t{r['threads']} | <details><summary>"
                      f"{tag.strip()}</summary>{r['prb']}</details> | "
                      f"{r['base_median_ms']:.3g} | {r['cand_median_ms']:.3g} | "
                      f"{r['speedup']:.3g}x | {r['ctime_ratio']:.3g} |")
    path = os.path.join(args.csv_dir, "perfci_report.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:len(md) if changed.empty else md.index("## Problems that changed")]))
    print(f"{len(rep)} problems; {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
