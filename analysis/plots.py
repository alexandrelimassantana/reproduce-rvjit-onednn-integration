#!/usr/bin/env python3
"""Plot the comparison of the flavors.

Usage: plots.py [csv-dir] [-o DIR] [--base upstream] [--cand rvjit]

Reads compare.csv, summary_by_impl.csv, summary_by_dataset.csv and
summary_by_primitive.csv (compare.py), and timings.csv (collect.py), from
csv-dir, and writes PNGs to DIR/<machine>/ (default figures/):

  summary.png                 geomean speedup over all problems of each
                              primitive's ensemble and of each dataset
  primitive.<driver>.png      one primitive, all its datasets together: per
                              (baseline impl, candidate impl), as by_impl
  report.<driver>.png         one primitive, one column per data type: the
                              speedup of every problem per dataset (box
                              plots), its heatmap over the problem's
                              dimensions, and the heatmap of the slower
                              problems only; report.md shows it
  dataset.<id>.png            one dataset (performance batch), all its data
                              types and tags: per implementation, as by_impl
  by_impl.png                 overview: geomean speedup of each candidate
                              implementation (rows) in each comparison
                              (columns), same-implementation problems only
  impl_overall.png            geomean speedup per primitive x implementation,
                              per data type and over all of them
  <comparison>.by_impl.png    one comparison per figure: per (baseline impl,
                              candidate impl) group, geomean GFLOPS of both
                              flavors, the speedup distribution, and the
                              geomean creation-time ratio
  <comparison>.speedup.png    one comparison: exec speedup and creation-time
                              ratio of each problem, sorted; problems on
                              different implementations are marked
  <experiment>.heatmap.png    matmul and ip: speedup over (M, N, K); conv:
                              over (OC, IC, output spatial size); per experiment x
                              data type x threads, same-implementation
                              problems
  timings.png                 total time of each phase (build and runs)

A comparison is one experiment x data type x weights tag x thread count.
Each case (unique problem, see compare.py) counts once: once per dataset in
the comparison, dataset and heatmap figures, once per primitive in the
primitive and report figures.
Speedup is the candidate's GFLOPS over the baseline's (> 1 is faster); the
creation-time ratio is the candidate's over the baseline's (< 1 is faster).
"""

import argparse
import os
import re

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

BASE_COLOR, CAND_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
# Diverging speedup: red (slower) <- neutral gray at 1x -> blue (faster)
SPEEDUP_CMAP = LinearSegmentedColormap.from_list(
    "speedup", ["#e34948", "#f0efec", "#2a78d6"])
SPEEDUP_NORM = LogNorm(vmin=0.5, vmax=2.0)

plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 8,
    "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight",
})


def gmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def slug(text):
    return re.sub(r"[^A-Za-z0-9.=-]+", "_", text).strip("_")


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def by_impl_figure(g, base, cand, title, path):
    """Per implementation group: GFLOPS bars, speedup box, ctime ratio."""
    bi, ci = f"{base}_impl", f"{cand}_impl"
    groups = list(g.groupby([bi, ci], sort=False))
    labels = [(c if b == c else f"{b} ->\n{c}") + f"\n(n={len(x)})"
              for (b, c), x in groups]
    pos = np.arange(len(groups))
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(4.5 + 2.6 * len(groups), 3.4), gridspec_kw={"wspace": 0.55})
    w = 0.38
    bg = [gmean(x[f"{base}_gflops"]) for _, x in groups]
    cg = [gmean(x[f"{cand}_gflops"]) for _, x in groups]
    ax1.bar(pos - w / 2 - 0.01, bg, w, color=BASE_COLOR, label=base)
    ax1.bar(pos + w / 2 + 0.01, cg, w, color=CAND_COLOR, label=cand)
    for p, v in zip(pos, bg):
        ax1.annotate(f"{v:.2g}", (p - w / 2, v), ha="center", va="bottom",
                     fontsize=7, color=INK, xytext=(0, 2), textcoords="offset points")
    for p, v in zip(pos, cg):
        ax1.annotate(f"{v:.2g}", (p + w / 2, v), ha="center", va="bottom",
                     fontsize=7, color=INK, xytext=(0, 2), textcoords="offset points")
    ax1.set_ylabel("GFLOP/s (geomean)")
    ax1.set_title("Throughput")
    ax1.legend(loc="upper left", bbox_to_anchor=(0, 1.0))
    ax1.set_ylim(0, max(bg + cg) * 1.25)

    data = [x["speedup"].dropna().values for _, x in groups]
    ax2.boxplot(data, positions=pos, widths=0.5, showfliers=False,
                medianprops={"color": INK}, boxprops={"color": MUTED},
                whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for p, d in zip(pos, data):
        jitter = (np.random.default_rng(0).random(len(d)) - 0.5) * 0.25
        ax2.scatter(p + jitter, d, s=9, color=CAND_COLOR, alpha=0.6, linewidths=0)
        gm = gmean(d)
        ax2.annotate(f"{gm:.2f}x", (p, gm), xytext=(14, 0), textcoords="offset points",
                     fontsize=7, color=INK, va="center")
    ax2.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax2.set_ylabel(f"speedup ({cand} / {base})")
    ax2.set_title("Speedup per problem")

    ct = [gmean(x["ctime_ratio"]) for _, x in groups]
    ax3.bar(pos, ct, 0.5, color=CAND_COLOR)
    for p, v in zip(pos, ct):
        ax3.annotate(f"{v:.2f}", (p, v), ha="center", va="bottom", fontsize=7,
                     color=INK, xytext=(0, 2), textcoords="offset points")
    ax3.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax3.set_ylabel(f"creation time ({cand} / {base})")
    ax3.set_title("Creation time (geomean)")
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(pos, labels, fontsize=7)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, fontsize=11, y=1.03)
    save(fig, path)


def speedup_figure(g, title, path):
    g = g.sort_values("speedup").reset_index(drop=True)
    x = np.arange(len(g))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    ax1.plot(x, g["speedup"], color=CAND_COLOR, linewidth=2, marker="o", markersize=3)
    diff = g[~g["same_impl"]]
    if len(diff):
        ax1.scatter(diff.index, diff["speedup"], marker="x", s=40, color=INK,
                    label="impl differs", zorder=3)
        ax1.legend(loc="upper left")
    ax1.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax1.set_yscale("log")
    ax1.set_ylabel("exec speedup")
    ax1.set_title(f"{title}: problems sorted by speedup (n={len(g)})")
    ax2.plot(x, g["ctime_ratio"], color=BASE_COLOR, linewidth=2, marker="o", markersize=3)
    ax2.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax2.set_yscale("log")
    ax2.set_ylabel("creation time ratio")
    ax2.set_xlabel("problem (sorted by exec speedup)")
    save(fig, path)


def parse_matmul(desc):
    try:
        src, wei = str(desc).split("_", 1)[0].split(":")[:2]
        src = [int(v) for v in src.split("x")]
        wei = [int(v) for v in wei.split("x")]
        return pd.Series({"M": src[-2], "K": src[-1], "N": wei[-1]})
    except (ValueError, IndexError):
        return pd.Series({"M": np.nan, "K": np.nan, "N": np.nan})


def parse_conv(desc):
    """(OC, IC, OD*OH*OW) of a benchdnn conv descriptor; benchdnn leaves out
    what it can infer (w copies h, h and w copy d, the output from the input)."""
    d = re.split(r"_?n[\"a-zA-Z]", str(desc), maxsplit=1)[0]
    v = {k: int(x) for k, x in re.findall(r"([a-z]+)(\d+)", d)}
    if "ic" not in v or "oc" not in v:
        return pd.Series({"OC": np.nan, "IC": np.nan, "S": np.nan})
    dims = "dhw" if "id" in v else "hw" if "ih" in v else "w"
    src = dims[0]
    size = 1
    for c in dims:
        get = lambda p, default=None: v.get(p + c, v.get(p + src, default))  # noqa: E731
        i, k, st, dl, o, p = (get("i"), get("k", 1), get("s", 1), get("d", 0),
                              get("o"), get("p", 0))
        if o is None and i is not None:
            o = (i - ((k - 1) * (dl + 1) + 1) + 2 * p) // st + 1
        size *= o or 1
    return pd.Series({"OC": v["oc"], "IC": v["ic"], "S": size})


def parse_ip(desc):
    """(M, N, K) of a benchdnn inner product: mb x oc x ic * spatial."""
    d = re.split(r"_?n[\"a-zA-Z]", str(desc), maxsplit=1)[0]
    v = {k: int(x) for k, x in re.findall(r"([a-z]+)(\d+)", d)}
    if "ic" not in v or "oc" not in v:
        return pd.Series({"M": np.nan, "N": np.nan, "K": np.nan})
    k = v["ic"]
    for c in "dhw":
        k *= v.get("i" + c, 1)
    return pd.Series({"M": v.get("mb", 2), "N": v["oc"], "K": k})


DIMS = {"matmul": (parse_matmul, ["M", "N", "K"]),
        "ip": (parse_ip, ["M", "N", "K"]),
        "conv": (parse_conv, ["OC", "IC", "S"])}
DIM_LABELS = {"S": "OD x OH x OW"}


def with_dims(g, driver):
    """g with the problem's dimensions as columns, and their names."""
    if driver not in DIMS or g.empty:
        return g.iloc[0:0], []
    parse, axes = DIMS[driver]
    dims = g["desc"].apply(parse)
    g = pd.concat([g.reset_index(drop=True), dims.reset_index(drop=True)], axis=1)
    return g.dropna(subset=axes), axes


def draw_heatmap(ax, g, axes, marked=None):
    """3D scatter of speedup over axes, in log2 space; problems in marked (a
    boolean column) drawn as crosses."""
    xyz = [np.log2(g[a].astype(float).clip(lower=1)) for a in axes]
    kw = dict(cmap=SPEEDUP_CMAP, norm=SPEEDUP_NORM, edgecolor="#52514e",
              linewidths=0.4)
    mask = g[marked].values if marked else np.zeros(len(g), dtype=bool)
    sc = ax.scatter(*(v[~mask] for v in xyz), c=g["speedup"][~mask].clip(0.5, 2.0),
                    s=40, **kw)
    if mask.any():
        ax.scatter(*(v[mask] for v in xyz), c=g["speedup"][mask].clip(0.5, 2.0),
                   s=60, marker="X", **kw)
    for setter, vals in zip((ax.set_xticks, ax.set_yticks, ax.set_zticks), xyz):
        ticks = np.unique(vals.round().astype(int))
        if len(ticks) > 6:
            ticks = ticks[:: int(np.ceil(len(ticks) / 6))]
        setter(ticks, [f"{2 ** t:,}" for t in ticks], fontsize=7)
    ax.set_xlabel(DIM_LABELS.get(axes[0], axes[0]), fontsize=9)
    ax.set_ylabel(DIM_LABELS.get(axes[1], axes[1]), fontsize=9)
    ax.set_zlabel(DIM_LABELS.get(axes[2], axes[2]), fontsize=9)
    return sc


def speedup_colorbar(fig, mappable, **kw):
    cbar = fig.colorbar(mappable, **kw)
    cbar.set_label("speedup (clipped to 0.5-2x)")
    cbar.set_ticks([0.5, 0.75, 1, 1.5, 2], labels=["0.5", "0.75", "1", "1.5", "2"])
    cbar.ax.minorticks_off()


def heatmap_figure(g, driver, title, path):
    g, axes = with_dims(g, driver)
    if len(g) < 8:
        return False
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sc = draw_heatmap(ax, g, axes)
    speedup_colorbar(fig, sc, ax=ax, shrink=0.6, pad=0.12)
    ax.set_title(f"{title} (n={len(g)})", fontsize=10)
    save(fig, path)
    return True


def dt_order(dt):
    """f32, then f16, bf16 and int8 families, as the README lists them."""
    families = ["f32", "f16", "bf16", "s8", "u8"]
    first = str(dt).split(":")[0]
    return (families.index(first) if first in families else len(families), dt)


def log_ticks(ax):
    """Readable ticks on a log speedup axis: round values within its range,
    1-2-5 steps only when the range is wide."""
    lo, hi = ax.get_ylim()
    fine = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4,
            5, 7]
    coarse = [m * 10 ** e for e in range(-2, 5) for m in (1, 2, 5)]
    medium = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10]
    ticks = [t for t in fine if lo <= t <= hi]
    if len(ticks) > 10:
        ticks = [t for t in medium if lo <= t <= hi]
    if hi > 10:
        ticks = [t for t in coarse if lo <= t <= hi]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def primitive_report_figure(g, per_dataset, driver, title, path):
    """One column per data type. Row 1: speedup of every case, one box over
    the primitive's unique cases (g) and one per dataset over its own
    (per_dataset). Row 2: heatmap of every case. Row 3: heatmap of the cases
    slower than the baseline. Cases that ran on different implementations
    are drawn as crosses."""
    dts = sorted(dict.fromkeys(g["dt"]), key=dt_order)
    n = len(dts)
    fig = plt.figure(figsize=(5.2 * n + 1.2, 15))
    grid = fig.add_gridspec(3, n, height_ratios=[1, 1.25, 1.25], hspace=0.25,
                            wspace=0.3)
    sc = None
    for col, dt in enumerate(dts):
        d = g[g["dt"] == dt]
        ax = fig.add_subplot(grid[0, col])
        groups = [("all", d)] + [
            (ds.replace(f"-{driver}-", "\n"), x)
            for ds, x in per_dataset[per_dataset["dt"] == dt].groupby("id", sort=True)]
        if len(groups) == 2:
            groups = groups[:1]
        data = [x["speedup"].dropna().values for _, x in groups]
        pos = np.arange(len(groups))
        ax.boxplot(data, positions=pos, widths=0.5, showfliers=False,
                   medianprops={"color": INK}, boxprops={"color": MUTED},
                   whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        rng = np.random.default_rng(0)
        for p, (_, x) in zip(pos, groups):
            x = x.dropna(subset=["speedup"])
            jitter = (rng.random(len(x)) - 0.5) * 0.25
            same = x["same_impl"].values
            ax.scatter(p + jitter[same], x["speedup"][same], s=9, color=CAND_COLOR,
                       alpha=0.6, linewidths=0)
            if (~same).any():
                ax.scatter(p + jitter[~same], x["speedup"][~same], s=18, marker="x",
                           color=INK, linewidths=0.8)
            gm = gmean(x["speedup"])
            ax.annotate(f"{gm:.2f}x", (p, gm), xytext=(16, 0),
                        textcoords="offset points", fontsize=7, color=INK,
                        va="center")
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        log_ticks(ax)
        ax.set_xticks(pos, [f"{name}\n(n={len(x)})" for name, x in groups],
                      fontsize=7)
        ax.grid(axis="x", visible=False)
        slower = int((d["speedup"] < 1).sum())
        ax.set_title(f"{dt}: {len(d)} unique cases, {slower} slower")
        if col == 0:
            ax.set_ylabel("speedup per problem")

        h, axes = with_dims(d, driver)
        for row, (sel, label) in enumerate(
                ((h, "all cases"), (h[h["speedup"] < 1], "slower cases")),
                start=1):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            ax.set_title(f"{dt}: {label} (n={len(sel)})", fontsize=9)
            if sel.empty:
                ax.set_axis_off()
                ax.text2D(0.5, 0.5, "none" if axes else "no dimensions to plot",
                          transform=ax.transAxes, ha="center", color=MUTED)
                continue
            sel = sel.assign(differs=~sel["same_impl"].astype(bool))
            sc = draw_heatmap(ax, sel, axes, marked="differs")
    if sc is not None:
        speedup_colorbar(fig, sc, ax=fig.axes, shrink=0.3, pad=0.04,
                         anchor=(0, 0.3))
    fig.suptitle(f"{title}; crosses: cases on different implementations",
                 fontsize=12, y=0.92)
    save(fig, path)


def overview_figure(by_impl, cand, title, path):
    s = by_impl[by_impl["same_impl"]]
    if s.empty:
        return
    table = s.pivot_table(index=f"{cand}_impl", columns="comparison",
                          values="speedup_geomean", aggfunc="first")
    counts = s.pivot_table(index=f"{cand}_impl", columns="comparison",
                           values="problems", aggfunc="first")
    fig, ax = plt.subplots(figsize=(max(5.0, 2.5 + 0.6 * table.shape[1]),
                                    1.6 + 0.5 * table.shape[0]))
    im = ax.imshow(table.values, cmap=SPEEDUP_CMAP, norm=SPEEDUP_NORM, aspect="auto")
    for (r, c), v in np.ndenumerate(table.values):
        if np.isfinite(v):
            ax.text(c, r, f"{v:.2f}\n({int(counts.values[r, c])})", ha="center",
                    va="center", fontsize=6, color=INK)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0.5, 0.75, 1, 1.5, 2], labels=["0.5", "0.75", "1", "1.5", "2"])
    cbar.ax.minorticks_off()
    cbar.set_label("geomean speedup")
    ax.set_title(f"{title}\ncell: geomean speedup (problems); empty: impl not reached",
                 fontsize=9)
    save(fig, path)


def overall_figure(overall, cand, title, path):
    if overall.empty:
        return
    overall = overall.copy()
    overall["group"] = overall["driver"] + "\n" + overall[f"{cand}_impl"]
    groups = list(dict.fromkeys(overall["group"]))
    dts = [d for d in dict.fromkeys(overall["dt"]) if d != "all"] + ["all"]
    w = 0.8 / len(dts)
    fig, ax = plt.subplots(figsize=(2 + 1.4 * len(groups), 3.6))
    shades = plt.get_cmap("Greys")(np.linspace(0.35, 0.75, max(len(dts) - 1, 1)))
    for j, dt in enumerate(dts):
        sub = overall[overall["dt"] == dt].set_index("group")
        vals = [sub["speedup_geomean"].get(g, np.nan) for g in groups]
        color = CAND_COLOR if dt == "all" else shades[j % len(shades)]
        pos = np.arange(len(groups)) - 0.4 + w * (j + 0.5)
        ax.bar(pos, vals, w * 0.92, color=color, label=dt)
        if dt == "all":
            for p, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.annotate(f"{v:.2f}x", (p, v), ha="center", va="bottom",
                                fontsize=7, color=INK, xytext=(0, 2),
                                textcoords="offset points")
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(groups)), groups, fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("geomean speedup")
    ax.legend(title="data type", ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title)
    save(fig, path)


def summary_figure(datasets, primitives, cand, title, path):
    """One bar per primitive ensemble, followed by its datasets."""
    d = datasets[datasets[f"{cand}_impl"] == "all"]
    p = primitives[(primitives[f"{cand}_impl"] == "all") & (primitives["dt"] == "all")]
    if d.empty:
        return
    labels, vals, counts, ensemble = [], [], [], []
    for driver in dict.fromkeys(d["driver"]):
        row = p[p["driver"] == driver]
        if not row.empty:
            labels.append(f"{driver}: all datasets")
            vals.append(row["speedup_geomean"].iat[0])
            counts.append(row["problems"].iat[0])
            ensemble.append(True)
        for _, r in d[d["driver"] == driver].iterrows():
            labels.append(f"    {r['id']}")
            vals.append(r["speedup_geomean"])
            counts.append(r["problems"])
            ensemble.append(False)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 0.8 + 0.38 * len(labels)))
    ax.barh(y, vals, 0.62, color=[CAND_COLOR if e else MUTED for e in ensemble])
    for i, (v, n) in enumerate(zip(vals, counts)):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}x ({int(n)})", (max(v, 1.0), i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.axvline(1.0, color=INK, linestyle="--", linewidth=1)
    ax.set_xlim(0, max([v for v in vals if np.isfinite(v)] + [1.0]) * 1.25)
    ax.set_yticks(y, labels, fontsize=8)
    for t, e in zip(ax.get_yticklabels(), ensemble):
        t.set_fontweight("bold" if e else "normal")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("geomean speedup over all problems (problems)")
    ax.set_title(title)
    save(fig, path)


def timings_figure(t, title, path):
    if t.empty:
        return
    t = t.copy()
    t["seconds"] = pd.to_numeric(t["seconds"], errors="coerce")
    # The outer phases (build, run) contain the others; show the leaves
    leaves = t[~t["phase"].isin(["build", "run"])]
    tot = leaves.groupby(["phase", "flavor"], sort=False)["seconds"].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.45 * len(tot)))
    left = np.zeros(len(tot))
    shades = plt.get_cmap("Greys")(np.linspace(0.3, 0.8, max(len(tot.columns), 1)))
    for k, flavor in enumerate(tot.columns):
        color = {"upstream": BASE_COLOR, "rvjit": CAND_COLOR}.get(flavor, shades[k])
        label = "shared" if flavor in ("-", "") else flavor
        ax.barh(tot.index, tot[flavor] / 60, left=left, color=color, label=label,
                edgecolor="white", linewidth=1)
        left += tot[flavor].values / 60
    for i, v in enumerate(left):
        ax.annotate(f"{v:.1f} min", (v, i), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("minutes (sum over runs)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title(title)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("csv_dir", nargs="?", default=os.path.join(here, "..", "csv"))
    parser.add_argument("-o", "--output", default=os.path.join(here, "..", "figures"))
    parser.add_argument("--base", default="upstream")
    parser.add_argument("--cand", default="rvjit")
    args = parser.parse_args()
    base, cand = args.base, args.cand

    def read(name, **kw):
        path = os.path.join(args.csv_dir, name)
        return pd.read_csv(path, **kw) if os.path.isfile(path) else pd.DataFrame()

    cmp_ = read("compare.csv", dtype={"threads": str, "dt": str, "wtag": str})
    by_impl = read("summary_by_impl.csv", dtype={"threads": str})
    datasets = read("summary_by_dataset.csv")
    primitives = read("summary_by_primitive.csv")
    timings = read("timings.csv", dtype=str)
    machines = set(cmp_.get("machine", pd.Series(dtype=str))) | set(
        timings.get("machine", pd.Series(dtype=str)))
    written = 0
    for machine in sorted(machines):
        out = os.path.join(args.output, slug(machine))
        os.makedirs(out, exist_ok=True)
        if not cmp_.empty:
            every = cmp_[cmp_["machine"] == machine]
            c = every[~every.get("dup_in_dataset", False)]
            unique = every[~every.get("dup_in_primitive", False)]
            for comp, g in c.groupby("comparison", sort=False):
                by_impl_figure(g, base, cand, f"{comp} on {machine}",
                               os.path.join(out, f"{slug(comp)}.by_impl.png"))
                speedup_figure(g, comp, os.path.join(out, f"{slug(comp)}.speedup.png"))
                written += 2
            for dataset, g in c.groupby("id", sort=False):
                by_impl_figure(g, base, cand, f"{dataset}, all data types, on {machine}",
                               os.path.join(out, f"dataset.{slug(dataset)}.png"))
                written += 1
            for driver, g in unique.groupby("driver", sort=False):
                by_impl_figure(g, base, cand,
                               f"{driver}, all datasets together, on {machine}",
                               os.path.join(out, f"primitive.{slug(driver)}.png"))
                primitive_report_figure(
                    g, c[c["driver"] == driver], driver, f"{driver}: {cand} vs {base} on {machine}",
                    os.path.join(out, f"report.{slug(driver)}.png"))
                written += 2
            summary_figure(datasets[datasets["machine"] == machine],
                           primitives[primitives["machine"] == machine], cand,
                           f"{cand} vs {base} per primitive and dataset on {machine}",
                           os.path.join(out, "summary.png"))
            written += 1
            same = c[c["same_impl"]]
            for (exp, driver, dt, thr), g in same.groupby(
                    ["id", "driver", "dt", "threads"], sort=False):
                name = slug(f"{exp} {dt} t{thr}")
                written += heatmap_figure(g, driver, f"{exp} {dt}, {thr} threads",
                                          os.path.join(out, f"{name}.heatmap.png"))
            overview_figure(by_impl[by_impl["machine"] == machine], cand,
                            f"{cand} vs {base} per implementation on {machine}",
                            os.path.join(out, "by_impl.png"))
            overall = primitives[(primitives["machine"] == machine)
                                 & (primitives[f"{cand}_impl"] != "all")]
            overall_figure(overall, cand,
                           f"{cand} vs {base}, same-implementation problems, on {machine}",
                           os.path.join(out, "impl_overall.png"))
            written += 2
        if not timings.empty:
            timings_figure(timings[timings["machine"] == machine],
                           f"Time per phase on {machine}",
                           os.path.join(out, "timings.png"))
            written += 1
    print(f"{written} figures in {args.output}")


if __name__ == "__main__":
    main()
