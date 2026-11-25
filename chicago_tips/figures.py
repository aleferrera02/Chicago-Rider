"""Matplotlib reproductions of the R displays, with the original dimensions and colours."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .geography import AREA_REGION, REGION_COLORS


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 12, "axes.labelsize": 13,
        "axes.labelpad": 9, "axes.edgecolor": "white", "axes.axisbelow": True,
        "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.grid": True, "grid.color": "#e7e7e7", "grid.linewidth": 0.65,
        "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d", "xtick.major.size": 0,
        "ytick.major.size": 0, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def export_figure(fig, stem, directory, formats=("png", "pdf"), dpi=300):
    """Use a stable R basename; keep the physical dimensions in every format."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = directory / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, facecolor="white")
        paths.append(path)
    return paths


def _lab_palette(name, endpoints):
    """Interpolate the two ggplot2 endpoint colours in CIE Lab (D65 white)."""
    rgb = np.array([colors.to_rgb(c) for c in endpoints])
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    matrix = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750],
                       [.0193339, .1191920, .9503041]])
    white = np.array([.95047, 1., 1.08883])
    delta = 6/29
    xyz = linear @ matrix.T / white
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz/(3*delta**2)+4/29)
    # Lab coordinates are a linear transform of f(X), f(Y), f(Z).
    steps = np.linspace(f[0], f[1], 256)
    xyz = np.where(steps > delta, steps**3, 3*delta**2*(steps-4/29))*white
    linear = np.clip(xyz @ np.linalg.inv(matrix).T, 0, 1)
    return colors.ListedColormap(np.where(linear <= .0031308, 12.92*linear,
                                          1.055*linear**(1/2.4)-.055), name=name)


def _gradient(ax, cmap, norm, title, count=False):
    panel = ax.get_subplotspec().get_position(ax.figure)
    bar_ax = ax.figure.add_axes([panel.x0 + .52*panel.width, panel.y1 + .058,
                                 .34*panel.width, .035])
    bar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                             cax=bar_ax, orientation="horizontal")
    ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], integer=count).tick_values(norm.vmin, norm.vmax)
    bar.set_ticks([v for v in ticks if norm.vmin <= v <= norm.vmax])
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=9, length=0, pad=1.5)
    ax.figure.text(panel.x0 + .49*panel.width, panel.y1 + .074, title,
                   ha="right", va="center", fontsize=12)


def time_heatmaps(sample):
    table = sample.groupby(["hour_start", "day_of_week"]).tip_given.agg(["size", "mean"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(left=.075, right=.99, bottom=.18, top=.78, wspace=.22)
    for ax, column, palette, title in zip(axes, ["size", "mean"],
            [("#C7DCE4", "#252D68"), ("#FFE6E6", "#AB080A")], ["# Trips", "Tip proportion"]):
        values = table[column].unstack().reindex(index=range(24), columns=range(7)).to_numpy()
        cmap = _lab_palette(column, palette)
        finite = values[np.isfinite(values)]
        norm = colors.Normalize(finite.min(), finite.max() if finite.max() > finite.min() else finite.min()+1)
        ax.grid(False)
        ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set(xticks=range(7), xticklabels="Mon Tue Wed Thu Fri Sat Sun".split(),
               yticks=range(0, 24, 3), xlabel="Day", ylabel="Hour")
        _gradient(ax, cmap, norm, title, count=column == "size")
    return fig


def _map_base(ax, areas, fills):
    all_rings = []
    for area, polygons in areas.items():
        for rings in polygons:
            paths = [MplPath(ring, closed=True) for ring in rings]
            ax.add_patch(PathPatch(MplPath.make_compound_path(*paths), facecolor=fills.get(area, "#999999"),
                                   edgecolor="#181818", lw=.42, zorder=2))
            all_rings.append(rings[0])
    points = np.concatenate(all_rings)
    lo, hi = points.min(axis=0), points.max(axis=0)
    margin = np.maximum((hi-lo)*.045, 1e-5)
    ax.set_xlim(lo[0]-margin[0], hi[0]+margin[0])
    ax.set_ylim(lo[1]-margin[1], hi[1]+margin[1])
    ax.set_aspect(1 / np.cos(np.deg2rad((hi[1]+lo[1])/2)), adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_locator(MultipleLocator(.05))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.1f}°W"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}°N"))


def region_map(areas):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=.12, right=.72, bottom=.08, top=.98)
    _map_base(ax, areas, {a: REGION_COLORS[r] for a, r in AREA_REGION.items()})
    handles = [Patch(facecolor=c, edgecolor="black", linewidth=.45, label=r)
               for r, c in REGION_COLORS.items()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, .5),
              fontsize=15, handlelength=1, handletextpad=.45, labelspacing=.34)
    return fig


def clustered_centroids(sample, end):
    coords = sample[[f"{end}_centroid_latitude", f"{end}_centroid_longitude"]].to_numpy()
    # Repeated centroids are common. Frequency-weighted unique points avoid a dense
    # 100,000 x 100,000 neighbourhood matrix. Keep their first-seen order for border ties.
    points, first, inverse, counts = np.unique(coords, axis=0, return_index=True,
                                                return_inverse=True, return_counts=True)
    order = np.argsort(first)
    labels = DBSCAN(eps=.005, min_samples=5, algorithm="kd_tree").fit_predict(
        points[order], sample_weight=counts[order])
    label_by_point = np.empty_like(labels)
    label_by_point[order] = labels
    df = pd.DataFrame(coords, columns=["lat", "lon"])
    df["cluster"] = label_by_point[np.asarray(inverse).ravel()]
    # R also aggregates its noise label (0) into a single marker.
    return df.groupby("cluster", sort=True).agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size"))


def trip_maps(trips, sample, areas, end):
    if end not in ("pickup", "dropoff"):
        raise ValueError("end must be 'pickup' or 'dropoff'")
    clusters = clustered_centroids(sample, end)
    means = trips.groupby(f"{end}_community_area").tip_given.mean()
    color, palette = (("#8A051F", ["#FBA1A8", "#371D51"]) if end == "pickup" else
                      ("#2C3F64", ["#B2E1BE", "#454B52"]))
    cmap = _lab_palette(end, palette)
    low, high = means.min(), means.max()
    norm = colors.Normalize(low, high if high > low else low+1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(left=.075, right=.995, top=.78, bottom=.10, wspace=.18)
    _map_base(axes[0], areas, dict.fromkeys(areas, "#DFD955"))
    _map_base(axes[1], areas, {int(a): cmap(norm(v)) for a, v in means.items()})
    # ggplot2 scale_size maps square-root count to a marker diameter.
    nmin, nmax = clusters.n.min(), clusters.n.max()
    def diameter(n):
        return 2.5 + 11.5*np.sqrt(np.maximum(0, (n-nmin) / max(nmax-nmin, 1)))
    axes[0].scatter(clusters.lon, clusters.lat, s=diameter(clusters.n)**2, c=color,
                    alpha=.5, linewidths=0, zorder=3)
    levels = [n for n in MaxNLocator(4, integer=True).tick_values(nmin, nmax) if nmin <= n <= nmax and n>0]
    if not levels:
        levels = [nmax]
    handles = [Line2D([], [], lw=0, label="# Trips")] + [
        Line2D([], [], marker="o", lw=0, color=color, markeredgewidth=0, alpha=.5,
               markersize=diameter(n), label=f"{int(n)}") for n in levels]
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.025),
                   fontsize=10, ncol=len(handles), handlelength=.9, handletextpad=.25, columnspacing=.55)
    _gradient(axes[1], cmap, norm, "Tip Proportion")
    for ax in axes:
        ax.set_xticks([-87.8, -87.6])
        ax.tick_params(labelsize=10)
    return fig


def routes_chart(trips):
    rates = trips.groupby("route", observed=True).tip_given.mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(left=.19, right=.985, top=.985, bottom=.18)
    ax.barh(rates.index.astype(str), rates, height=.88, color="#798675")
    ax.set(xlabel="Tip Proportion", ylabel="Route")
    ax.margins(y=.015)
    ax.tick_params(axis="y", labelsize=11)
    return fig


def correlation_chart(sample):
    columns = ["fare", "trip_miles", "additional_charges", "trip_seconds"]
    labels = ["Fare", "Miles", "Charges", "Seconds"]
    matrix = sample[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=.14, right=.91, top=.82, bottom=.03)
    ax.grid(False)
    image = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.xaxis.tick_top()
    ax.set(xticks=range(4), xticklabels=labels, yticks=range(4), yticklabels=labels)
    ax.tick_params(labelsize=15, colors="black")
    ax.tick_params(axis="x", labelrotation=90)
    bar = fig.colorbar(image, ax=ax, fraction=.115, pad=.04, ticks=np.linspace(-1, 1, 11))
    bar.outline.set_linewidth(.5)
    bar.ax.tick_params(labelsize=12)
    return fig


def _binned_rates(values, tips, edges):
    fmt = lambda x: f"{x:.8g}"
    if len(edges) == 1:
        labels = [f"[{fmt(edges[0])},{fmt(edges[0])}]"]
        codes = np.where(values == edges[0], 0, -1)
    else:
        labels = [f"{'[' if i == 0 else '('}{fmt(a)},{fmt(b)}]"
                  for i, (a, b) in enumerate(zip(edges, edges[1:]))]
        codes = pd.cut(values, edges, labels=False, include_lowest=True).fillna(-1).astype(int)
    table = pd.DataFrame({"code": codes, "tip": tips}).groupby("code").tip.mean()
    ids = [i for i in range(len(labels)) if i in table.index]
    values = [table[i] for i in ids]
    names = [labels[i] for i in ids]
    if -1 in table.index:
        names.append("NA")
        values.append(table[-1])
    return names, values


def fare_chart(sample):
    # fare.R uses *fare* quantiles for both charts. Tied quantiles collapse to one boundary.
    edges = np.unique(np.quantile(sample.fare, np.linspace(0, 1, 9)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=.08, right=.985, top=.96, bottom=.27, wspace=.20)
    for ax, col, label, color in zip(axes, ["fare", "additional_charges"],
                                      ["Fare", "Additional Charges"], ["#3A1C5E", "#094047"]):
        intervals, values = _binned_rates(sample[col], sample.tip_given, edges)
        ax.bar(intervals, values, color=color, width=.9)
        ax.set_xlabel(label, fontsize=18)
        ax.tick_params(axis="y", labelsize=15)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    axes[0].set_ylabel("Tip Proportion", fontsize=18)
    return fig


def pooling_chart(trips):
    rates = trips.groupby("trips_pooled").tip_given.mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=.15, right=.985, top=.975, bottom=.17)
    ax.plot(rates.index, rates, color="#988EC3", lw=2.3)
    ax.scatter(rates.index, rates, color="#3C3156", s=30, zorder=3)
    ax.set_xlabel("Bookings pooled (including this trip)", fontsize=16)
    ax.set_ylabel("Rides with a recorded tip", fontsize=16)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True, steps=[1,2,5,10]))
    ax.tick_params(labelsize=16)
    return fig


def residual_hist(values, weights, lasso=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(left=.13, bottom=.18, right=.985, top=.98)
    width = np.ptp(values) / 29
    if width > 0:
        start = (np.floor(values.min()/width + .5) - .5) * width
        edges = start + np.arange(int(np.ceil((values.max()-start)/width))+1) * width
    else:
        edges = 30
    ax.hist(values, bins=edges, weights=weights, density=True,
            color="#014441" if lasso else "#589BAB", edgecolor="black", linewidth=.7)
    ax.set(xlabel="Deviance Residuals", ylabel="Relative Frequency")
    return fig


def roc_chart(curves, tree=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(left=.145, right=.98, top=.98, bottom=.205)
    if tree:
        fpr, tpr = curves["Tree"]
        ax.plot(fpr, tpr, color="#413E92", lw=2)
        ax.set(xlabel="False Positive Rate (1 - Specificity)", ylabel="True Positive Rate (Sensitivity)")
    else:
        for name, col, width in [("Lasso Regression", "#C0B1ED", 3.8),
                                  ("Logistic Regression", "#47316D", 1.15)]:
            fpr, tpr = curves[name]
            ax.plot(fpr, tpr, color=col, lw=width, label=name)
        ax.legend(loc="center", bbox_to_anchor=(.33, .79), fontsize=10)
        ax.set(xlabel="1 - Specificity (False Positive Rate)", ylabel="Sensitivity (True Positive Rate)")
    ax.plot([0,1], [0,1], "--", lw=1, color="#CC3337" if tree else "#101537")
    marks = np.linspace(0, 1, 5)
    ax.set(xticks=marks, yticks=marks, xlim=(-.045,1.045), ylim=(-.045,1.045))
    return fig


def confusion_tables(matrices):
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.9*len(matrices), 1.72), squeeze=False)
    fig.subplots_adjust(left=.025, right=.98, top=.98, bottom=.12, wspace=.13)
    for ax, (name, matrix) in zip(axes.flat, matrices.items()):
        ax.axis("off")
        cells = [["", "Predicted 0", "Predicted 1"]] + [
            [f"Actual {i}", *(f"{v:,}" for v in row)] for i, row in enumerate(matrix)]
        table = ax.table(cellText=cells, cellLoc="center", bbox=(.005,.30,.985,.66))
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        for cell in table.get_celld().values():
            cell.set_linewidth(.65)
            cell.get_text().set_fontfamily("serif")
        ax.text(.5, .10, name, ha="center", va="center", family="serif", fontsize=12)
    return fig


def importance_chart(importance):
    values = importance[importance > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(left=.28, right=.985, top=.98, bottom=.18)
    ax.barh(values.index, values, color="#FF800E", height=.9)
    ax.set_xlabel("Importance")
    return fig
