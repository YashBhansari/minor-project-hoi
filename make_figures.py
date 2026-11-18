import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import IdentityTransform

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "outputs" / "trained-evaluation"
OUT = ROOT / "figures"

PAPER_HICO = {"Full": 31.66, "Rare": 25.94, "Non-rare": 33.36}
PAPER_VCOCO = {"Scenario 1": 59.0, "Scenario 2": 64.5}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PAPER, RELEASED, OURS, OURS_CROSS = "#9a9994", "#eb6834", "#2a78d6", "#86b6ef"
DPI = 200
PX = DPI / 96


def metrics(name):
    return json.loads((EVAL / name / "metrics.json").read_text(encoding="utf-8"))


def new_figure(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=DPI, facecolor=SURFACE)
    style(ax)
    return fig, ax


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def round_bar(ax, x, height, width, color):
    (x0, y0), (x1, y1) = ax.transData.transform([(x - width / 2, 0), (x + width / 2, height)])
    r = min(4 * PX, (x1 - x0) / 2, y1 - y0)
    verts = [(x0, y0), (x0, y1 - r), (x0, y1), (x0 + r, y1), (x1 - r, y1),
             (x1, y1), (x1, y1 - r), (x1, y0), (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none",
                           transform=IdentityTransform(), clip_on=False))


def grouped_bars(ax, groups, series, ymax, fmt, bar_in=0.22, gap_in=0.22):
    fig = ax.figure
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, ymax)
    fig.canvas.draw()
    unit_in = ax.get_window_extent().width / DPI / (len(groups) + 0.2)
    width = bar_in / unit_in
    step = (bar_in + gap_in) / unit_in
    n = len(series)
    for s, (_, color, values) in enumerate(series):
        offset = (s - (n - 1) / 2) * step
        for g, value in enumerate(values):
            round_bar(ax, g + offset, value, width, color)
            ax.annotate(fmt(value, s), (g + offset, value), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(groups)), groups, fontsize=10, color=INK)


def legend(ax, entries, ncol):
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in entries]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol,
                    frameon=False, fontsize=9, handlelength=1.0, handleheight=1.0,
                    columnspacing=1.6, borderaxespad=0.2)
    for text in leg.get_texts():
        text.set_color(INK2)


def title(ax, text, sub):
    ax.text(0, 1.2, text, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.14, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def hico_headline():
    ours, released = metrics("hico-on-hico"), metrics("official-hico")
    keys = [("Full", "full_mAP"), ("Rare", "rare_mAP"), ("Non-rare", "non_rare_mAP")]
    series = [("UPT paper", PAPER, [PAPER_HICO[k] for k, _ in keys]),
              ("Authors' released checkpoint", RELEASED, [released[m] * 100 for _, m in keys]),
              ("Ours (trained here)", OURS, [ours[m] * 100 for _, m in keys])]
    fig, ax = new_figure(7.2, 4.0)
    fig.subplots_adjust(top=0.72, bottom=0.1, left=0.07, right=0.98)
    grouped_bars(ax, [k for k, _ in keys], series, 40, lambda v, s: f"{v:.2f}")
    ax.set_yticks(range(0, 41, 10))
    legend(ax, [(l, c) for l, c, _ in series], 3)
    title(ax, "HICO-DET: Default mAP (%) on test2015", "UPT-R50, 9,546 test images")
    save(fig, "hicodet_results.png")


def vcoco_headline():
    ours, released = metrics("vcoco-on-vcoco"), metrics("official-vcoco")
    keys = [("Scenario 1", "scenario_1"), ("Scenario 2", "scenario_2")]
    series = [("UPT paper", PAPER, [PAPER_VCOCO[k] for k, _ in keys]),
              ("Authors' released checkpoint", RELEASED,
               [released["scenarios_upt_corrected_AP_percent"][m] for _, m in keys]),
              ("Ours (trained here)", OURS,
               [ours["scenarios_upt_corrected_AP_percent"][m] for _, m in keys])]
    fig, ax = new_figure(7.2, 4.0)
    fig.subplots_adjust(top=0.72, bottom=0.1, left=0.07, right=0.98)
    grouped_bars(ax, [k for k, _ in keys], series, 80,
                 lambda v, s: f"{v:.1f}" if s == 0 else f"{v:.2f}")
    ax.set_yticks(range(0, 81, 20))
    legend(ax, [(l, c) for l, c, _ in series], 3)
    title(ax, "V-COCO: AP_role (%) on the test split", "UPT-R50, 4,532 test images, corrected by 25/24")
    save(fig, "vcoco_results.png")


def hico_per_class():
    ours = [v * 100 for v in metrics("hico-on-hico")["per_interaction_AP"]]
    released = [v * 100 for v in metrics("official-hico")["per_interaction_AP"]]
    n = len(ours)
    mo, mr = sum(ours) / n, sum(released) / n
    cov = sum((a - mo) * (b - mr) for a, b in zip(ours, released))
    r = cov / (sum((a - mo) ** 2 for a in ours) * sum((b - mr) ** 2 for b in released)) ** 0.5
    mad = sum(abs(a - b) for a, b in zip(ours, released)) / n
    higher = sum(a > b for a, b in zip(ours, released))
    fig, ax = new_figure(5.4, 5.4)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.plot([0, 100], [0, 100], color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(released, ours, s=26, color=OURS, alpha=0.55, linewidths=0, zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("Authors' released checkpoint: AP (%)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Ours: AP (%)", color=INK2, fontsize=9.5)
    ax.text(97, 6, f"Pearson r = {r:.3f}\nmean |difference| = {mad:.1f} AP points\n"
                   f"ours higher on {higher} of {n} classes",
            ha="right", va="bottom", fontsize=9, color=INK2, linespacing=1.5)
    ax.text(62, 52, "equal AP", rotation=45, fontsize=8.5, color=INK2, ha="center", va="center",
            rotation_mode="anchor")
    ax.text(0, 1.1, "HICO-DET: per-interaction AP", transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.04, "600 interaction classes, one dot each", transform=ax.transAxes, fontsize=9,
            color=INK2, va="bottom")
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.12, right=0.97)
    save(fig, "hicodet_per_class_ap.png")


def cross_dataset():
    hh, vh = metrics("hico-on-hico"), metrics("vcoco-on-hico")
    hv, vv = metrics("hico-on-vcoco"), metrics("vcoco-on-vcoco")
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 4.0), dpi=DPI, facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1, 2]}, sharey=True)
    for ax in (left, right):
        style(ax)
    fig.subplots_adjust(top=0.72, bottom=0.1, left=0.07, right=0.98, wspace=0.08)
    grouped_bars(left, ["Default Full mAP"], [
        ("Trained on HICO-DET", OURS, [hh["full_mAP"] * 100]),
        ("Trained on V-COCO", OURS_CROSS, [vh["full_mAP"] * 100])], 80, lambda v, s: f"{v:.2f}")
    keys = [("Scenario 1", "scenario_1"), ("Scenario 2", "scenario_2")]
    corrected = "scenarios_upt_corrected_AP_percent"
    grouped_bars(right, [k for k, _ in keys], [
        ("Trained on V-COCO", OURS, [vv[corrected][m] for _, m in keys]),
        ("Trained on HICO-DET", OURS_CROSS, [hv[corrected][m] for _, m in keys])], 80,
        lambda v, s: f"{v:.2f}")
    left.set_yticks(range(0, 81, 20))
    left.set_title("Tested on HICO-DET", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    right.set_title("Tested on V-COCO (AP_role, corrected)", loc="left", fontsize=10.5, color=INK,
                    fontweight="bold")
    fig.legend(handles=[Patch(facecolor=OURS, label="Model trained on the test dataset"),
                        Patch(facecolor=OURS_CROSS, label="Model trained on the other dataset")],
               loc="lower left", bbox_to_anchor=(0.05, 0.8), ncol=2, frameon=False, fontsize=9,
               handlelength=1.0, handleheight=1.0, labelcolor=INK2)
    fig.suptitle("Cross-dataset evaluation (%)", x=0.07, y=0.97, ha="left", va="top", fontsize=12,
                 fontweight="bold", color=INK)
    save(fig, "cross_dataset.png")


def vcoco_per_role():
    ours = metrics("vcoco-on-vcoco")["role_AP_percent"]["scenario_1"]
    cross = metrics("hico-on-vcoco")["role_AP_percent"]["scenario_1"]
    roles = sorted(ours, key=lambda k: ours[k] - cross[k], reverse=True)
    fig, ax = new_figure(7.2, 0.27 * len(roles) + 1.5)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(roles) - 0.3)
    ys = list(range(len(roles) - 1, -1, -1))
    for y, role in zip(ys, roles):
        ax.plot([cross[role], ours[role]], [y, y], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
        ax.scatter([cross[role]], [y], s=46, color=OURS_CROSS, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.scatter([ours[role]], [y], s=46, color=OURS, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(ys, [r.replace("_", " ").replace("-", " (") + (")" if "-" in r else "") for r in roles],
                  fontsize=9, color=INK)
    ax.set_xlabel("AP_role, Scenario 1 (%), raw evaluator values", color=INK2, fontsize=9.5)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=OURS, label="Trained on V-COCO"),
               Line2D([], [], marker="o", linestyle="", markersize=7, color=OURS_CROSS, label="Trained on HICO-DET")]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
                    fontsize=9, borderaxespad=0.2)
    for text in leg.get_texts():
        text.set_color(INK2)
    ax.text(0, 1.1, "V-COCO: per-role AP by training dataset", transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=INK, va="bottom")
    fig.subplots_adjust(top=1 - 0.95 / fig.get_figheight(), bottom=0.6 / fig.get_figheight() + 0.02,
                        left=0.26, right=0.97)
    save(fig, "vcoco_per_role_ap.png")


if __name__ == "__main__":
    hico_headline()
    vcoco_headline()
    hico_per_class()
    cross_dataset()
    vcoco_per_role()
