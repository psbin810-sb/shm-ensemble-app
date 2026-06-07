"""plotting.py — 변형형상+축응력 색분포 + 앙상블 불확실성 (streamlit 비의존)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

from ensemble_core import (ELEMENTS_1BASED, NODE_COORDS, ELEM_SEGMENTS_UNDEF,
                           SENSOR_NODE_IDS_1B, SENSOR_FILE_TAG)


def draw_response(u, sigma, scale_factor, vmax=None, title="", u_std=None):
    """u (102,2) mm, sigma (201,) MPa. u_std (102,2) 주어지면 불확실성 구름 오버레이."""
    deformed = NODE_COORDS + u * scale_factor
    seg_def = np.array([[deformed[a - 1], deformed[b - 1]]
                        for _e, a, b, _s in ELEMENTS_1BASED])
    if vmax is None:
        vmax = max(float(np.abs(sigma).max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.add_collection(LineCollection(ELEM_SEGMENTS_UNDEF, colors="0.78",
                                     linestyles="--", linewidths=0.6, alpha=0.7))
    # 불확실성 점 (노드별 |u_std| 크기의 작은 반투명 점) — 센서(초록) 원보다 작게
    if u_std is not None:
        unc_mm = np.linalg.norm(u_std, axis=1)                  # (102,) mm (과장 안 함)
        ax.scatter(deformed[:, 0], deformed[:, 1],
                   s=np.clip(unc_mm * 12, 2, 24), color="orange", alpha=0.30,
                   zorder=2, label="uncertainty (±std)")
    lc = LineCollection(seg_def, cmap="RdBu_r", norm=norm, linewidths=2.4, zorder=3)
    lc.set_array(sigma)
    ax.add_collection(lc)
    sidx = SENSOR_NODE_IDS_1B - 1
    ax.scatter(deformed[sidx, 0], deformed[sidx, 1], s=55, facecolors="none",
               edgecolors="lime", linewidths=1.8, zorder=6, label="sensors")

    cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("axial stress  sigma = E*eps  [MPa]   (tension + / compression -)")
    xmin = min(NODE_COORDS[:, 0].min(), deformed[:, 0].min())
    xmax = max(NODE_COORDS[:, 0].max(), deformed[:, 0].max())
    ymin = min(NODE_COORDS[:, 1].min(), deformed[:, 1].min())
    ymax = max(NODE_COORDS[:, 1].max(), deformed[:, 1].max())
    mx, my = (xmax - xmin) * 0.04, (ymax - ymin) * 0.08
    ax.set_xlim(xmin - mx, xmax + mx); ax.set_ylim(ymin - my, ymax + my)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    ax.set_title(title + f"   (deformation x{scale_factor:g})", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def bar_wind_coefs(c_mean, c_std=None):
    """풍하중계수 c1~c4 막대 + 앙상블 오차막대."""
    labels = ["c1\nL-col", "c2\nL-roof", "c3\nR-roof", "c4\nR-col"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(c_mean))
    err = c_std if c_std is not None else None
    ax.bar(x, c_mean, yerr=err, capsize=4, color="steelblue",
           edgecolor="k", alpha=0.85, error_kw={"ecolor": "darkorange", "lw": 1.5})
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("wind load coef")
    ax.set_title("Estimated wind load (mean +/- ensemble std)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
