import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Configuration
# ============================================================

SAVE_DIR = "./figures/time_block_sensitivity"
SAVE_NAME = "time_blocks_sptt_window_2x2_nature"
DPI = 600

SAVE_PDF = True
SAVE_SVG = True
SAVE_PNG = True
SAVE_JPG = True

# Equal-spacing x positions, but label with actual tested values
BLOCK_VALUES = [1, 4, 8, 12, 16, 20]
XPOS = np.arange(1, len(BLOCK_VALUES) + 1)

# ============================================================
# Data
# ============================================================

# -------------------------
# Forecasting - BPTT LSTM
# -------------------------
bptt_lstm_forecast = {
    "ETTh1": {
        "mean": [0.8047, 0.9002, 0.9671, 1.2938, 1.0049, 1.3568],
        "std":  [0.0758, 0.0438, 0.0536, 0.0661, 0.0930, 0.0591],
    },
    "ETTh2": {
        "mean": [1.7893, 3.8098, 3.4769, 3.8277, 3.4796, 4.4113],
        "std":  [0.3461, 0.2824, 0.8386, 0.5852, 0.6267, 1.0292],
    },
    "Exchange Rate": {
        "mean": [0.6663, 1.0104, 1.0909, 1.1093, 1.2027, 1.2893],
        "std":  [0.0719, 0.1401, 0.0411, 0.0696, 0.0452, 0.1301],
    },
    "Weather": {
        "mean": [0.3809, 0.4462, 0.4996, 0.5684, 0.5440, 0.6401],
        "std":  [0.0360, 0.0689, 0.0257, 0.1560, 0.0165, 0.0601],
    },
}

# -------------------------
# Forecasting - SPTT LSTM
# -------------------------
sptt_lstm_forecast = {
    "ETTh1": {
        "mean": [0.6749, 0.8336, 0.8420, 1.1007, 0.8608, 1.1184],
        "std":  [0.0319, 0.0296, 0.0231, 0.0501, 0.0147, 0.0634],
    },
    "ETTh2": {
        "mean": [0.9261, 2.5709, 2.8849, 3.0093, 3.2145, 2.9411],
        "std":  [0.0341, 0.1130, 0.2963, 0.5801, 0.5115, 0.4920],
    },
    "Exchange Rate": {
        "mean": [0.2090, 0.3932, 0.6410, 1.0176, 0.9037, 0.9734],
        "std":  [0.0519, 0.0167, 0.0814, 0.5531, 0.1852, 0.2668],
    },
    "Weather": {
        "mean": [0.2870, 0.3739, 0.3902, 0.4408, 0.4840, 0.4235],
        "std":  [0.0154, 0.0308, 0.0212, 0.0400, 0.0483, 0.0102],
    },
}

# -------------------------
# Forecasting - BPTT GRU
# -------------------------
bptt_gru_forecast = {
    "ETTh1": {
        "mean": [1.1348, 0.9322, 0.9845, 1.1177, 0.9908, 1.0865],
        "std":  [0.2374, 0.0958, 0.1032, 0.0847, 0.0803, 0.0468],
    },
    "ETTh2": {
        "mean": [1.5455, 4.0689, 3.6476, 3.9147, 3.4226, 3.7048],
        "std":  [0.0991, 0.2962, 0.2556, 0.6238, 1.2134, 1.0368],
    },
    "Exchange Rate": {
        "mean": [0.6945, 0.8113, 1.0731, 1.1313, 1.2264, 1.1997],
        "std":  [0.0282, 0.0965, 0.1267, 0.2019, 0.2964, 0.0636],
    },
    "Weather": {
        "mean": [0.3389, 0.4788, 0.5220, 0.5372, 0.5195, 0.5536],
        "std":  [0.0199, 0.0653, 0.0629, 0.1048, 0.0563, 0.0046],
    },
}

# -------------------------
# Forecasting - SPTT GRU
# -------------------------
sptt_gru_forecast = {
    "ETTh1": {
        "mean": [0.6687, 0.9287, 0.9605, 1.2556, 1.0086, 1.2707],
        "std":  [0.0243, 0.0325, 0.0773, 0.0538, 0.1742, 0.0896],
    },
    "ETTh2": {
        "mean": [0.9350, 2.2902, 2.5858, 2.4990, 3.2732, 3.2845],
        "std":  [0.0348, 0.0950, 0.4796, 0.2929, 0.6702, 0.7541],
    },
    "Exchange Rate": {
        "mean": [0.1824, 0.4043, 0.5179, 0.6697, 0.8274, 0.8868],
        "std":  [0.0055, 0.0454, 0.0680, 0.0637, 0.2102, 0.4590],
    },
    "Weather": {
        "mean": [0.2723, 0.3636, 0.3409, 0.3520, 0.3982, 0.4011],
        "std":  [0.0047, 0.0649, 0.0128, 0.0072, 0.0282, 0.0320],
    },
}

# -------------------------
# Classification - BPTT LSTM
# -------------------------
bptt_lstm_cls = {
    "FaceDetection": {
        "mean": [58.78, 49.08, 51.03, 52.25, 53.28, 50.10],
        "std":  [1.39, 1.12, 0.98, 2.13, 0.87, 2.58],
    },
    "JapaneseVowels": {
        "mean": [90.22, 91.51, 90.31, 86.23, 85.51, 87.23],
        "std":  [0.93, 0.26, 0.61, 2.45, 3.29, 1.18],
    },
    "PEMS-SF": {
        "mean": [84.48, 80.63, 82.63, 80.42, 82.29, 78.54],
        "std":  [2.83, 2.20, 1.62, 3.55, 1.57, 0.72],
    },
    "SCP1": {
        "mean": [77.60, 73.40, 77.89, 76.39, 81.37, 73.50],
        "std":  [3.83, 1.06, 5.85, 5.46, 3.71, 1.78],
    },
}

# -------------------------
# Classification - SPTT LSTM
# -------------------------
sptt_lstm_cls = {
    "FaceDetection": {
        "mean": [61.99, 50.04, 50.22, 50.91, 51.16, 51.12],
        "std":  [0.74, 0.56, 1.09, 1.48, 1.30, 0.88],
    },
    "JapaneseVowels": {
        "mean": [91.44, 91.58, 90.58, 89.81, 89.76, 88.99],
        "std":  [1.82, 0.70, 2.42, 0.58, 1.91, 1.35],
    },
    "PEMS-SF": {
        "mean": [88.91, 82.71, 73.75, 64.58, 64.06, 62.81],
        "std":  [1.39, 3.48, 3.58, 4.07, 3.09, 3.98],
    },
    "SCP1": {
        "mean": [68.06, 74.24, 74.54, 73.26, 73.84, 73.73],
        "std":  [1.20, 0.83, 0.28, 2.17, 1.12, 0.72],
    },
}

# -------------------------
# Classification - BPTT GRU
# -------------------------
bptt_gru_cls = {
    "FaceDetection": {
        "mean": [58.95, 49.05, 49.56, 49.29, 50.06, 49.71],
        "std":  [1.37, 2.13, 1.96, 1.49, 1.96, 1.48],
    },
    "JapaneseVowels": {
        "mean": [93.70, 90.90, 89.49, 89.04, 88.04, 84.69],
        "std":  [1.06, 1.66, 2.00, 0.31, 2.12, 2.57],
    },
    "PEMS-SF": {
        "mean": [81.13, 78.54, 78.96, 74.79, 76.04, 74.17],
        "std":  [3.46, 1.71, 2.00, 1.91, 2.19, 2.89],
    },
    "SCP1": {
        "mean": [74.31, 77.86, 75.17, 81.48, 84.14, 82.41],
        "std":  [1.96, 3.43, 2.52, 6.25, 3.71, 4.40],
    },
}

# -------------------------
# Classification - SPTT GRU
# -------------------------
sptt_gru_cls = {
    "FaceDetection": {
        "mean": [64.88, 50.58, 51.14, 52.34, 50.98, 51.04],
        "std":  [0.80, 1.69, 1.67, 1.12, 0.68, 1.30],
    },
    "JapaneseVowels": {
        "mean": [93.95, 94.25, 95.56, 89.22, 91.94, 91.85],
        "std":  [1.05, 1.15, 0.33, 1.37, 1.26, 0.72],
    },
    "PEMS-SF": {
        "mean": [84.13, 79.58, 73.59, 63.13, 57.71, 62.92],
        "std":  [4.11, 3.08, 1.72, 6.53, 2.95, 2.95],
    },
    "SCP1": {
        "mean": [70.28, 75.17, 75.56, 76.04, 79.40, 79.98],
        "std":  [1.40, 0.45, 2.27, 3.31, 3.55, 2.83],
    },
}

# ============================================================
# Style
# ============================================================

def setup_nature_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",   # if Arial exists, you can switch
        "font.size": 8.8,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.7,
        "xtick.labelsize": 8.1,
        "ytick.labelsize": 8.1,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# Muted palette
forecast_colors = {
    "ETTh1": "#7BA6D0",
    "ETTh2": "#E6A35A",
    "Exchange Rate": "#6FBE73",
    "Weather": "#B07AA1",
}

classification_colors = {
    "FaceDetection": "#B07AA1",
    "JapaneseVowels": "#E6A35A",
    "PEMS-SF": "#6FBE73",
    "SCP1": "#7BA6D0",
}

# Method styles
METHOD_STYLES = {
    "BPTT": {
        "linestyle": "-",
        "marker": "o",
    },
    "SPTT-Window": {
        "linestyle": "--",
        "marker": "s",
    },
}

# ============================================================
# Plot helpers
# ============================================================

def format_axis(ax):
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        direction="out",
        width=0.8,
        length=3.2,
    )

    ax.grid(axis="y", linestyle="-", linewidth=0.35, alpha=0.15)
    ax.grid(axis="x", visible=False)


def add_dataset_legend(ax, dataset_names, color_map, loc="best"):
    handles = [
        Line2D(
            [0], [0],
            color=color_map[name],
            linestyle="-",
            marker="o",
            linewidth=1.6,
            markersize=4.0,
            label=name,
        )
        for name in dataset_names
    ]
    leg = ax.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        title="Dataset",
        title_fontsize=7.8,
        handlelength=1.8,
        borderaxespad=0.25,
        labelspacing=0.30,
    )
    return leg


def plot_comparison_panel(
    ax,
    data_bptt,
    data_sptt,
    color_map,
    ylabel,
    panel_title,
    panel_letter,
    dataset_legend_loc="best",
):
    format_axis(ax)

    dataset_names = list(data_bptt.keys())

    for name in dataset_names:
        color = color_map[name]

        # BPTT
        y_b = np.array(data_bptt[name]["mean"], dtype=float)
        e_b = np.array(data_bptt[name]["std"], dtype=float)
        ax.errorbar(
            XPOS,
            y_b,
            yerr=e_b,
            color=color,
            linestyle=METHOD_STYLES["BPTT"]["linestyle"],
            marker=METHOD_STYLES["BPTT"]["marker"],
            markersize=4.2,
            linewidth=1.45,
            elinewidth=0.85,
            capsize=2.4,
            capthick=0.85,
            alpha=0.95,
            zorder=3,
        )

        # SPTT-Window
        y_s = np.array(data_sptt[name]["mean"], dtype=float)
        e_s = np.array(data_sptt[name]["std"], dtype=float)
        ax.errorbar(
            XPOS,
            y_s,
            yerr=e_s,
            color=color,
            linestyle=METHOD_STYLES["SPTT-Window"]["linestyle"],
            marker=METHOD_STYLES["SPTT-Window"]["marker"],
            markersize=4.0,
            linewidth=1.45,
            elinewidth=0.85,
            capsize=2.4,
            capthick=0.85,
            alpha=0.95,
            zorder=3,
        )

    ax.set_xlim(0.7, len(BLOCK_VALUES) + 0.3)
    ax.set_xticks(XPOS)
    ax.set_xticklabels([str(v) for v in BLOCK_VALUES])

    ax.set_xlabel("Number of time blocks")
    ax.set_ylabel(ylabel)

    ax.set_title(
        f"{panel_letter}  {panel_title}",
        loc="left",
        pad=5,
        fontweight="bold",
    )

    dataset_legend = add_dataset_legend(
        ax=ax,
        dataset_names=dataset_names,
        color_map=color_map,
        loc=dataset_legend_loc,
    )
    ax.add_artist(dataset_legend)


# ============================================================
# Main
# ============================================================

def main():
    setup_nature_style()
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.2, 8.2),
        facecolor="white",
    )

    # a: LSTM forecasting
    plot_comparison_panel(
        ax=axes[0, 0],
        data_bptt=bptt_lstm_forecast,
        data_sptt=sptt_lstm_forecast,
        color_map=forecast_colors,
        ylabel="MSE",
        panel_title="LSTM long-term forecasting",
        panel_letter="a",
        dataset_legend_loc="upper left",
    )

    # b: LSTM classification
    plot_comparison_panel(
        ax=axes[0, 1],
        data_bptt=bptt_lstm_cls,
        data_sptt=sptt_lstm_cls,
        color_map=classification_colors,
        ylabel="Accuracy (%)",
        panel_title="LSTM classification",
        panel_letter="b",
        dataset_legend_loc="lower left",
    )

    # c: GRU forecasting
    plot_comparison_panel(
        ax=axes[1, 0],
        data_bptt=bptt_gru_forecast,
        data_sptt=sptt_gru_forecast,
        color_map=forecast_colors,
        ylabel="MSE",
        panel_title="GRU long-term forecasting",
        panel_letter="c",
        dataset_legend_loc="upper left",
    )

    # d: GRU classification
    plot_comparison_panel(
        ax=axes[1, 1],
        data_bptt=bptt_gru_cls,
        data_sptt=sptt_gru_cls,
        color_map=classification_colors,
        ylabel="Accuracy (%)",
        panel_title="GRU classification",
        panel_letter="d",
        dataset_legend_loc="lower left",
    )

    # Figure-level method legend
    method_handles = [
        Line2D(
            [0], [0],
            color="#333333",
            linestyle=METHOD_STYLES["BPTT"]["linestyle"],
            marker=METHOD_STYLES["BPTT"]["marker"],
            linewidth=1.5,
            markersize=4.2,
            label="BPTT",
        ),
        Line2D(
            [0], [0],
            color="#333333",
            linestyle=METHOD_STYLES["SPTT-Window"]["linestyle"],
            marker=METHOD_STYLES["SPTT-Window"]["marker"],
            linewidth=1.5,
            markersize=4.0,
            label="SPTT-Window",
        ),
    ]

    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
        title="Method",
        title_fontsize=7.8,
    )

    fig.text(
        0.065,
        0.018,
        " ",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#444444",
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.93,
        bottom=0.08,
        hspace=0.34,
        wspace=0.25,
    )

    base = os.path.join(SAVE_DIR, SAVE_NAME)

    if SAVE_PDF:
        fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    if SAVE_SVG:
        fig.savefig(f"{base}.svg", bbox_inches="tight", facecolor="white")
    if SAVE_PNG:
        fig.savefig(f"{base}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    if SAVE_JPG:
        fig.savefig(f"{base}.jpg", dpi=DPI, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print("Saved:")
    if SAVE_PDF:
        print(f"  {base}.pdf")
    if SAVE_SVG:
        print(f"  {base}.svg")
    if SAVE_PNG:
        print(f"  {base}.png")
    if SAVE_JPG:
        print(f"  {base}.jpg")


if __name__ == "__main__":
    main()