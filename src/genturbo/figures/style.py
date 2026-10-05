"""Small shared publication style; sizes are points at the final placed width."""
from pathlib import Path

import matplotlib.pyplot as plt

COLUMN_WIDTH = 3.25
PAGE_WIDTH = 6.75

# Keep method identities stable across tasks; diagnostic colour scales are separate.
COLOURS = {
    "tr_gp": "#0072B2", "kermut": "#0072B2", "turdbo": "#0072B2",
    "tanimoto": "#CC79A7",
    "unconditional": "#777777", "generator": "#777777",
    "unconditional_gp": "#56B4E9",
    "no_tr_no_gp": "#56B4E9", "no_tr_gp": "#AA4499",
    "tr_no_gp": "#009E73", "oracle_upper_bound": "#333333",
    "hamming_random": "#999999", "hamming_gp": "#332288",
    "hamming_gp_ei": "#332288",
    "adalead_gp_ei": "#882255", "adalead": "#009E73",
    "bo_evo": "#D55E00", "lambo2": "#9467BD", "holo_ga": "#E69F00",
    "grid_ga": "#009E73", "turdbo_sequence": "#0072B2", "adalead_official": "#882255",
}
LINE_STYLES = {
    "tr_gp": "-", "kermut": "-", "turdbo": "-", "tanimoto": "--",
    "unconditional": "-", "generator": "-", "unconditional_gp": "--",
    "no_tr_no_gp": "--", "no_tr_gp": "--", "tr_no_gp": "-.",
    "oracle_upper_bound": "--", "hamming_random": ":",
    "hamming_gp": "-.", "hamming_gp_ei": "-.",
    "adalead_gp_ei": ":", "adalead": "-.", "grid_ga": "--",
    "turdbo_sequence": "-", "adalead_official": ":",
    "bo_evo": ":", "lambo2": "--", "holo_ga": "--",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9, "axes.titlepad": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "legend.handlelength": 2.3, "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.5, "legend.handletextpad": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "axes.axisbelow": True,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": "#dedede", "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def style_axis(axis):
    axis.grid(False)
    axis.grid(axis="y")
    axis.spines[["top", "right"]].set_visible(False)


def legend_above(figure, axes, *, ncol, anchor=None, **kwargs):
    """One frameless, centred legend above the panel(s), in plotting order."""
    handles, labels = [], []
    for axis in axes:
        hs, ls = axis.get_legend_handles_labels()
        for handle, label in zip(hs, ls, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    location = {"loc": "outside upper center"} if anchor is None else {
        "loc": "upper center", "bbox_to_anchor": anchor}
    return figure.legend(handles, labels, **location, ncol=ncol,
                         frameon=False, borderaxespad=0.5, **kwargs)


def save_figure(figure, path, formats=("pdf", "png")):
    """Do not tight-crop: preserve the physical width used by LaTeX."""
    path = Path(path)
    for suffix in formats:
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        figure.savefig(path.with_suffix("." + suffix), dpi=300, metadata=metadata)
    plt.close(figure)
