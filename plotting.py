import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def _label_rects(ax, rects):
    min_height = 1e8
    for rect in rects:
        height = rect.get_height()
        min_height = min(min_height, height)
        plt.text(rect.get_x() + rect.get_width() / 2.,
                 1.05 * height,
                 '%.2g' % height,
                 ha='center',
                 va='bottom')

    lims = ax.get_ylim()
    lims = (min_height / 2, 1.6 * lims[1])
    ax.set_ylim(lims)
    return ax


def plot_bar_chart(names,
                   height,
                   cmap="viridis_r",
                   title="",
                   ylabel="",
                   **kwargs):
    fig, ax = plt.subplots(1, 1)

    bar_cmap = cm.get_cmap(cmap)
    log_norm = LogNorm(vmin=min(height), vmax=max(height))

    rects = ax.bar(names, height, color=bar_cmap(log_norm(height)), **kwargs)
    ax = _label_rects(ax, rects)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax
