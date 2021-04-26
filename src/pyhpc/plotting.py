import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm


def plot_potential_grid(potential_grid, *args):
    num_ax = 1 + len(args)
    fig, axes = plt.subplots(ncols=num_ax)
    if num_ax == 1:
        axes = [axes]
    for ax, grid in zip(axes, [potential_grid, *args]):
        ax = _plot_grid(grid, ax)
    return fig, axes


def _plot_grid(grid, ax, **imshow_kwargs):
    ax.imshow(grid, origin="lower", **imshow_kwargs)
    ax.axis("off")
    return ax


def _label_rects(ax, rects, yscale):
    min_height = 1e8
    for rect in rects:
        height = rect.get_height()
        min_height = min(min_height, height)
        plt.text(
            rect.get_x() + rect.get_width()/2.,
            1.05*height,
            '%.2g' % height,
            ha='center',
            va='bottom'
        )

    if yscale == "Log":
        lims = ax.get_ylim()
        lims = (min_height/2, 1.6*lims[1])
        ax.set_ylim(lims)
    return ax


def plot_bar_chart(
    names,
    height,
    cmap="viridis_r",
    title="",
    ylabel="",
    yscale="Linear",
    **kwargs
):
    fig, ax = plt.subplots(1, 1)

    bar_cmap = cm.get_cmap(cmap)
    log_norm = LogNorm(vmin=min(height), vmax=max(height))

    rects = ax.bar(names, height, color=bar_cmap(log_norm(height)), **kwargs)
    ax = _label_rects(ax, rects, yscale)
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def animate_frames(frames):
    fig, ax = plt.subplots()
    ax = _plot_grid(frames[:, :, 0], ax)
    img = ax.get_images()[0]

    def animate_img(i):
        img.set_array(frames[:, :, i + 1])
        return [img]

    num_frames = frames.shape[2]
    return animation.FuncAnimation(
        fig, animate_img, frames=num_frames - 1, interval=50
    )
