import matplotlib.pyplot as plt


def plot_field_with_samples(ax, x1, x2, field, samples, title, cmap="viridis"):
    """Plot field heatmap and overlay sampled coordinates."""
    c = ax.contourf(x1, x2, field, levels=50, cmap=cmap)
    ax.scatter(samples[:, 0], samples[:, 1], s=22, c="white", edgecolor="black", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return c
