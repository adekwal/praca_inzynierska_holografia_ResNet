import numpy as np
import matplotlib.pyplot as plt

def plot_charts(charts, titles=None, suptitle=None, cbar_label=None):
    n_charts = len(charts)
    n_cols = min(4, n_charts) # maximum 4 plots per row
    n_rows = int(np.ceil(n_charts / 4)) # calculate the number of rows

    # Calculate global vmin and vmax for consistent scaling
    vmin = min(chart.min() for chart in charts)
    vmax = max(chart.max() for chart in charts)

    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.subplots_adjust(right=0.85) # leave space for colorbar

    # Ensure axs is always 2D array for indexing, even with a single row
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    # Plot each chart
    for idx, chart in enumerate(charts):
        row, col = divmod(idx, n_cols)
        im = axs[row][col].imshow(chart, vmin=vmin, vmax=vmax)
        axs[row][col].set_title(titles[idx] if titles and idx < len(titles) else f"Chart {idx + 1}")

    # Hide unused subplots if any
    for idx in range(n_charts, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axs[row][col].axis('off')

    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label(cbar_label)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    plt.show()
