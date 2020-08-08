import itertools as itr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_CMAP = plt.get_cmap('tab10')
DEFAULT_FIGSIZE = (16, 9)
PHI = 1.618033988749895  # (1 + np.sqrt(5)) / 2  # golden ratio

randy = np.random.default_rng()


########################################################################################
# private utility functions
########################################################################################


def _argsort_by_group_median(x, group):
    """Indices for x sorted by median x value for each group"""
    mask = [group == g for g in np.unique(group)]
    medians = [np.median(x[m]) for m in mask]
    order = np.argsort(medians)
    indices = np.arange(len(x))
    ordered_indices = [indices[mask[i]] for i in order]
    return list(itr.chain(*ordered_indices)), order


def _conditional_ax(axis=None):
    """axis if specified, pyplot otherwise"""
    return plt if axis is None else axis


def _conditional_title(title=None, axis=None):
    """axis if specified, pyplot otherwise"""
    if title is not None:
        if axis is None:
            plt.title(title)
        else:
            axis.set_title(title)
    return None


def _default_alpha(n, alpha=None):
    """Default alpha level if no alpha is specified"""
    return min(max(25 / n, 0), 1) if alpha is None else alpha


def _jitter(x):
    """Random values to use in a 2D plot of 1D data"""
    range = np.std(x) / 2
    return randy.uniform(-range, range, len(x))


########################################################################################
# public utility functions
########################################################################################


def hide_ticks(axis):
    """Hides tick marks on a matplotlib axis, and makes axes use equal units"""
    axis.axes.xaxis.set_visible(False)
    axis.axes.yaxis.set_visible(False)
    return None


def hide_ticks_square(axis):
    """Hides tick marks on a matplotlib axis, and makes axes use equal units"""
    hide_ticks(axis)
    square_axes(axis)
    return None


def square_axes(axis):
    """Sets plot axis dimensions to equal units"""
    axis.set_aspect('equal', adjustable='datalim')
    return None


########################################################################################
# plots
########################################################################################


def emb_scatter(
    data,
    ax,
    xdim=0,
    ydim=1,
    labels=None,
    title=None,
    palette=DEFAULT_CMAP,
    alpha=None,
    hide_axes=False,
    hide_legend=False,
):
    """Scatter plot of 2 dimensions of an embedding in a subplot"""
    n = len(data)
    alpha = _default_alpha(n, alpha)

    legend = False if hide_legend or labels is None else 'brief'
    sns.scatterplot(
        data[:, xdim],
        data[:, ydim],
        hue=labels,
        ax=ax,
        alpha=alpha,
        palette=palette,
        legend=legend,
    )

    if hide_axes:
        hide_ticks_square(ax)
    else:
        ax.set_xlabel(f'Dimension {xdim}')
        ax.set_ylabel(f'Dimension {ydim}')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        square_axes(ax)
    _conditional_title(title, ax)
    return None


def parallel_coordinates(
    data, ax, labels=None, title=None, cmap=DEFAULT_CMAP, alpha=None
):
    """Parallel coordinates plot"""
    n = len(data)
    n_dims = data.shape[1]
    alpha = _default_alpha(n, alpha)
    plot_x = np.arange(n_dims)

    if labels is None:
        labels = np.zeros(n)

    if n_dims > 1:
        for y, label in zip(data, labels):
            ax.plot(plot_x, y, c=cmap(label), alpha=alpha)
            ax.xaxis.set_ticks(plot_x)
            ax.axes.yaxis.set_visible(False)
    else:
        x = _jitter(data)
        y = data[:, 0]
        emb_scatter(
            np.array([x, y]).T, ax, labels=labels, title=title, cmap=cmap, alpha=alpha
        )

    _conditional_title(title, ax)
    return None


def scatter_matrix(
    data,
    n_rows,
    n_cols,
    labels=None,
    figsize=DEFAULT_FIGSIZE,
    cmap=DEFAULT_CMAP,
    scatter_alpha=None,
    parallel_alpha=None,
):
    """Matrix of scatter plots showing several combinations of components"""
    n_dims = data.shape[1]
    n_rows = min(n_rows, n_dims - 1)
    n_cols = min(n_cols, n_dims - 1)
    multiple_cells = n_rows * n_cols > 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 9), constrained_layout=True)

    for row in range(n_rows):
        for col, col_dim in zip(range(n_cols), range(row + 1, row + n_cols + 1)):
            if multiple_cells:
                ax = axes[row, col]
            else:
                ax = axes

            if multiple_cells and col == n_cols - 1 and row == n_rows - 1:
                parallel_coordinates(
                    data,
                    ax,
                    labels=labels,
                    title='Parallel Coordinates',
                    cmap=cmap,
                    alpha=parallel_alpha,
                )

            elif col_dim < n_dims:
                emb_scatter(
                    data,
                    ax,
                    row,
                    col_dim,
                    labels=labels,
                    palette=cmap,
                    alpha=scatter_alpha,
                    hide_legend=True,
                )

            else:
                ax.axis('off')

    plt.show()
    return None


def scatter_parallel(
    data,
    labels=None,
    scatter_title=None,
    parallel_title=None,
    title=None,
    figsize=DEFAULT_FIGSIZE,
    cmap=DEFAULT_CMAP,
    scatter_alpha=None,
    parallel_alpha=None,
    hide_legend=False,
):
    """Paired scatter and parallel coordinates plots for visualizing embeddings"""
    n = len(data)
    x_is_flat = len(data.shape) == 1
    n_dims = 1 if x_is_flat else data.shape[1]

    n_cols = 2 if n_dims > 1 else 1
    fig, ax = plt.subplots(1, n_cols, figsize=figsize, constrained_layout=True)

    palette = cmap if labels is None else {x: cmap(x) for x in np.unique(labels)}
    scatter_params = {
        'ax': ax[0] if n_dims > 1 else ax,
        'labels': labels,
        'title': scatter_title,
        'palette': palette,
        'alpha': scatter_alpha,
        'hide_legend': hide_legend,
    }
    if n_dims > 1:
        emb_scatter(data, **scatter_params)
        parallel_coordinates(
            data,
            ax=ax[1],
            labels=labels,
            title=parallel_title,
            cmap=cmap,
            alpha=parallel_alpha,
        )

    else:
        x = data if x_is_flat else data[:, 0]
        y = _jitter(data)
        sns.swarmplot(x, ax=ax, hue=labels, cmap=cmap, alpha=scatter_alpha)
        ax.set_xlabel('Component 0')
        ax.xaxis.set_ticks([])
        _conditional_title(scatter_title, ax)

    if title is not None:
        plt.suptitle(title)
    plt.show()
    return None


def scatter_param_search(
    n_rows, n_cols, params, data_func, title=None, alpha=None, figsize=DEFAULT_FIGSIZE
):
    """
    Tests each of several parameters, plots results, and returns best parameters
    """
    best_score = np.NINF
    best_params = None
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    for p, ax in zip(params, axes.flat):
        points, labels, score, subtitle = data_func(**p)
        # print(f"scatter_param_search, points: {len(points)}, labels: {len(labels)}")

        # check for new best parameters
        if score > best_score:
            best_score = score
            best_params = p

        emb_scatter(
            points,
            ax,
            labels=labels,
            title=subtitle,
            alpha=alpha,
            palette='tab10',
            hide_axes=True,
            hide_legend=True,
        )

    if title is not None:
        plt.suptitle(title)
    plt.show()
    return best_params, best_score


def violin_scatter(
    cat,
    emb_data,
    xdim,
    ydim,
    palette,
    title=None,
    figsize=DEFAULT_FIGSIZE,
    alpha=None,
    scatter=True,
):
    """
    Paired violin-scatter plots showing the distribution of a categorical variable
    against 2 dimensions of a spatial embedding.

    Parameters
    ----------
    cat : string array
        The categorical (text) data determining the violin categories and the point
        colors in the scatter plot
    emb_data : number matrix
        Spatial embedding data used in the scatter plot and horizontal component of the
        violin plot
    xdim : int
        Dimension of emb_data for horizontal component of violin and scatter plots
    ydim : int
        Dimension of emb_data for vertical component of scatter plot
    palette
        seaborn palette
    title : string
    figsize : tuple(int, int)
    alpha : float
        Transparency of scatter plot points
    scatter : boolean
        Whether or not to include the scatter plot
    """
    # order categories by median value in xdim
    row_order, group_order = _argsort_by_group_median(emb_data[:, xdim], cat)
    plot_cat = cat[row_order]
    plot_emb = emb_data[row_order]

    if scatter:
        fig, ax = plt.subplots(
            1,
            2,
            figsize=figsize,
            constrained_layout=True,
            gridspec_kw={'width_ratios': [PHI, 1]},
        )
        violin_ax = ax[0]
        scatter_ax = ax[1]
    else:
        fig, violin_ax = plt.subplots(1, 1, figsize=figsize)

    # violin plot: categorical series vs. xdim
    try:
        sns.violinplot(
            x=plot_emb[:, xdim],
            y=plot_cat,
            scale='count',
            palette=palette,
            ax=violin_ax,
        )
        hide_legend = True
    except:
        sns.violinplot(x=plot_emb[:, xdim], y=plot_cat, scale='count', ax=violin_ax)
        hide_legend = False
    violin_ax.set_xlabel(f'Embedded Component {xdim}')
    violin_ax.xaxis.set_ticks([])
    violin_ax.set_ylabel('')

    # scatter plot: 2D distribution of categories
    if scatter:
        emb_scatter(
            plot_emb,
            scatter_ax,
            xdim,
            ydim,
            labels=plot_cat,
            palette=palette,
            alpha=alpha,
            hide_legend=hide_legend,
        )

    if title is not None:
        plt.suptitle(title)
    plt.show()
    return None
