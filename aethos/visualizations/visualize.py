import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_bokeh
import plotly.express as px
import ptitprince as pt
import seaborn as sns
from aethos.config import IMAGE_DIR, cfg
from aethos.util import _make_dir
from bokeh.io import export_png
from scipy import stats
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.decomposition import PCA, TruncatedSVD



def raincloud(col: str, target_col: str, data: pd.DataFrame, output_file="", **params):
    """
    Visualizes 2 columns using raincloud.
    
    Parameters
    ----------
    col : str
        Column name of general data

    target_col : str
        Column name of measurable data, numerical

    data : Dataframe
        Dataframe of the data

    params: dict
        Parameters for the RainCloud visualization

    ouput_file : str
        Output file name for the image including extension (.jpg, .png, etc.)
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    if not params:
        params = {
            "pointplot": True,
            "width_viol": 0.8,
            "width_box": 0.4,
            "orient": "h",
            "move": 0.0,
            "ax": ax,
        }

    ax = pt.RainCloud(x=col, y=target_col, data=data.infer_objects(), **params)

    if output_file:  # pragma: no cover
        fig.savefig(os.path.join(IMAGE_DIR, output_file))


def barplot(
    x,
    y,
    data,
    method=None,
    orient="v",
    stacked=False,
    output_file="",
    **barplot_kwargs,
):
    """
    Visualizes a bar plot.

    Kwargs are bokeh plot, vbar, and hbar bokeh plots.
    
    Parameters
    ----------
    x : str
        Column name for the x axis.

    y : list
        Columns for the y axis

    data : Dataframe
        Dataset

    method : str
        Method to aggregate groupy data
        Examples: min, max, mean, etc., optional
        by default None

    orient : str, optional
        Orientation of graph, 'h' for horizontal
        'v' for vertical, by default 'v'

    stacked : bool
        Whether to stack the different columns resulting in a stacked bar chart,
        by default False
    """

    alpha = barplot_kwargs.pop("alpha", 0.6)
    y.append(x)

    data_copy = data[y].copy()
    data_copy = data_copy.groupby(x, as_index=False)
    data_copy = getattr(data_copy, method)()
    data_copy = data_copy.set_index(x)

    if orient == "v":

        p_bar = data_copy.plot_bokeh.bar(
            stacked=stacked, alpha=alpha, **barplot_kwargs,
        )

    else:
        p_bar = data_copy.plot_bokeh.barh(
            stacked=stacked, alpha=alpha, **barplot_kwargs
        )

    if output_file:  # pragma: no cover

        if Path(output_file).suffix == ".html":
            pandas_bokeh.output_file(os.path.join(IMAGE_DIR, output_file))
        else:
            export_png(p_bar, os.path.join(IMAGE_DIR, output_file))


def scatterplot(
    x: str,
    y: str,
    z=None,
    data=None,
    category=None,
    title="Scatter Plot",
    output_file="",
    **scatterplot_kwargs,
):
    """
    Plots a scatter plot.
    
    Parameters
    ----------
    x : str
        X axis column

    y : str
        Y axis column

    z: str
        Z axis column

    data : Dataframe
        Dataframe

    category : str, optional
        Category to group your data, by default None

    title : str, optional
        Title of the plot, by default 'Scatterplot'

    size : int or str, optional
        Size of the circle, can either be a number
        or a column name to scale the size, by default 8

    output_file : str, optional
        If a name is provided save the plot to an html file, by default ''
    """

    if z is None:

        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=category,
            title=title,
            template="plotly_white",
            **scatterplot_kwargs
        )

    else:
        fig = px.scatter_3d(data, x=x, y=y, z=z, color=category, title=title, **scatterplot_kwargs)

    if output_file:  # pragma: no cover
        fig.write_image(os.path.join(IMAGE_DIR, output_file))

    fig.show()


def lineplot(
    x: str, y: list, data, title="Line Plot", output_file="", **lineplot_kwargs
):
    """
    Plots a line plot.
    
    Parameters
    ----------
    x : str
        X axis column

    y : list
        Y axis column

    data : Dataframe
        Dataframe

    title : str, optional
        Title of the plot, by default 'Line Plot'

    output_file : str, optional
        If a name is provided save the plot to an html file, by default ''
    """

    y.append(x)
    data_copy = data[y].copy()
    data_copy = data_copy.set_index(x)
    xlabel = lineplot_kwargs.pop("xlabel", x)

    p_line = data_copy.plot_bokeh.line(title=title, xlabel=xlabel, **lineplot_kwargs)

    if output_file:  # pragma: no cover

        if Path(output_file).suffix == ".html":
            pandas_bokeh.output_file(os.path.join(IMAGE_DIR, output_file))
        else:
            export_png(p_line, os.path.join(IMAGE_DIR, output_file))


def correlation_matrix(
    df, data_labels=False, hide_mirror=False, output_file="", **kwargs
):
    """
    Plots a correlation matrix.
    
    Parameters
    ----------
    df : DataFrame
        Data

    data_labels : bool, optional
        Whether to display the correlation values, by default False

    hide_mirror : bool, optional
        Whether to display the mirroring half of the correlation plot, by default False

    ouput_file : str
        Output file name for the image including extension (.jpg, .png, etc.)
    """

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))

    if hide_mirror:
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = None

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        mask=mask,
        annot=data_labels,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        **kwargs,
    )

    if output_file:  # pragma: no cover
        fig.savefig(os.path.join(IMAGE_DIR, output_file))


def pairplot(
    df, kind="scatter", diag_kind="auto", upper_kind=None, lower_kind=None, hue=None, output_file=None, **kwargs
):
    """
    Plots pairplots of the variables in the DataFrame
    
    Parameters
    ----------
    df : DataFrame
        Data

    kind : {'scatter', 'reg'}, optional
        Type of plot for off-diag plots, by default 'scatter'

    diag_kind : {'auto', 'hist', 'kde'}, optional
        Type of plot for diagonal, by default 'auto'

    upper_kind : str {'scatter', 'kde'}, optional
        Type of plot for upper triangle of pair plot, by default None

    lower_kind : str {'scatter', 'kde'}, optional
        Type of plot for lower triangle of pair plot, by default None

    hue : str, optional
        Column to colour points by, by default None

    ouput_file : str
        Output file name for the image including extension (.jpg, .png, etc.)
    """

    plot_mapping = {
        'scatter': plt.scatter,
        'kde': sns.kdeplot,
        'hist': sns.distplot
    }

    palette = kwargs.pop("color", sns.color_palette("pastel"))

    if upper_kind or lower_kind:
        assert (upper_kind is not None), "upper_kind cannot be None."
        assert (lower_kind is not None), "lower_kind cannot be None."
        assert (diag_kind is not "auto"), "diag_kind must be either `hist` or `kde`."

        g = sns.PairGrid(df, hue=hue, palette=palette, **kwargs)
        g = g.map_upper(plot_mapping[upper_kind])
        g = g.map_diag(plot_mapping[diag_kind])
        g = g.map_lower(plot_mapping[lower_kind])
        g = g.add_legend()        

    else:
        g = sns.pairplot(
            df, kind=kind, diag_kind=diag_kind, hue=hue, palette=palette, **kwargs
        )

    if output_file:  # pragma: no cover
        g.savefig(os.path.join(IMAGE_DIR, output_file))


def jointplot(x, y, df, kind="scatter", output_file="", **kwargs):
    """
    Plots a joint plot of 2 variables.
    
    Parameters
    ----------
    x : str
        X axis column

    y : str
        y axis column

    df : DataFrame
        Data

    kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
        Kind of plot to draw, by default 'scatter'

    ouput_file : str
        Output file name for the image including extension (.jpg, .png, etc.)
    """

    # NOTE: Ignore the deprecation warning for showing the R^2 statistic until Seaborn reimplements it
    import warnings

    warnings.simplefilter("ignore", UserWarning)

    sns.set(style="ticks", color_codes=True)
    color = kwargs.pop("color", "crimson")

    g = sns.jointplot(x=x, y=y, data=df, kind=kind, color=color, **kwargs).annotate(
        stats.pearsonr
    )

    if output_file:  # pragma: no cover
        g.savefig(os.path.join(IMAGE_DIR, output_file))


def histogram(x: list, x_train: pd.DataFrame, x_test=None, output_file="", **kwargs):
    """
    Plots a histogram.
    
    Parameters
    ----------
    x : list
        Columns to plot histogram for.

    x_train : pd.DataFrame
        Dataframe of the data.

    x_test : pd.DataFrame
        Dataframe of the data.

    ouput_file : str
        Output file name for the image including extension (.jpg, .png, etc.)
    """

    import math

    sns.set(style="ticks", color_codes=True)
    sns.set_palette(sns.color_palette("pastel"))

    # Make the single plot look pretty
    if len(x) == 1:
        data = x_train[~x_train[x[0]].isnull()]
        g = sns.distplot(data[x], label='Train Data', **kwargs)

        if x_test is not None:
            data = x_test[~x_test[x[0]].isnull()]
            g = sns.distplot(data[x], label='Test Data', **kwargs)
            g.legend(loc='upper right')

        g.set_title(f'Histogram for {x[0].capitalize()}')        

    else:
        n_cols = 2
        n_rows = math.ceil(len(x) / n_cols)

        _, ax = plt.subplots(n_rows, n_cols, figsize=(30, 10 * n_cols))

        for ax, col in zip(ax.flat, x):
            g = None

            data = x_train[~x_train[col].isnull()]
            g = sns.distplot(data[col], ax=ax, label='Train Data', **kwargs)

            if x_test is not None:
                data = x_test[~x_test[col].isnull()]
                g = sns.distplot(data[col], ax=ax, label='Test Data', **kwargs)
                ax.legend(loc='upper right')
            
            ax.set_title(f'Histogram for {col.capitalize()}', fontsize=20)

        plt.tight_layout()

    if output_file:  # pragma: no cover
        g.figure.savefig(os.path.join(IMAGE_DIR, output_file))

def viz_clusters(data: pd.DataFrame, algo: str, category: str, dim=2, output_file="", **kwargs):
    """
    Visualize clusters
    
    Parameters
    ----------
    data : pd.DataFrame
        Data

    algo : str {'tsne', 'lle', 'pca', 'tsvd'}, optional
        Algorithm to reduce the dimensions by, by default 'tsne'

    category : str
        Column name of the labels/data points to highlight in the plot

    dim : int {2, 3}
        Dimensions of the plot to show, either 2d or 3d, by default 2

    output_file : str, optional
        Output file name for image with extension (i.e. jpeg, png, etc.)
    """

    if dim != 2 and dim != 3:
        raise ValueError("Dimension must be either 2d (2) or 3d (3)")

    algorithms = {
        'tsne': TSNE(n_components=dim, random_state=42,),
        'lle': LocallyLinearEmbedding(n_components=dim, random_state=42,),
        'pca': PCA(n_components=dim, random_state=42,),
        'tsvd': TruncatedSVD(n_components=dim, random_state=42,), 
    }

    reducer = algorithms[algo]
    reduced_df = pd.DataFrame(reducer.fit_transform(data.drop(category, axis=1)))
    reduced_df.columns = map(str, reduced_df.columns)
    reduced_df[category] = data[category]
    reduced_df[category] = reduced_df[category].astype(str)

    if dim == 2:
        scatterplot(
            "0",
            "1",
            data=reduced_df,
            category=category,
            output_file=output_file,
            **kwargs,
        )
    else:
        scatterplot(
            "0",
            "1",
            "2",
            data=reduced_df,
            category=category,
            output_file=output_file,
            **kwargs,
        )
