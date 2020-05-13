import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from aethos.config import IMAGE_DIR, cfg
from aethos.util import _make_dir


class VizCreator(object):
    def raincloud(
        self, col: str, target_col: str, data: pd.DataFrame, output_file="", **params
    ):
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

        import ptitprince as pt

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

        return ax

    def barplot(
        self,
        x: str,
        y: str,
        data: pd.DataFrame,
        method=None,
        asc=None,
        output_file="",
        **barplot_kwargs,
    ):
        """
        Visualizes a bar plot.
        
        Parameters
        ----------
        x : str
            Column name for the x axis.

        y : str
            Columns for the y axis

        data : Dataframe
            Dataset

        method : str
            Method to aggregate groupy data
            Examples: min, max, mean, etc., optional
            by default None

        asc : bool
            To sort values in ascending order, False for descending
        """

        import plotly.express as px

        orient = barplot_kwargs.get("orientation", None)

        if method:
            if orient == "h":
                data = data.groupby(y, as_index=False)
            else:
                data = data.groupby(x, as_index=False)

            data = getattr(data, method)()

            if not y:
                y = data.iloc[:, 1].name

        if asc is not None:
            data[x] = data[x].astype(str)
            data = data.sort_values(y, ascending=asc)

        fig = px.bar(data, x=x, y=y, **barplot_kwargs)

        if asc is not None:
            fig.update_layout(xaxis_type="category")

        if output_file:  # pragma: no cover
            fig.write_image(os.path.join(IMAGE_DIR, output_file))

        return fig

    def scatterplot(
        self,
        x: str,
        y: str,
        z=None,
        data=None,
        color=None,
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

        color : str, optional
            Category to group your data, by default None

        title : str, optional
            Title of the plot, by default 'Scatterplot'

        size : int or str, optional
            Size of the circle, can either be a number
            or a column name to scale the size, by default 8

        output_file : str, optional
            If a name is provided save the plot to an html file, by default ''
        """

        if color:
            data[color] = data[color].astype(str)

        if z is None:
            fig = px.scatter(
                data, x=x, y=y, color=color, title=title, **scatterplot_kwargs
            )

        else:
            fig = px.scatter_3d(
                data, x=x, y=y, z=z, color=color, title=title, **scatterplot_kwargs
            )

        if output_file:  # pragma: no cover
            fig.write_image(os.path.join(IMAGE_DIR, output_file))

        return fig

    def lineplot(
        self,
        x: str,
        y: str,
        z: str,
        data,
        color=None,
        title="Line Plot",
        output_file="",
        **lineplot_kwargs,
    ):
        """
        Plots a line plot.
        
        Parameters
        ----------
        x : str
            X axis column

        y : str
            Y axis column

        z : str
            Z axis column

        data : Dataframe
            Dataframe

        color : str
            Column to draw multiple line plots of

        title : str, optional
            Title of the plot, by default 'Line Plot'

        output_file : str, optional
            If a name is provided save the plot to an html file, by default ''
        """

        if color:
            data[color] = data[color].astype(str)

        if z is None:
            fig = px.line(data, x=x, y=y, color=color, title=title, **lineplot_kwargs)

            fig.data[0].update(mode="markers+lines")

        else:
            fig = px.line_3d(
                data, x=x, y=y, z=z, color=color, title=title, **lineplot_kwargs
            )

        if output_file:  # pragma: no cover
            fig.write_image(os.path.join(IMAGE_DIR, output_file))

        return fig

    def viz_correlation_matrix(
        self, df, data_labels=False, hide_mirror=False, output_file="", **kwargs
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

        fig, ax = plt.subplots(figsize=(11, 9))

        if hide_mirror:
            mask = np.zeros_like(df, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
        else:
            mask = None

        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(
            df,
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

        return ax

    def pairplot(
        self,
        df,
        kind="scatter",
        diag_kind="auto",
        upper_kind=None,
        lower_kind=None,
        hue=None,
        output_file=None,
        **kwargs,
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
            "scatter": plt.scatter,
            "kde": sns.kdeplot,
            "hist": sns.distplot,
        }

        palette = kwargs.pop("color", sns.color_palette("pastel"))

        if upper_kind or lower_kind:
            assert upper_kind is not None, "upper_kind cannot be None."
            assert lower_kind is not None, "lower_kind cannot be None."
            assert diag_kind is not "auto", "diag_kind must be either `hist` or `kde`."

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

        return g

    def jointplot(self, x, y, df, kind="scatter", output_file="", **kwargs):
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
        from scipy import stats

        warnings.simplefilter("ignore", UserWarning)

        sns.set(style="ticks", color_codes=True)
        color = kwargs.pop("color", "crimson")

        g = sns.jointplot(x=x, y=y, data=df, kind=kind, color=color, **kwargs).annotate(
            stats.pearsonr
        )

        if output_file:  # pragma: no cover
            g.savefig(os.path.join(IMAGE_DIR, output_file))

        return g

    def histogram(
        self,
        x: list,
        x_train: pd.DataFrame,
        x_test=None,
        hue=None,
        output_file="",
        **kwargs,
    ):
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

        hue : str, optional
            Column to colour points by, by default None

        ouput_file : str
            Output file name for the image including extension (.jpg, .png, etc.)
        """

        import math

        sns.set(style="ticks", color_codes=True)
        sns.set_palette(sns.color_palette("pastel"))

        if hue:
            classes = np.unique(x_train[hue])

        # Make the single plot look pretty
        if len(x) == 1:
            data = x_train[~x_train[x[0]].isnull()]

            if hue:
                for item in classes:
                    g = sns.distplot(
                        data[data[hue] == item][x],
                        label=f"Train Data, {item}",
                        **kwargs,
                    )
            else:
                g = sns.distplot(data[x], label="Train Data", **kwargs)

            if x_test is not None:
                data = x_test[~x_test[x[0]].isnull()]

                if hue:
                    for item in classes:
                        g = sns.distplot(
                            data[data[hue] == item][x],
                            label=f"Test Data, {item}",
                            **kwargs,
                        )
                else:
                    g = sns.distplot(data[x], label="Test Data", **kwargs)

            g.legend(loc="upper right")
            g.set_title(f"Histogram for {x[0].capitalize()}")

        else:
            n_cols = 2
            n_rows = math.ceil(len(x) / n_cols)

            _, ax = plt.subplots(n_rows, n_cols, figsize=(30, 5 * n_cols))

            for ax, col in zip(ax.flat, x):
                g = None
                data = x_train[~x_train[col].isnull()]

                if hue:
                    for item in classes:
                        g = sns.distplot(
                            data[data[hue] == item][col],
                            ax=ax,
                            label=f"Train Data, {item}",
                            **kwargs,
                        )
                else:
                    g = sns.distplot(data[col], ax=ax, label="Train Data", **kwargs)

                if x_test is not None:
                    data = x_test[~x_test[col].isnull()]

                    if hue:
                        for item in classes:
                            g = sns.distplot(
                                data[data[hue] == item][col],
                                ax=ax,
                                label=f"Test Data, {item}",
                                **kwargs,
                            )
                    else:
                        g = sns.distplot(data[col], ax=ax, label="Test Data", **kwargs)

                ax.legend(loc="upper right")

                ax.set_title(f"Histogram for {col.capitalize()}", fontsize=20)

            plt.tight_layout()

        if output_file:  # pragma: no cover
            g.figure.savefig(os.path.join(IMAGE_DIR, output_file))

        return g

    def boxplot(
        self,
        x=None,
        y=None,
        data=None,
        orient="v",
        title="",
        output_file="",
        **boxplot_kwargs,
    ):
        """
        Plots a box plot

        Parameters
        ----------
        x : str, optional
            Column from data, by default None

        y : str, optional
            Column from data, by default None

        data : DataFrame, optional
            Data, by default None

        orient : str, optional
            Orientation of the plot, by default "v"

        title : str, optional
            Title of the plot, by default ''

        output_file : str, optional
            File name, by default ""
        """

        fig = px.box(data, x=x, y=y, orientation=orient, title=title, **boxplot_kwargs)

        if output_file:  # pragma: no cover
            fig.write_image(os.path.join(IMAGE_DIR, output_file))

        return fig

    def violinplot(
        self,
        x=None,
        y=None,
        data=None,
        orient="v",
        title="",
        output_file="",
        **violin_kwargs,
    ):
        """
        Plots a violin plot.

        Parameters
        ----------
        x : str, optional
            Column from data, by default None

        y : str, optional
            Column from data, by default None

        data : DataFrame, optional
            Data, by default None

        orient : str, optional
            Orientation of the plot, by default "v"

        title : str, optional
            Title of the plot, by default ''

        output_file : str, optional
            File name, by default ""
        """

        fig = px.violin(
            data, x=x, y=y, orientation=orient, title=title, **violin_kwargs
        )

        if output_file:  # pragma: no cover
            fig.write_image(os.path.join(IMAGE_DIR, output_file))

        return fig

    def pieplot(
        self,
        values: str,
        names: str,
        data=None,
        textposition=None,
        textinfo=None,
        output_file="",
        **pieplot_kwargs,
    ):
        """
        Plots a pie plot.
        
        Parameters
        ----------
        values : str
            Values of the pie plot.

        names : str
            Labels for the pie plot

        data : DataFrame, optional
            Data, by default None

        textposition : str
            Text position location

        textinfo : str
            Text info

        output_file : str, optional
            File name, by default ""
        """

        fig = px.pie(data, names=names, values=values, **pieplot_kwargs)

        fig.update_traces(textposition=textposition, textinfo=textinfo)

        return fig

    def create_table(self, matrix, index, output_file, **kwargs):
        """
        Creates a table using plotly.
        
        Parameters
        ----------
        matrix : 2d array
            Table values
        """

        from plotly.tools import FigureFactory as FF

        table = FF.create_table(matrix, index=True)

        if output_file:  # pragma: no cover
            table.write_image(os.path.join(IMAGE_DIR, output_file))

        table.show()
