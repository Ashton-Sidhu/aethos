from aethos.visualizations.visualize import VizCreator
import numpy as np
import pandas as pd


class Visualizations(object):
    @property
    def _viz(self):
        return VizCreator()

    @property
    def plot_colors(self):  # pragma: no cover
        """
        Displays all plot colour names
        """

        from IPython.display import IFrame

        IFrame(
            "https://python-graph-gallery.com/wp-content/uploads/100_Color_names_python.png"
        )

    @property
    def plot_colorpalettes(self):  # pragma: no cover
        """
        Displays color palette configuration guide.
        """

        from IPython.display import IFrame

        IFrame("https://seaborn.pydata.org/tutorial/color_palettes.html")

    def raincloud(self, x=None, y=None, output_file="", **params):
        """
        Combines the box plot, scatter plot and split violin plot into one data visualization.
        This is used to offer eyeballed statistical inference, assessment of data distributions (useful to check assumptions),
        and the raw data itself showing outliers and underlying patterns.

        A raincloud is made of:
        1) "Cloud", kernel desity estimate, the half of a violinplot.
        2) "Rain", a stripplot below the cloud
        3) "Umberella", a boxplot
        4) "Thunder", a pointplot connecting the mean of the different categories (if `pointplot` is `True`)

        Useful parameter documentation
        ------------------------------
        https://seaborn.pydata.org/generated/seaborn.boxplot.html

        https://seaborn.pydata.org/generated/seaborn.violinplot.html

        https://seaborn.pydata.org/generated/seaborn.stripplot.html

        Parameters
        ----------
        x : str
            X axis data, reference by column name, any data

        y : str
            Y axis data, reference by column name, measurable data (numeric)
            by default target

        hue : Iterable, np.array, or dataframe column name if 'data' is specified
            Second categorical data. Use it to obtain different clouds and rainpoints

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        orient : str                  
            vertical if "v" (default), horizontal if "h"

        width_viol : float            
            width of the cloud

        width_box : float             
            width of the boxplot

        palette : list or dict        
            Colours to use for the different levels of categorical variables

        bw : str or float
            Either the name of a reference rule or the scale factor to use when computing the kernel bandwidth,
            by default "scott"

        linewidth : float             
            width of the lines

        cut : float
            Distance, in units of bandwidth size, to extend the density past the extreme datapoints.
            Set to 0 to limit the violin range within the range of the observed data,
            by default 2

        scale : str
            The method used to scale the width of each violin.
            If area, each violin will have the same area.
            If count, the width of the violins will be scaled by the number of observations in that bin.
            If width, each violin will have the same width.
            By default "area"

        jitter : float, True/1
            Amount of jitter (only along the categorical axis) to apply.
            This can be useful when you have many points and they overlap,
            so that it is easier to see the distribution. You can specify the amount of jitter (half the width of the uniform random variable support),
            or just use True for a good default.

        move : float                  
            adjust rain position to the x-axis (default value 0.)

        offset : float                
            adjust cloud position to the x-axis

        color : matplotlib color
            Color for all of the elements, or seed for a gradient palette.

        ax : matplotlib axes
            Axes object to draw the plot onto, otherwise uses the current Axes.

        figsize : (int, int)    
            size of the visualization, ex (12, 5)

        pointplot : bool   
            line that connects the means of all categories, by default False

        dodge : bool 
            When hue nesting is used, whether elements should be shifted along the categorical axis.

        Source: https://micahallen.org/2018/03/15/introducing-raincloud-plots/
        
        Examples
        --------
        >>> data.raincloud('col1') # Will plot col1 values on the x axis and your target variable values on the y axis
        >>> data.raincloud('col1', 'col2') # Will plot col1 on the x and col2 on the y axis
        >>> data.raincloud('col1', 'col2', output_file='raincloud.png')
        """

        if y is None:
            y = self.target

        fig = self._viz.raincloud(y, x, self.x_train, output_file=output_file, **params)

        return fig

    def barplot(
        self,
        x: str,
        y=None,
        method=None,
        asc=None,
        orient="v",
        title="",
        output_file="",
        **barplot_kwargs,
    ):
        """
        Plots a bar plot for the given columns provided using Plotly.

        If `groupby` is provided, method must be provided for example you may want to plot Age against survival rate,
        so you would want to `groupby` Age and then find the `mean` as the method.

        For a list of group by methods please checkout the following pandas link:
        https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#computations-descriptive-stats

        For a list of possible arguments for the bar plot please checkout the following links:
        https://plot.ly/python-api-reference/generated/plotly.express.bar.html

        Parameters
        ----------
        x : str
            Column name for the x axis.

        y : str, optional
            Column(s) you would like to see plotted against the x_col

        method : str
            Method to aggregate groupy data
            Examples: min, max, mean, etc., optional
            by default None

        asc : bool
            To sort values in ascending order, False for descending

        orient : str (default 'v')
            One of 'h' for horizontal or 'v' for vertical

        title : str
            The figure title.

        color : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to assign color to marks.

        hover_name : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like appear in bold in the hover tooltip.

        hover_data : list of str or int, or Series or array-like
            Either names of columns in data_frame, or pandas Series, or array_like objects Values from these columns appear as extra data in the hover tooltip.

        custom_data : list of str or int, or Series or array-like
            Either names of columns in data_frame, or pandas Series, or array_like objects
            Values from these columns are extra data, to be used in widgets or Dash callbacks for example.
            This data is not user-visible but is included in events emitted by the figure (lasso selection etc.)

        text : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like appear in the figure as text labels.

        animation_frame : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to assign marks to animation frames.

        animation_group : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to provide object-constancy across animation frames: rows with matching `animation_group`s will be treated as if they describe the same object in each frame.

        labels : dict with str keys and str values (default {})
            By default, column names are used in the figure for axis titles, legend entries and hovers.
            This parameter allows this to be overridden.
            The keys of this dict should correspond to column names, and the values should correspond to the desired label to be displayed.

        color_discrete_sequence : list of str
            Strings should define valid CSS-colors.
            When color is set and the values in the corresponding column are not numeric, values in that column are assigned colors by cycling through color_discrete_sequence in the order described in category_orders, unless the value of color is a key in color_discrete_map.
            Various useful color sequences are available in the plotly.express.colors submodules, specifically plotly.express.colors.qualitative.

        color_discrete_map : dict with str keys and str values (default {})
            String values should define valid CSS-colors Used to override color_discrete_sequence to assign a specific colors to marks corresponding with specific values.
            Keys in color_discrete_map should be values in the column denoted by color.

        color_continuous_scale : list of str
            Strings should define valid CSS-colors. 
            This list is used to build a continuous color scale when the column denoted by color contains numeric data.
            Various useful color scales are available in the plotly.express.colors submodules, specifically plotly.express.colors.sequential, plotly.express.colors.diverging and plotly.express.colors.cyclical.

        opacity : float
            Value between 0 and 1. Sets the opacity for markers.

        barmode : str (default 'relative')
            One of 'group', 'overlay' or 'relative'
            In 'relative' mode, bars are stacked above zero for positive values and below zero for negative values.
            In 'overlay' mode, bars are drawn on top of one another.
            In 'group' mode, bars are placed beside each other.

        width : int (default None)
            The figure width in pixels.

        height : int (default 600)
            The figure height in pixels.

        output_file : str, optional
            Output html file name for image

        Returns
        -------
        Plotly Figure
            Plotly Figure Object of Bar Plot

        Examples
        --------
        >>> data.barplot(x='x', y='y')
        >>> data.barplot(x='x', method='mean')
        >>> data.barplot(x='x', y='y', method='max', orient='h')
        """

        if orient == "h":
            x, y = y, x

        fig = self._viz.barplot(
            x,
            y,
            self.x_train.copy(),
            method=method,
            asc=asc,
            output_file=output_file,
            orientation=orient,
            title=title,
            **barplot_kwargs,
        )

        return fig

    def scatterplot(
        self,
        x=None,
        y=None,
        z=None,
        color=None,
        title="Scatter Plot",
        output_file="",
        **scatterplot_kwargs,
    ):
        """
        Plots a scatterplot for the given x and y columns provided using Plotly Express.

        For a list of possible scatterplot_kwargs for 2 dimensional data please check out the following links:

            https://plot.ly/python-api-reference/generated/plotly.express.scatter.html

        For more information on key word arguments for 3d data, please check them out here:

            https://www.plotly.express/plotly_express/#plotly_express.scatter_3d
        
        Parameters
        ----------
        x : str
            X column name

        y : str
            Y column name

        z : str
            Z column name, 

        color : str, optional
            Category to group your data, by default None

        title : str, optional
            Title of the plot, by default 'Scatter Plot'

        output_file : str, optional
            Output html file name for image

        symbol : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to assign symbols to marks.

        size : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to assign mark sizes.

        hover_name : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like appear in bold in the hover tooltip.

        hover_data : list of str or int, or Series or array-like, or dict)
            Either a list of names of columns in data_frame, or pandas Series, or array_like objects or a dict with column names as keys, with values True (for default formatting) False (in order to remove this column from hover information), or a formatting string, for example ‘:.3f’ or ‘|%a’ or list-like data to appear in the hover tooltip or tuples with a bool or formatting string as first element, and list-like data to appear in hover as second element Values from these columns appear as extra data in the hover tooltip.

        custom_data : list of str or int, or Series or array-like
            Either names of columns in data_frame, or pandas Series, or array_like objects Values from these columns are extra data, to be used in widgets or Dash callbacks for example. This data is not user-visible but is included in events emitted by the figure (lasso selection etc.)

        text : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like appear in the figure as text labels.

        facet_row : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to assign marks to facetted subplots in the vertical direction.

        facet_col : str or int or Series or array-like)
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to assign marks to facetted subplots in the horizontal direction.

        facet_col_wrap : int
            Maximum number of facet columns. Wraps the column variable at this width, so that the column facets span multiple rows. Ignored if 0, and forced to 0 if facet_row or a marginal is set.

        error_x : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to size x-axis error bars. If error_x_minus is None, error bars will be symmetrical, otherwise error_x is used for the positive direction only.

        error_x_minus : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to size x-axis error bars in the negative direction. Ignored if error_x is None.

        error_y : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to size y-axis error bars. If error_y_minus is None, error bars will be symmetrical, otherwise error_y is used for the positive direction only.

        error_y_minus : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to size y-axis error bars in the negative direction. Ignored if error_y is None.

        animation_frame : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to assign marks to animation frames.

        animation_group : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object. Values from this column or array_like are used to provide object-constancy across animation frames: rows with matching `animation_group`s will be treated as if they describe the same object in each frame.

        labels : dict with str keys and str values (default {})
            By default, column names are used in the figure for axis titles, legend entries and hovers. This parameter allows this to be overridden. The keys of this dict should correspond to column names, and the values should correspond to the desired label to be displayed.

        color_discrete_sequence : list of str
            Strings should define valid CSS-colors. When color is set and the values in the corresponding column are not numeric, values in that column are assigned colors by cycling through color_discrete_sequence in the order described in category_orders, unless the value of color is a key in color_discrete_map. Various useful color sequences are available in the plotly.express.colors submodules, specifically plotly.express.colors.qualitative.

        color_discrete_map : dict with str keys and str values (default {})
            String values should define valid CSS-colors Used to override color_discrete_sequence to assign a specific colors to marks corresponding with specific values. Keys in color_discrete_map should be values in the column denoted by color.

        color_continuous_scale : list of str
            Strings should define valid CSS-colors This list is used to build a continuous color scale when the column denoted by color contains numeric data. Various useful color scales are available in the plotly.express.colors submodules, specifically plotly.express.colors.sequential, plotly.express.colors.diverging and plotly.express.colors.cyclical.

        range_color : list of two numbers
            If provided, overrides auto-scaling on the continuous color scale.

        color_continuous_midpoint : number (default None)
            If set, computes the bounds of the continuous color scale to have the desired midpoint. Setting this value is recommended when using plotly.express.colors.diverging color scales as the inputs to color_continuous_scale.

        opacity : float
            Value between 0 and 1. Sets the opacity for markers.

        size_max : int (default 20)
            Set the maximum mark size when using size.

        marginal_x : str
            One of 'rug', 'box', 'violin', or 'histogram'. If set, a horizontal subplot is drawn above the main plot, visualizing the x-distribution.

        marginal_y : str
            One of 'rug', 'box', 'violin', or 'histogram'. If set, a vertical subplot is drawn to the right of the main plot, visualizing the y-distribution.

        trendline : str
            One of 'ols' or 'lowess'. If 'ols', an Ordinary Least Squares regression line will be drawn for each discrete-color/symbol group. If 'lowess’, a Locally Weighted Scatterplot Smoothing line will be drawn for each discrete-color/symbol group.

        trendline_color_override : str)
            Valid CSS color. If provided, and if trendline is set, all trendlines will be drawn in this color.

        log_x : boolean (default False)
            If True, the x-axis is log-scaled in cartesian coordinates.

        log_y : boolean (default False)
            If True, the y-axis is log-scaled in cartesian coordinates.

        range_x : list of two numbers
            If provided, overrides auto-scaling on the x-axis in cartesian coordinates.

        range_y : list of two numbers
            If provided, overrides auto-scaling on the y-axis in cartesian coordinates.

        width : int (default None)
            The figure width in pixels.

        height : int (default None)
            The figure height in pixels.

        Returns
        -------
        Plotly Figure
            Plotly Figure Object of Scatter Plot

        Examples
        --------
        >>> data.scatterplot(x='x', y='y') #2d
        >>> data.scatterplot(x='x', y='y', z='z') #3d
        >>> data.scatterplot(x='x', y='y', z='z', output_file='scatt')
        """

        fig = self._viz.scatterplot(
            x,
            y,
            z=z,
            data=self.x_train.copy(),
            title=title,
            color=color,
            output_file=output_file,
            **scatterplot_kwargs,
        )

        return fig

    def lineplot(
        self,
        x: str,
        y: str,
        z=None,
        color=None,
        title="Line Plot",
        output_file="",
        **lineplot_kwargs,
    ):
        """
        Plots a lineplot for the given x and y columns provided using Plotly Express.

        For a list of possible lineplot_kwargs please check out the following links:

        For 2d:

            https://plot.ly/python-api-reference/generated/plotly.express.line.html#plotly.express.line

        For 3d:

            https://plot.ly/python-api-reference/generated/plotly.express.line_3d.html#plotly.express.line_3d
        
        Parameters
        ----------
        x : str
            X column name

        y : str
            Column name to plot on the y axis.

        z: str
            Column name to plot on the z axis.

        title : str, optional
            Title of the plot, by default 'Line Plot'

        color : str
            Category column to draw multiple line plots of

        output_file : str, optional
            Output html file name for image

        text : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like appear in the figure as text labels.

        facet_row : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to assign marks to facetted subplots in the vertical direction.

        facet_col : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to assign marks to facetted subplots in the horizontal direction.

        error_x : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to size x-axis error bars.
            If error_x_minus is None, error bars will be symmetrical, otherwise error_x is used for the positive direction only.

        error_x_minus : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to size x-axis error bars in the negative direction.
            Ignored if error_x is None.

        error_y : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to size y-axis error bars.
            If error_y_minus is None, error bars will be symmetrical, otherwise error_y is used for the positive direction only.

        error_y_minus : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to size y-axis error bars in the negative direction.
            Ignored if error_y is None.

        animation_frame : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to assign marks to animation frames.

        animation_group : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to provide object-constancy across animation frames: rows with matching `animation_group`s will be treated as if they describe the same object in each frame.

        labels : dict with str keys and str values (default {})
            By default, column names are used in the figure for axis titles, legend entries and hovers. 
            his parameter allows this to be overridden. The keys of this dict should correspond to column names, and the values should correspond to the desired label to be displayed.

        color_discrete_sequence : list of str
            Strings should define valid CSS-colors. 
            When color is set and the values in the corresponding column are not numeric, values in that column are assigned colors by cycling through color_discrete_sequence in the order described in category_orders, unless the value of color is a key in color_discrete_map.
            Various useful color sequences are available in the plotly.express.colors submodules, specifically plotly.express.colors.qualitative.

        color_discrete_map : dict with str keys and str values (default {})
            String values should define valid CSS-colors Used to override color_discrete_sequence to assign a specific colors to marks corresponding with specific values.
            Keys in color_discrete_map should be values in the column denoted by color.

        Returns
        -------
        Plotly Figure
            Plotly Figure Object of Line Plot

        Examples
        --------
        >>> data.line_plot(x='x', y='y')
        >>> data.line_plot(x='x', y='y', output_file='line')
        """

        fig = self._viz.lineplot(
            x,
            y,
            z,
            self.x_train.copy(),
            color=color,
            title=title,
            output_file=output_file,
            **lineplot_kwargs,
        )

        return fig

    def pairplot(
        self,
        cols=[],
        kind="scatter",
        diag_kind="auto",
        upper_kind=None,
        lower_kind=None,
        hue=None,
        output_file="",
        **kwargs,
    ):
        """
        Plots pairplots of the variables from the training data.

        If hue is not provided and a target variable is set, the data will separated and highlighted by the classes in that column.

        For more info and kwargs on pair plots, please see: https://seaborn.pydata.org/generated/seaborn.pairplot.html
        
        Parameters
        ----------
        cols : list
            Columns to view pairplot of.

        kind : str {'scatter', 'reg'}, optional
            Type of plot for off-diag plots, by default 'scatter'

        diag_kind : str {'auto', 'hist', 'kde'}, optional
            Type of plot for diagonal, by default 'auto'

        upper_kind : str {'scatter', 'kde'}, optional
            Type of plot for upper triangle of pair plot, by default None

        lower_kind : str {'scatter', 'kde'}, optional
            Type of plot for lower triangle of pair plot, by default None

        hue : str, optional
            Column to colour points by, by default None

        {x, y}_vars : lists of variable names, optional
            Variables within data to use separately for the rows and columns of the figure; i.e. to make a non-square plot.

        palette : dict or seaborn color palette
            Set of colors for mapping the hue variable. If a dict, keys should be values in the hue variable.

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        Examples
        --------
        >>> data.pairplot(kind='kde')
        >>> data.pairplot(kind='kde', output_file='pair.png')
        """

        data = self.x_train if not cols else self.x_train[cols]

        fig = self._viz.pairplot(
            data,
            kind=kind,
            diag_kind=diag_kind,
            upper_kind=upper_kind,
            lower_kind=lower_kind,
            hue=hue,
            output_file=output_file,
            **kwargs,
        )

        return fig

    def jointplot(self, x: str, y: str, kind="scatter", output_file="", **kwargs):
        """
        Plots joint plots of 2 different variables.

        Scatter ('scatter'): Scatter plot and histograms of x and y.

        Regression ('reg'): Scatter plot, with regression line and histograms with kernel density fits.

        Residuals ('resid'): Scatter plot of residuals and histograms of residuals.

        Kernel Density Estimates ('kde'): Density estimate plot and histograms.

        Hex ('hex'): Replaces scatterplot with joint histogram using hexagonal bins and histograms on the axes.

        For more info and kwargs for joint plots, see https://seaborn.pydata.org/generated/seaborn.jointplot.html
        
        Parameters
        ----------
        x : str
            X axis column

        y : str
            y axis column

        kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
            Kind of plot to draw, by default 'scatter'

        color : matplotlib color, optional
            Color used for the plot elements.

        dropna : bool, optional
            If True, remove observations that are missing from x and y.

        {x, y}lim : two-tuples, optional
            Axis limits to set before plotting.

        {joint, marginal, annot}_kws : dicts, optional
            Additional keyword arguments for the plot components.

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        Examples
        --------
        >>> data.jointplot(x='x', y='y', kind='kde', color='crimson')
        >>> data.jointplot(x='x', y='y', kind='kde', color='crimson', output_file='pair.png')
        """

        fig = self._viz.jointplot(
            x=x, y=y, df=self.x_train, kind=kind, output_file=output_file, **kwargs
        )

        return fig

    def histogram(self, *x, hue=None, plot_test=False, output_file="", **kwargs):
        """
        Plots a histogram of the given column(s).

        If no columns are provided, histograms are plotted for all numeric columns

        For more histogram key word arguments, please see https://seaborn.pydata.org/generated/seaborn.distplot.html

        Parameters
        ----------
        x: str or str(s)
            Column(s) to plot histograms for.

        hue : str, optional
            Column to colour points by, by default None

        plot_test : bool, optional
            True to plot distribution of the test data for the same variable

        bins : argument for matplotlib hist(), or None, optional
            Specification of hist bins, or None to use Freedman-Diaconis rule.

        hist : bool, optional
            Whether to plot a (normed) histogram.

        kde : bool, optional
            Whether to plot a gaussian kernel density estimate.

        rug : bool, optional
            Whether to draw a rugplot on the support axis.

        fit : random variable object, optional
            An object with fit method, returning a tuple that can be passed to a pdf method a positional arguments following an grid of values to evaluate the pdf on.

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        Examples
        --------
        >>> data.histogram()
        >>> data.histogram('col1')
        >>> data.histogram('col1', 'col2', hue='col3', plot_test=True)
        >>> data.histogram('col1', kde=False)
        >>> data.histogram('col1', 'col2', hist=False)
        >>> data.histogram('col1', kde=False, fit=stat.normal)
        >>> data.histogram('col1', kde=False, output_file='hist.png')
        """

        x_test = self.x_test if plot_test else None
        columns = (
            list(x)
            if x
            else list(self.x_train.select_dtypes(include=[np.number]).columns)
        )

        self._viz.histogram(
            columns,
            x_train=self.x_train,
            x_test=x_test,
            hue=hue,
            output_file=output_file,
            **kwargs,
        )

    def plot_dim_reduction(
        self, col: str, dim=2, algo="tsne", output_file="", **kwargs
    ):
        """
        Reduce the dimensions of your data and then view similarly grouped data points (clusters)

        For 2d plotting options, see:
        
            https://plot.ly/python-api-reference/generated/plotly.express.scatter.html

        For 3d plotting options, see:

            https://www.plotly.express/plotly_express/#plotly_express.scatter_3d
        
        Parameters
        ----------
        col : str
            Column name of the labels/data points to highlight in the plot
            
        dim : int {2, 3}
            Dimensions of the plot to show, either 2d or 3d, by default 2

        algo : str {'tsne', 'lle', 'pca', 'tsvd'}, optional
            Algorithm to reduce the dimensions by, by default 'tsne'

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        kwargs: 
            See plotting options

        Returns
        -------
        Plotly Figure
            Plotly Figure Object of Scatter Plot

        Examples
        --------
        >>> data.plot_dim_reduction('cluster_labels', dim=3)
        """

        from sklearn.manifold import LocallyLinearEmbedding, TSNE
        from sklearn.decomposition import PCA, TruncatedSVD

        if dim != 2 and dim != 3:
            raise ValueError("Dimension must be either 2d (2) or 3d (3)")

        algorithms = {
            "tsne": TSNE(n_components=dim, random_state=42,),
            "lle": LocallyLinearEmbedding(n_components=dim, random_state=42,),
            "pca": PCA(n_components=dim, random_state=42,),
            "tsvd": TruncatedSVD(n_components=dim, random_state=42,),
        }

        reducer = algorithms[algo]
        reduced_df = pd.DataFrame(reducer.fit_transform(self.x_train.drop(col, axis=1)))
        reduced_df.columns = map(str, reduced_df.columns)
        reduced_df[col] = self.x_train[col]
        reduced_df[col] = reduced_df[col].astype(str)

        if dim == 2:
            fig = self._viz.scatterplot(
                "0", "1", data=reduced_df, color=col, output_file=output_file, **kwargs,
            )
        else:
            fig = self._viz.scatterplot(
                "0",
                "1",
                "2",
                data=reduced_df,
                color=col,
                output_file=output_file,
                **kwargs,
            )

        return fig

    def boxplot(self, x=None, y=None, color=None, title="", output_file="", **kwargs):
        """
        Plots a box plot for the given x and y columns.

        For more info and kwargs for box plots, see https://plot.ly/python-api-reference/generated/plotly.express.box.html#plotly.express.box
        and https://plot.ly/python/box-plots/ 
        
        Parameters
        ----------
        x : str
            X axis column

        y : str
            y axis column

        color : str, optional
            Column name to add a dimension by color.

        orient : str, optional
            Orientation of graph, 'h' for horizontal
            'v' for vertical, by default 'v',

        points : str, bool {'outlier', 'suspectedoutliers', 'all', False}
            One of 'outliers', 'suspectedoutliers', 'all', or False.
            If 'outliers', only the sample points lying outside the whiskers are shown.
            If 'suspectedoutliers', all outlier points are shown and those less than 4*Q1-3*Q3 or greater than 4*Q3-3*Q1 are highlighted with the marker’s 'outliercolor'.
            If 'outliers', only the sample points lying outside the whiskers are shown.
            If 'all', all sample points are shown. If False, no sample points are shown and the whiskers extend to the full range of the sample.

        notched : bool, optional
            If True, boxes are drawn with notches, by default False.

        title : str, optional
            Title of the plot, by default "".

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        Returns
        -------
        Plotly Figure
            Plotly Figure Object of Box Plot

        Examples
        --------
        >>> data.boxplot(y='y', color='z')
        >>> data.boxplot(x='x', y='y', color='z', points='all')
        >>> data.boxplot(x='x', y='y', output_file='pair.png')
        """

        assert (
            x is not None or y is not None
        ), "An x column or a y column must be provided."

        fig = self._viz.boxplot(
            x=x,
            y=y,
            data=self.x_train.copy(),
            color=color,
            title=title,
            output_file=output_file,
            **kwargs,
        )

        return fig

    def violinplot(
        self, x=None, y=None, color=None, title="", output_file="", **kwargs
    ):
        """
        Plots a violin plot for the given x and y columns.

        For more info and kwargs for violin plots, see https://plot.ly/python-api-reference/generated/plotly.express.violin.html#plotly.express.violin 
        and https://plot.ly/python/violin/
        
        Parameters
        ----------
        x : str
            X axis column

        y : str
            y axis column

        color : str, optional
            Column name to add a dimension by color.

        orient : str, optional
            Orientation of graph, 'h' for horizontal
            'v' for vertical, by default 'v',

        points : str, bool {'outlier', 'suspectedoutliers', 'all', False}
            One of 'outliers', 'suspectedoutliers', 'all', or False.
            If 'outliers', only the sample points lying outside the whiskers are shown.
            If 'suspectedoutliers', all outlier points are shown and those less than 4*Q1-3*Q3 or greater than 4*Q3-3*Q1 are highlighted with the marker’s 'outliercolor'.
            If 'outliers', only the sample points lying outside the whiskers are shown.
            If 'all', all sample points are shown. If False, no sample points are shown and the whiskers extend to the full range of the sample.

        violinmode : str {'group', 'overlay'}
            In 'overlay' mode, violins are on drawn top of one another.
            In 'group' mode, violins are placed beside each other.

        box : bool, optional
            If True, boxes are drawn inside the violins.

        title : str, optional
            Title of the plot, by default "".

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        Returns
        -------
        Plotly Figure
            Plotly Figure Object of Violin Plot

        Examples
        --------
        >>> data.violinplot(y='y', color='z', box=True)
        >>> data.violinplot(x='x', y='y', color='z', points='all')
        >>> data.violinplot(x='x', y='y', violinmode='overlay', output_file='pair.png')
        """

        assert (
            x is not None or y is not None
        ), "An x column or a y column must be provided."

        fig = self._viz.violinplot(
            x=x,
            y=y,
            data=self.x_train.copy(),
            color=color,
            title=title,
            output_file=output_file,
            **kwargs,
        )

        return fig

    def pieplot(
        self,
        values: str,
        names: str,
        title="",
        textposition="inside",
        textinfo="percent",
        output_file="",
        **pieplot_kwargs,
    ):
        """
        Plots a Pie plot of a given column.

        For more information regarding pie plots please see the following links: https://plot.ly/python/pie-charts/#customizing-a-pie-chart-created-with-pxpie
        and https://plot.ly/python-api-reference/generated/plotly.express.pie.html#plotly.express.pie.
        
        Parameters
        ----------
        values : str
            Column in the DataFrame.
            Values from this column or array_like are used to set values associated to sectors.

        names : str
            Column in the DataFrame.
            Values from this column or array_like are used as labels for sectors.

        title : str, optional
            The figure title, by default ''

        textposition : {'inside', 'outside'}, optional
            Position the text in the plot, by default 'inside'

        textinfo : str, optional
            textinfo' can take any of the following values, joined with a '+':
                'label' - displays the label on the segment
                'text' - displays the text on the segment (this can be set separately to the label)
                'value' - displays the value passed into the trace
                'percent' - displayed the computer percentage
        , by default 'percent'

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        color : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like are used to assign color to marks.

        color_discrete_sequence : list of str
            Strings should define valid CSS-colors.
            When color is set and the values in the corresponding column are not numeric, values in that column are assigned colors by cycling through color_discrete_sequence in the order described in category_orders, unless the value of color is a key in color_discrete_map.
            Various useful color sequences are available in the plotly.express.colors submodules, specifically plotly.express.colors.qualitative.

        color_discrete_map : dict with str keys and str values (default {})
            String values should define valid CSS-colors Used to override color_discrete_sequence to assign a specific colors to marks corresponding with specific values.
            Keys in color_discrete_map should be values in the column denoted by color.

        hover_name : str or int or Series or array-like
            Either a name of a column in data_frame, or a pandas Series or array_like object.
            Values from this column or array_like appear in bold in the hover tooltip.

        hover_data : list of str or int, or Series or array-like
            Either names of columns in data_frame, or pandas Series, or array_like objects.
            Values from these columns appear as extra data in the hover tooltip.

        custom_data : list of str or int, or Series or array-like
            Either names of columns in data_frame, or pandas Series, or array_like objects Values from these columns are extra data, to be used in widgets or Dash callbacks for example.
            This data is not user-visible but is included in events emitted by the figure (lasso selection etc.)

        labels : dict with str keys and str values (default {})
            By default, column names are used in the figure for axis titles, legend entries and hovers.
            This parameter allows this to be overridden.
            The keys of this dict should correspond to column names, and the values should correspond to the desired label to be displayed.

        width : int (default None)
            The figure width in pixels.

        height : int (default 600)
            The figure height in pixels.

        opacity : float
            Value between 0 and 1.
            Sets the opacity for markers.

        hole : float
            Value between 0 and 1.
            Sets the size of the hole in the middle of the pie chart.
        
        Returns
        -------
        Plotly Figure
            Plotly Figure Object of Pie Chart

        Examples
        --------
        >>> data.pieplot(val_column, name_column)
        """

        fig = self._viz.pieplot(
            values,
            names,
            data=self.x_train.copy(),
            textposition=textposition,
            textinfo=textinfo,
            title=title,
            output_file=output_file,
            **pieplot_kwargs,
        )

        return fig
