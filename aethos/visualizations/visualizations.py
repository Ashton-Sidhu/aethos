from aethos.visualizations.visualize import *


class Visualizations(object):
    def __init__(self, data):
        self.data = data

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
            by default target_field

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
            y = self.target_field

        raincloud(y, x, self.data, output_file=output_file, **params)

        if output_file and self.report:  # pragma: no cover
            self.report.write_image(output_file)

    def barplot(
        self,
        x: str,
        y=None,
        method=None,
        orient="v",
        stacked=False,
        output_file="",
        **barplot_kwargs,
    ):
        """
        Plots a bar plot for the given columns provided using Bokeh.

        If `groupby` is provided, method must be provided for example you may want to plot Age against survival rate,
        so you would want to `groupby` Age and then find the `mean` as the method.

        For a list of group by methods please checkout the following pandas link:
        https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#computations-descriptive-stats

        For a list of possible arguments for the bar plot please checkout the following links:
        https://github.com/PatrikHlobil/Pandas-Bokeh#barplot and

        https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.vbar or

        https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.hbar for horizontal
        
        Parameters
        ----------
        x : str
            Column name for the x axis.

        y : list
            Columns you would like to see plotted against the x_col

        method : str
            Method to aggregate groupy data
            Examples: min, max, mean, etc., optional
            by default None

        orient : str, optional
            Orientation of graph, 'h' for horizontal
            'v' for vertical, by default 'v',

        stacked : bool
            Whether to stack the different columns resulting in a stacked bar chart,
            by default False

        output_file : str, optional
            Output html file name for image

        Examples
        --------
        >>> data.barplot(x='x', y=['y', 'z'], method='mean')
        >>> data.barplot(x='x', y=['y', 'z'], method='max', orient='h')
        """

        barplot(
            x,
            y,
            self.data,
            method=method,
            orient=orient,
            stacked=stacked,
            output_file=output_file,
            **barplot_kwargs,
        )

        if output_file and self.report:  # pragma: no cover
            self.report.write_image(output_file)

    def scatterplot(
        self,
        x=None,
        y=None,
        z=None,
        category=None,
        title="Scatter Plot",
        size=8,
        output_file="",
        **scatterplot_kwargs,
    ):
        """
        Plots a scatterplot for the given x and y columns provided using Bokeh.

        For a list of possible scatterplot_kwargs for 2 dimensional data please check out the following links:

        https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.scatter

        https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#userguide-styling-line-properties 

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

        category : str, optional
            Category to group your data, by default None

        title : str, optional
            Title of the plot, by default 'Scatter Plot'

        size : int or str, optional
            Size of the circle, can either be a number
            or a column name to scale the size, by default 8

        output_file : str, optional
            Output html file name for image

        **scatterplot_kwargs : optional
            See above links for list of possible scatterplot options.

        Examples
        --------
        >>> data.scatterplot(x='x', y='y') #2d
        >>> data.scatterplot(x='x', y='y', z='z') #3d
        >>> data.scatterplot(x='x', y='y', z='z', output_file='scatt')
        """

        scatterplot(
            x,
            y,
            z=z,
            data=self.data,
            title=title,
            category=category,
            size=size,
            output_file=output_file,
            **scatterplot_kwargs,
        )

        if output_file and self.report:  # pragma: no cover
            self.report.write_image(output_file)

    def lineplot(
        self, x: str, y: list, title="Line Plot", output_file="", **lineplot_kwargs
    ):
        """
        Plots a lineplot for the given x and y columns provided using Bokeh.

        For a list of possible lineplot_kwargs please check out the following links:

        https://github.com/PatrikHlobil/Pandas-Bokeh#lineplot

        https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.line 
        
        Parameters
        ----------
        x : str
            X column name

        y : list
            Column names to plot on the y axis.

        title : str, optional
            Title of the plot, by default 'Line Plot'

        output_file : str, optional
            Output html file name for image

        color : str, optional
            Define a single color for the plot

        colormap : list or Bokeh color palette, optional
            Can be used to specify multiple colors to plot.
            Can be either a list of colors or the name of a Bokeh color palette : https://bokeh.pydata.org/en/latest/docs/reference/palettes.html

        rangetool : bool, optional
            If true, will enable a scrolling range tool.

        xlabel : str, optional
            Name of the x axis

        ylabel : str, optional
            Name of the y axis

        xticks : list, optional
            Explicitly set ticks on x-axis

        yticks : list, optional
            Explicitly set ticks on y-axis

        xlim : tuple (int or float), optional
            Set visible range on x axis

        ylim : tuple (int or float), optional
            Set visible range on y axis.

        **lineplot_kwargs : optional
            For a list of possible keyword arguments for line plot please see https://github.com/PatrikHlobil/Pandas-Bokeh#lineplot
            and https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.line

        Examples
        --------
        >>> data.line_plot(x='x', y='y')
        >>> data.line_plot(x='x', y='y', output_file='line')
        """

        lineplot(
            x, y, self.data, title=title, output_file=output_file, **lineplot_kwargs,
        )

        if output_file and self.report:  # pragma: no cover
            self.report.write_image(output_file)

    def correlation_matrix(
        self, data_labels=False, hide_mirror=False, output_file="", **kwargs
    ):
        """
        Plots a correlation matrix of all the numerical variables.

        For more information on possible kwargs please see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
        
        Parameters
        ----------
        data_labels : bool, optional
            True to display the correlation values in the plot, by default False

        hide_mirror : bool, optional
            Whether to display the mirroring half of the correlation plot, by default False

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        Examples
        --------
        >>> data.correlation_matrix(data_labels=True)
        >>> data.correlation_matrix(data_labels=True, output_file='corr.png')
        """

        correlation_matrix(
            self.data,
            data_labels=data_labels,
            hide_mirror=hide_mirror,
            output_file=output_file,
            **kwargs,
        )

        if output_file and self.report:  # pragma: no cover
            self.report.write_image(output_file)

    def pairplot(
        self, kind="scatter", diag_kind="auto", hue=None, output_file="", **kwargs
    ):
        """
        Plots pairplots of the variables from the training data.

        If hue is not provided and a target variable is set, the data will separated and highlighted by the classes in that column.

        For more info and kwargs on pair plots, please see: https://seaborn.pydata.org/generated/seaborn.pairplot.html
        
        Parameters
        ----------
        df : DataFrame
                Data

        kind : {'scatter', 'reg'}, optional
            Type of plot for off-diag plots, by default 'scatter'

        diag_kind : {'auto', 'hist', 'kde'}, optional
            Type of plot for diagonal, by default 'auto'

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

        if self.target_field and not hue:
            hue = self.target_field
        elif not self.target_field and hue:
            hue = hue

        pairplot(
            self.data,
            kind=kind,
            diag_kind=diag_kind,
            hue=hue,
            output_file=output_file,
            **kwargs,
        )

        if output_file and self.report:  # pragma: no cover
            self.report.write_image(output_file)

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

        jointplot(x=x, y=y, df=self.data, kind=kind, output_file=output_file, **kwargs)

        if output_file and self.report:  # pragma: no cover
            self.report.write_image(output_file)

    def histogram(self, *x, output_file="", **kwargs):
        """
        Plots a histogram of the given column(s).

        For more histogram key word arguments, please see https://seaborn.pydata.org/generated/seaborn.distplot.html

        Parameters
        ----------
        x: str or str(s)
            Column(s) to plot histograms for.

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
        >>> data.histogram('col1')
        >>> data.histogram('col1', 'col2')
        >>> data.histogram('col1', kde=False)
        >>> data.histogram('col1', 'col2', hist=False)
        >>> data.histogram('col1', kde=False, fit=stat.normal)
        >>> data.histogram('col1', kde=False, output_file='hist.png')
        """

        histogram(list(x), data=self.data, output_file=output_file, **kwargs)

        if output_file and self.report:  # pragma: no cover
            self.report.write_image(output_file)
