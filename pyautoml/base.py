import copy

import pandas as pd
from IPython import get_ipython
from IPython.display import display
from pandas_summary import DataFrameSummary

from pyautoml.data.data import Data
from pyautoml.util import _function_input_validation, split_data
from pyautoml.visualizations.visualize import *


class MethodBase(object):

    def __init__(self, **kwargs):

        data = kwargs.pop('data')
        train_data = kwargs.pop('train_data')
        test_data = kwargs.pop('test_data')
        use_full_data = kwargs.pop('use_full_data')
        target_field = kwargs.pop('target_field')
        report_name = kwargs.pop('report_name')
        test_split_percentage = kwargs.pop('test_split_percentange')

        if not _function_input_validation(data, train_data, test_data):
            raise ValueError("Error initialzing constructor, please provide one of either data or train_data and test_data, not both.")

        self.data_properties = Data(data, train_data, test_data, use_full_data=use_full_data, target_field=target_field, report_name=report_name)

        if data is not None and not use_full_data:
            # Generate train set and test set.
            # NOTE: Test if setting data to `None` is a good idea.
            self.data_properties.train_data, self.data_properties.test_data = split_data(self.data_properties.data, test_split_percentage)
            self.data_properties.data = None
            self.data_properties.train_data.reset_index(drop=True, inplace=True)
            self.data_properties.test_data.reset_index(drop=True, inplace=True)

        if self.data_properties.report is None:
            self.report = None
        else:
            self.report = self.data_properties.report        
            
    def __repr__(self):

        shell = get_ipython().__class__.__name__

        if shell == 'ZMQInteractiveShell':
            if self.data_properties.use_full_data:
                display(self.data_properties.data.head(10)) # Hack for jupyter notebooks
                
                return ''
            else:
                display(self.data_properties.train_data.head(10)) # Hack for jupyter notebooks

                return ''
        
        else:
            if self.data_properties.use_full_data:
                return self.data_properties.data.to_string()
            else:
                return self.data_properties.train_data.to_string()


    def __getitem__(self, column):

        if self.data_properties.use_full_data:
            return self.data_properties.data[column]
        else:
            return self.data_properties.train_data[column]


    def __setitem__(self, column, value):

        if self.data_properties.use_full_data:
            self.data_properties.data[column] = value

            return self.data_properties.data.head(10)
        else:
            train_data_length = self.data_properties.train_data.shape[0]
            test_data_length = self.data_properties.test_data.shape[0]

            if isinstance(value, list):
                ## If the number of entries in the list does not match the number of rows in the training or testing
                ## set raise a value error
                if len(value) != train_data_length and len(value) != test_data_length:
                    raise ValueError(f"Length of list: {len(value)} does not equal the number rows as the training set or test set.")

                self._set_item(column, value, train_data_length, test_data_length)

            elif isinstance(value, tuple):
                for data in value:
                    if len(data) != train_data_length and len(data) != test_data_length:
                        raise ValueError(f"Length of list: {len(data)} does not equal the number rows as the training set or test set.")

                    self._set_item(column, data, train_data_length, test_data_length)

            else:
                self.data_properties.train_data[column] = value
                self.data_properties.test_data[column] = value

            return self.data_properties.train_data.head(10)


    def __getattr__(self, column):

        if self.data_properties.use_full_data:
            return self.data_properties.data[column]
        else:
            return self.data_properties.train_data[column]

    def __setattr__(self, item, value):
        
        if item not in self.__dict__:       # any normal attributes are handled normally
            dict.__setattr__(self, item, value)
        else:
            self.__setitem__(item, value)            

    @property
    def data(self):
        """
        Property function for the entire dataset.
        """
        
        if self.data_properties.data is None:
            return "There seems to be nothing here. Try .train_data or .test_data"
        
        return self.data_properties.data

    @data.setter
    def data(self, value):
        """
        Setter function for the entire dataset.
        """

        self.data_properties.data = value


    @property
    def train_data(self):
        """
        Property function for the training dataset.
        """
        
        if self.data_properties.train_data is None:
            return "There seems to be nothing here. Try .data"

        return self.data_properties.train_data

    @train_data.setter
    def train_data(self, value):
        """
        Setter function for the training dataset.
        """

        if self.data_properties.train_data is None:
            return "There seems to be nothing here. Try .data"

        self.data_properties.train_data = value
        
    @property
    def test_data(self):
        """
        Property function for the entire dataset.
        """

        return self.data_properties.test_data

    @test_data.setter
    def test_data(self, value):
        """
        Property function for the entire dataset.
        """

        self.data_properties.test_data = value

    @property
    def target_field(self):

        return self.data_properties.target_field

    @target_field.setter
    def target_field(self, value):

        self.data_properties.target_field = values
        
    @property
    def missing_values(self):
        """
        Property function that shows how many values are missing in each column.
        """

        dataframes = list(filter(lambda x: x is not None, [
                          self.data_properties.data, self.data_properties.train_data, self.data_properties.test_data]))

        for dataframe in dataframes:
            if not dataframe.isnull().values.any():            
                print("No missing values!")
            else:
                total = dataframe.isnull().sum().sort_values(ascending=False)
                percent = (dataframe.isnull().sum() /
                           dataframe.isnull().count()).sort_values(ascending=False)
                missing_data = pd.concat(
                    [total, percent], axis=1, keys=['Total', 'Percent'])

                display(missing_data.T)


    def describe(self, dataset='train'):
        """
        Describes your dataset using the DataFrameSummary library with basic descriptive info.
        Extends the DataFrame.describe() method to give more info.

        Credits go to @mouradmourafiq for his pandas-summary library.
        
        Parameters
        ----------
        dataset : str, optional
            Type of dataset to describe. Can either be `train` or `test`.
            If you are using the full dataset it will automatically describe
            your full dataset no matter the input, 
            by default 'train'
        
        Returns
        -------
        DataFrame
            Dataframe describing your dataset with basic descriptive info
        """

        if self.data_properties.use_full_data:
            data_summary = DataFrameSummary(self.data)

            return data_summary.summary()
        else:
            if dataset == 'train':            
                train_data_summary = DataFrameSummary(self.train_data)

                return train_data_summary.summary()
            else:
                test_data_summary = DataFrameSummary(self.test_data)

                return test_data_summary.summary()


    def column_info(self, dataset='train'):
        """
        Describes your columns using the DataFrameSummary library with basic descriptive info.

        Credits go to @mouradmourafiq for his pandas-summary library.

        Info
        ----
        counts
        uniques
        missing
        missing_perc
        types
        
        Parameters
        ----------
        dataset : str, optional
            Type of dataset to describe. Can either be `train` or `test`.
            If you are using the full dataset it will automatically describe
            your full dataset no matter the input, 
            by default 'train'
        
        Returns
        -------
        DataFrame
            Dataframe describing your columns with basic descriptive info
        """

        if self.data_properties.use_full_data:
            data_summary = DataFrameSummary(self.data)

            return data_summary.columns_stats
        else:
            if dataset == 'train':            
                train_data_summary = DataFrameSummary(self.train_data)

                return train_data_summary.columns_stats
            else:
                test_data_summary = DataFrameSummary(self.test_data)

                return test_data_summary.columns_stats


    def describe_column(self, column, dataset='train'):
        """
        Analyzes a column and reports descriptive statistics about the columns.

        Credits go to @mouradmourafiq for his pandas-summary library.

        Statistics
        ----------
        std                                      
        max                                      
        min                                      
        variance                                 
        mean
        mode                                     
        5%                                       
        25%                                      
        50%                                      
        75%                                      
        95%                                      
        iqr                                      
        kurtosis                                 
        skewness                                 
        sum                                      
        mad                                      
        cv                                       
        zeros_num                                
        zeros_perc                               
        deviating_of_mean                        
        deviating_of_mean_perc                   
        deviating_of_median                      
        deviating_of_median_perc                 
        top_correlations                         
        counts                                   
        uniques                                  
        missing                                  
        missing_perc                             
        types                            
        
        Parameters
        ----------
        column : str
            Column in your dataset you want to analze.
        dataset : str, optional
            Type of dataset to describe. Can either be `train` or `test`.
            If you are using the full dataset it will automatically describe
            your full dataset no matter the input, 
            by default 'train'
        
        Returns
        -------
        dict
            Dictionary mapping a statistic and its value for a specific column
        """

        if self.data_properties.use_full_data:
            data_summary = DataFrameSummary(self.data)

            return data_summary[column]
        else:
            if dataset == 'train':            
                train_data_summary = DataFrameSummary(self.train_data)

                return train_data_summary[column]
            else:
                test_data_summary = DataFrameSummary(self.test_data)

                return test_data_summary[column]
                

    def drop(self, *drop_columns, reason=''):
        """
        Drops columns from the dataframe.
        
        Parameters
        ----------
        reason : str, optional
            Reasoning for dropping columns, by default ''

        Column names must be provided as strings that exist in the data.
        
        Raises
        ------
        ValueError
            If all columns provided don't exist in Dataframe

        Examples
        --------
        >>> clean.drop('A', 'B', reason="Columns were unimportant")
        >>> feature.drop('A', 'B', reason="SME deemed columns were unnecessary")
        >>> preprocess.drop('A', 'B')        
        """

        try:
            data_columns = self.data.columns
        except:
            data_columns = self.train_data.columns

        drop_columns = list(drop_columns)

        if  not all(elem in data_columns for elem in drop_columns):
            raise ValueError("Not all columns exist in Dataframe")

        if self.data_properties.use_full_data:
            self.data_properties.data = self.data.drop(drop_columns, axis=1)

            if self.report is not None:
                self.report.log(f'Dropped columns: {", ".join(drop_columns)}. {reason}')

            return self.data.head(10)
        else:
            self.data_properties.train_data = self.train_data.drop(drop_columns, axis=1)
            self.data_properties.test_data = self.test_data.drop(drop_columns, axis=1)

            if self.report is not None:
                self.report.log(f'Dropped columns {", ".join(drop_columns)} in both train and test set. {reason}')

            return self.train_data.head(10)

    def _set_item(self, column: str, value: list, train_length: int, test_length: int):
        """
        Utility function for __setitem__ for determining which input is for which dataset
        and then sets the input to the new column for the correct dataset.
        
        Parameters
        ----------
        column : str
            New column name
        value : list
            List of values for new column
        train_length : int
            Length of training data
        test_length : int
            Length of training data
        """

        ## If the training data and testing data have the same number of rows, apply the value to both
        ## train and test data set
        if len(value) == train_length and len(value) == test_length:
            self.data_properties.train_data[column] = value
            self.data_properties.test_data[column] = value

        elif len(value) == train_length:
            self.data_properties.train_data[column] = value

        else:
            self.data_properties.test_data[column] = value


    def visualize_raincloud(self, x_col: str, y_col=None, params={}):
        """
        Combines the box plot, scatter plot and split violin plot into one data visualization.
        This is used to offer eyeballed statistical inference, assessment of data distributions (useful to check assumptions),
        and the raw data itself showing outliers and underlying patterns.

        A raincloud is made of:
        1) "Cloud", kernel desity estimate, the half of a violinplot.
        2) "Rain", a stripplot below the cloud
        3) "Umberella", a boxplot
        4) "Thunder", a pointplot connecting the mean of the different categories (if `pointplot` is `True`)

        Parameters
        ----------
        x_col : str
            X axis data, reference by column name, any data
        y_col : str
            Y axis data, reference by column name, measurable data (numeric)
            by default target_field
        params : dict, optional
            Parameters for the rain cloud plot, by default
                { 'x'=target_col
                'y'=col
                'data'=data.infer_objects()
                'pointplot'=True
                'width_viol'=0.8
                'width_box'=.4
                'figsize'=(12,8)
                'orient'='h'
                'move'=0. }

        Possible Params for Raincloud Plot
        ----------------------------------
        x : Iterable, np.array, or dataframe column name if 'data' is specified
            Categorical data.
        y : Iterable, np.array, or dataframe column name if 'data' is specified
            Measure data (Numeric)
        hue : Iterable, np.array, or dataframe column name if 'data' is specified
            Second categorical data. Use it to obtain different clouds and rainpoints
        data : Dataframe              
            input pandas dataframe
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

        Useful parameter documentation
        ------------------------------
            https://seaborn.pydata.org/generated/seaborn.boxplot.html
            https://seaborn.pydata.org/generated/seaborn.violinplot.html
            https://seaborn.pydata.org/generated/seaborn.stripplot.html
        
        Examples
        --------
        >>> clean.visualize_raincloud('col1') # Will plot col1 values on the x axis and your target variable values on the y axis
        >>> clean.visualize_raincloud('col1', 'col2') # Will plot col1 on the x and col2 on the y axis
        """

        if y_col is None:
            y_col = self.target_field

        if self.data_properties.use_full_data:
            raincloud(y_col, x_col, self.data)
        else:
            raincloud(y_col, x_col, self.train_data)

    def visualize_barplot(self, x_col, y_col, groupby=None, method=None, orient='v', **kwargs):
        """
        Plots a bar plot for the given columns provided.

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
        y : str
            Column for the y axis
        groupby : str
            Data to groupby - x-axis, optional, by default None
        method : str
            Method to aggregate groupy data
            Examples: min, max, mean, etc., optional
            by default None
        orient : str, optional
            Orientation of graph, 'h' for horizontal
            'v' for vertical, by default 'v'
        """

        if self.data_properties.use_full_data:
            barplot(x_col, y_col, self.data, groupby=groupby, method=method, orient=orient, **kwargs)
        else:
            barplot(x_col, y_col, self.train_data, groupby=groupby, method=method, orient=orient, **kwargs)
