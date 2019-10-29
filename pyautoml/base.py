import copy
import os
import re

import numpy as np
import pandas as pd
import pandas_profiling
import yaml
from IPython import get_ipython
from IPython.display import display
from pandas_summary import DataFrameSummary
import ipywidgets as widgets
from ipywidgets import Layout

import pyautoml
from pyautoml.data.data import Data
from pyautoml.util import _get_columns, _set_item, label_encoder, split_data, DATA_CHECKLIST, CLEANING_CHECKLIST, UNI_ANALYSIS_CHECKLIST, MULTI_ANALYSIS_CHECKLIST, ISSUES_CHECKLIST, PREPARATION_CHECKLIST
from pyautoml.visualizations.visualize import *


# TODO: Move to a config file

SHELL = get_ipython().__class__.__name__

pkg_directory = os.path.dirname(pyautoml.__file__)

with open("{}/technique_reasons.yml".format(pkg_directory), 'r') as stream:
    try:
        technique_reason_repo = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Could not load yaml file.")

class MethodBase(object):

    def __init__(self, **kwargs):

        x_train = kwargs.pop('x_train')
        x_test = kwargs.pop('x_test')
        split = kwargs.pop('split')
        target_field = kwargs.pop('target_field')
        target_mapping = kwargs.pop('target_mapping')
        report_name = kwargs.pop('report_name')
        test_split_percentage = kwargs.pop('test_split_percentage')

        self._data_properties = Data(x_train, x_test, split=split, target_field=target_field, target_mapping=target_mapping, report_name=report_name)

        if split and x_test is None:
            # Generate train set and test set.
            self._data_properties.x_train, self._data_properties.x_test = split_data(self._data_properties.x_train, test_split_percentage)
            self._data_properties.x_train.reset_index(drop=True, inplace=True)
            self._data_properties.x_test.reset_index(drop=True, inplace=True)

        if self._data_properties.report is None:
            self.report = None
        else:
            self.report = self._data_properties.report        
            
    def __repr__(self):

        if SHELL == 'ZMQInteractiveShell':            
            display(self._data_properties.x_train.head()) # Hack for jupyter notebooks

            return ''
        
        else:
            return self._data_properties.x_train.to_string()


    def __getitem__(self, column):

        try: 
            return self._data_properties.x_train[column]

        except Exception as e:
            raise AttributeError(e)
        

    def __setitem__(self, column, value):

        if not self._data_properties.split:
            self._data_properties.x_train[column] = value

            return self._data_properties.x_train.head()
        else:
            x_train_length = self._data_properties.x_train.shape[0]
            x_test_length = self._data_properties.x_test.shape[0]

            if isinstance(value, list):
                ## If the number of entries in the list does not match the number of rows in the training or testing
                ## set raise a value error
                if len(value) != x_train_length and len(value) != x_test_length:
                    raise ValueError("Length of list: {} does not equal the number rows as the training set or test set.".format(str(len(value))))

                self._data_properties.x_train, self._data_properties.x_test = _set_item(
                    self._data_properties.x_train, self._data_properties.x_test, column, value, x_train_length, x_test_length)

            elif isinstance(value, tuple):
                for data in value:
                    if len(data) != x_train_length and len(data) != x_test_length:
                        raise ValueError("Length of list: {} does not equal the number rows as the training set or test set.".format(str(len(data))))

                    self._data_properties.x_train, self._data_properties.x_test = _set_item(
                        self._data_properties.x_train, self._data_properties.x_test, column, data, x_train_length, x_test_length)

            else:
                self._data_properties.x_train[column] = value
                self._data_properties.x_test[column] = value

            return self._data_properties.x_train.head()

    def __getattr__(self, column):

        try:
            return self._data_properties.x_train[column]

        except Exception as e:
            raise AttributeError(e)

    def __setattr__(self, item, value):
        
        if item not in self.__dict__:       # any normal attributes are handled normally
            dict.__setattr__(self, item, value)
        else:
            self.__setitem__(item, value)

    def __deepcopy__(self, memo):
        
        data_props = copy.deepcopy(self._data_properties)
        new_inst = type(self)(data_props)

        return new_inst

    @property
    def x_train(self):
        """
        Property function for the training dataset.
        """

        return self._data_properties.x_train

    @x_train.setter
    def x_train(self, value):
        """
        Setter function for the training dataset.
        """

        self._data_properties.x_train = value
        
    @property
    def x_test(self):
        """
        Property function for the test dataset.
        """

        return self._data_properties.x_test

    @x_test.setter
    def x_test(self, value):
        """
        Setter for the test data set.
        """

        self._data_properties.x_test = value

    @property
    def target_field(self):
        """
        Property function for the target field.
        """

        return self._data_properties.target_field

    @target_field.setter
    def target_field(self, value):
        """
        Setter for the target field
        """

        self._data_properties.target_field = value

    @property
    def target_mapping(self):
        """
        Property function for the label mapping
        """

        return self._data_properties.target_mapping

    @target_mapping.setter
    def target_mapping(self, value):
        """
        Setter for the label mapping
        """

        self._data_properties.target_mapping = value
        
    @property
    def missing_values(self):
        """
        Property function that shows how many values are missing in each column.
        """

        dataframes = list(filter(lambda x: x is not None, [
                          self._data_properties.x_train, self._data_properties.x_train, self._data_properties.x_test]))

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

    def copy(self):
        """
        Returns deep copy of object.
        
        Returns
        -------
        Object
            Deep copy of object
        """

        return copy.deepcopy(self)

    def search(self, *values, not_equal=False, replace=False):
        """
        Searches the entire dataset for specified value(s) and returns rows that contain the values.
        
        Parameters
        ----------
        values : Any
            Value to search for in dataframe

        not_equal : bool, optional
            True if you want filter by values in the dataframe that are not equal to the value provided, by default False

        replace : bool, optional
            Whether to permanently transform your data, by default False
        """

        # TODO: Refactor this to take in boolean expressions

        if not values:
            return ValueError("Please provided columns to groupby.")        

        if replace:
            if not_equal:
                self._data_properties.x_train = self._data_properties.x_train[self._data_properties.x_train.isin(list(values)).any(axis=1)]
                
                if self._data_properties.x_test is not None:
                    self._data_properties.x_test = self._data_properties.x_test[self._data_properties.x_test.isin(list(values)).any(axis=1)]
            else:
                self._data_properties.x_train = self._data_properties.x_train[self._data_properties.x_train.isin(list(values)).any(axis=1)]
                
                if self._data_properties.x_test is not None:
                    self._data_properties.x_test = self._data_properties.x_test[self._data_properties.x_test.isin(list(values)).any(axis=1)]

            return self.copy()            
        else:
            data = self._data_properties.x_train.copy()
           
            if not not_equal:
                data = data[data.isin(list(values))].dropna(how='all')
            else:
                data = data[~data.isin(list(values))].dropna(how='all')
                
            return data

    def where(self, *filter_columns, **columns):
        """
        Filters the dataframe down for highlevel analysis. 

        Can only handle '==', for more complex queries, interact with pandas.
        
        Parameters
        ----------
        filter_columns : str(s)
            Columns you want to see at the end result

        columns : key word arguments
            Columns and the associated value to filter on.
            Columns can equal a value or a list of values to include.
        
        Returns
        -------
        Dataframe
            A view of your data or training data

        Examples
        --------
        >>> clean.where('col1', col2=3, col3=4, col4=[1,2,3])
        This translates to your data where col2 is equal to 3 and col 3 is equal to 4 and column 4 is equal to 1, 2 or 3.
        The col1 specifies that this this is the only column you want to see at the output.
        """

        filtered_data = self._data_properties.x_train.copy()

        for col in columns.keys():
            if isinstance(columns[col], list):
                filtered_data = filtered_data[filtered_data[col].isin(columns[col])]
            else:
                filtered_data = filtered_data[filtered_data[col] == columns[col]]

        if filter_columns:
            return filtered_data[list(filter_columns)]
        else:
            return filtered_data
        
    def groupby(self, *groupby, replace=False):
        """
        Groups data by the provided columns.
        
        Parameters
        ----------
        groupby : str(s)
            Columns to group the data by.

        replace : bool, optional
            Whether to permanently transform your data, by default False
        
        Returns
        -------
        Dataframe, Clean, Preprocess or Feature
            Dataframe or copy of object
        """

        if not groupby:
            return ValueError("Please provided columns to groupby.")

        if replace:
            self._data_properties.x_train = self._data_properties.x_train.groupby(list(groupby))

            if self._data_properties.x_test is not None:
                self._data_properties.x_test = self._data_properties.x_test.groupby(list(groupby))

            return self.copy()            
        else:
            data = self._data_properties.x_train.copy()

            return data.groupby(list(groupby))

    def groupby_analysis(self, groupby: list, *cols, data_filter=None):
        """
        Groups your data and then provides descriptive statistics for the other columns on the grouped data.

        For numeric data, the descriptive statistics are:

            - count
            - min
            - max
            - mean
            - standard deviation
            - variance
            - median
            - most common
            - sum
            - Median absolute deviation
            - number of unique values

        For other types of data:

            - count
            - most common
            - number of unique values
        
        Parameters
        ----------
        groupby : list
            List of columns to groupby.

        cols : str(s)
            Columns you want statistics on, if none are provided, it will provide statistics for every column.

        data_filter : Dataframe, optional
            Filtered dataframe, by default None
        
        Returns
        -------
        Dataframe
            Dataframe of grouped columns and statistics for each column.
        """

        analysis = {}
        numeric_analysis = ['count', 'min', 'max', 'mean', 'std', 'var', 'median', ('most_common', lambda x: pd.Series.mode(x)[0]), 'sum', 'mad', 'nunique']
        other_analysis = ['count', ('most_common', lambda x: pd.Series.mode(x)[0]), 'nunique']

        list_of_cols = _get_columns(list(cols), self._data_properties.x_train)

        if isinstance(data_filter, pd.DataFrame):
            data = data_filter
        else:
            data = self._data_properties.x_train.copy()            

        for col in list_of_cols:
            if col not in groupby:
                #biufc - bool, int, unsigned, float, complex
                if data[col].dtype.kind in 'biufc':
                    analysis[col] = numeric_analysis
                else:
                    analysis[col] = other_analysis

        analyzed_data = data.groupby(groupby).agg(analysis)
        
        return analyzed_data

    def data_report(self, title='Profile Report', output_file='', suppress=False):
        """
        Generates a full Exploratory Data Analysis report using Pandas Profiling.

        Credits: https://github.com/pandas-profiling/pandas-profiling
        
        For each column the following statistics - if relevant for the column type - are presented in an interactive HTML report:

        - Essentials: type, unique values, missing values
        - Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range
        - Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
        - Most frequent values
        - Histogram
        - Correlations highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
        - Missing values matrix, count, heatmap and dendrogram of missing values
        
        Parameters
        ----------
        title : str, optional
            Title of the report, by default 'Profile Report'

        output_file : str, optional
            File name of the output file for the report, by default ''

        suppress : bool, optional
            True if you do not want to display the report, by default False
        
        Returns
        -------
        HTML display of Exploratory Data Analysis report
        """


        if SHELL == "ZMQInteractiveShell":
            report = self._data_properties.x_train.profile_report(title=title, style={'full_width':True})
        else:
            report = self._data_properties.x_train.profile_report(title=title)

        if output_file:
            report.to_file(output_file=output_file)

        if not suppress:
            return report
        

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

       
        if dataset == 'train':            
            x_train_summary = DataFrameSummary(self.x_train)

            return x_train_summary.summary()
        else:
            x_test_summary = DataFrameSummary(self.x_test)

            return x_test_summary.summary()


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

        if dataset == 'train':            
            x_train_summary = DataFrameSummary(self.x_train)

            return x_train_summary.columns_stats
        else:
            x_test_summary = DataFrameSummary(self.x_test)

            return x_test_summary.columns_stats

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

        if dataset == 'train':            
            x_train_summary = DataFrameSummary(self.x_train)

            return x_train_summary[column]
        else:
            x_test_summary = DataFrameSummary(self.x_test)

            return x_test_summary[column]
            

    def drop(self, *drop_columns, keep=[], regexp='', reason=''):
        """
        Drops columns from the dataframe.
        
        Parameters
        ----------
        keep : list: optional
            List of columns to not drop, by default []

        regexp : str, optional
            Regular Expression of columns to drop, by default ''

        reason : str, optional
            Reasoning for dropping columns, by default ''

        Column names must be provided as strings that exist in the data.
        
        Returns
        -------
        self : Object
            Return deep copy of itself.

        Examples
        --------
        >>> clean.drop('A', 'B', reason="Columns were unimportant")
        >>> feature.drop('A', 'B', reason="SME deemed columns were unnecessary")
        >>> preprocess.drop('A', 'B')        
        """

        data_columns = self.x_train.columns

        if not drop_columns:
            drop_columns = data_columns
        else:
            drop_columns = set(data_columns).intersection(drop_columns)

        if regexp:
            regex = re.compile(regexp)
            regex_columns = list(filter(regex.search, data_columns))
        else:
            regex_columns = []

        drop_columns = list(set(set(drop_columns).union(regex_columns)).difference(keep))
        
        self._data_properties.x_train = self.x_train.drop(drop_columns, axis=1)

        if self._data_properties.x_test is not None:
            self._data_properties.x_test = self.x_test.drop(drop_columns, axis=1)

        if self.report is not None:
            self.report.log('Dropped columns: {}. {}'.format(", ".join(drop_columns), reason))

        return self.copy()

    
    def encode_target(self):
        """
        Encodes target variables with value between 0 and n_classes-1.

        Running this function will automatically set the corresponding mapping for the target variable mapping number to the original value.

        Note that this will not work if your test data will have labels that your train data does not.        

        Returns
        -------
        Clean, Preprocess, Feature or Model
            Copy of object
        """

        if not self._data_properties.target_field:
            raise ValueError('Please set the `target_field` field variable before encoding.')
    
        self._data_properties.x_train, self._data_properties.x_test, self._data_properties.target_mapping = label_encoder(
                x_train=self._data_properties.x_train, x_test=self._data_properties.x_test, list_of_cols=self._data_properties.target_field, target=True)

        if self.report is not None:
            self.report.log('Encoded the target variable as numeric values.')

        return self.copy()

    def to_csv(self, name: str, index=False, **kwargs):
        """
        Write data to csv with the name and path provided.

        The function will automatically add '.csv' to the end of the name.

        By default it writes 10000 rows at a time to file to consider memory on different machines.

        Training data will end in '_train.csv' andt test data will end in '_test.csv'.

        For a full list of keyword args for writing to csv please see the following link: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
        
        Parameters
        ----------
        name : str
            File path
        index : bool, optional
            True to write 'index' column, by default False
        """

        index = kwargs.pop('index', index)
        chunksize = kwargs.pop('chunksize', 10000)

        self._data_properties.x_train.to_csv(name + '_train.csv', index=index, chunksize=chunksize, **kwargs)

        if self._data_properties.x_test is not None:
            self._data_properties.x_test.to_csv(name + '_test.csv', index=index, chunksize=chunksize, **kwargs)

    def checklist(self):
        """
        Displays a checklist dashboard with reminders for a Data Science project.
        """

        data_checkboxes = []
        clean_checkboxes = []
        analysis_checkboxes = [[widgets.Label(value='Univariate Analysis')], [widgets.Label(value='Multivariate Analysis')], [widgets.Label(value='Timeseries Analysis')]]
        issue_checkboxes = []
        preparation_checkboxes = []

        for item in DATA_CHECKLIST:        
            data_checkboxes.append(widgets.Checkbox(
                description=item, layout=Layout(width='100%')))
        data_box = widgets.VBox(data_checkboxes)

        for item in CLEANING_CHECKLIST:
            clean_checkboxes.append(widgets.Checkbox(
                description=item, layout=Layout(width='100%')))
        clean_box = widgets.VBox(clean_checkboxes)

        for item in UNI_ANALYSIS_CHECKLIST:
            analysis_checkboxes[0].append(widgets.Checkbox(
                description=item, layout=Layout(width='100%')))
        uni_box = widgets.VBox(analysis_checkboxes[0])

        for item in MULTI_ANALYSIS_CHECKLIST:
            analysis_checkboxes[1].append(widgets.Checkbox(
                description=item, layout=Layout(width='100%')))
            
        multi_box = widgets.VBox(analysis_checkboxes[1])
                
        analysis_box = widgets.HBox([uni_box, multi_box])

        for item in ISSUES_CHECKLIST:
            issue_checkboxes.append(widgets.Checkbox(
                description=item, layout=Layout(width='100%')))
        issue_box = widgets.VBox(issue_checkboxes)

        for item in PREPARATION_CHECKLIST:
            preparation_checkboxes.append(widgets.Checkbox(
                description=item, layout=Layout(width='100%')))
        prep_box = widgets.VBox(preparation_checkboxes)

        tab_list = [data_box, clean_box, analysis_box, issue_box, prep_box]

        tab = widgets.Tab()
        tab.children = tab_list
        tab.set_title(0, 'Data')
        tab.set_title(1, 'Cleaning')
        tab.set_title(2, 'Analysis')
        tab.set_title(3, 'Issues')
        tab.set_title(4, 'Preparation')

        display(tab)

    def visualize_raincloud(self, x_col: str, y_col=None, **params):
        """
        Combines the box plot, scatter plot and split violin plot into one data visualization.
        This is used to offer eyeballed statistical inference, assessment of data distributions (useful to check assumptions),
        and the raw data itself showing outliers and underlying patterns.

        A raincloud is made of:
        1) "Cloud", kernel desity estimate, the half of a violinplot.
        2) "Rain", a stripplot below the cloud
        3) "Umberella", a boxplot
        4) "Thunder", a pointplot connecting the mean of the different categories (if `pointplot` is `True`)

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
        
        Examples
        --------
        >>> clean.visualize_raincloud('col1') # Will plot col1 values on the x axis and your target variable values on the y axis
        >>> clean.visualize_raincloud('col1', 'col2') # Will plot col1 on the x and col2 on the y axis
        """

        if y_col is None:
            y_col = self.target_field

        raincloud(y_col, x_col, self.x_train)

    def visualize_barplot(self, x_col, *cols, groupby=None, method=None, orient='v', stacked=False, output_file='', **barplot_kwargs):
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
        x_col : str
            Column name for the x axis.

        cols : str
            Columns you would like to see plotted against the x_col

        groupby : str
            Data to groupby - x-axis, optional, by default None

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
        """
        
        barplot(x_col, list(cols), self._data_properties.x_train, groupby=groupby, method=method, orient=orient, stacked=stacked, **barplot_kwargs)

    def visualize_scatterplot(self, x_col: str, y_col: str, category=None, title='Scatter Plot', size=8, output_file='', **scatterplot_kwargs):
        """
        Plots a scatterplot for the given x and y columns provided using Bokeh.

        For a list of possible scatterplot_kwargs please check out the following links:

        https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.scatter

        https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#userguide-styling-line-properties 

        https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#userguide-styling-fill-properties 
        
        Parameters
        ----------
        x_col : str
            X column name

        y_col : str
            Y column name

        category : str, optional
            Category to group your data, by default None

        title : str, optional
            Title of the plot, by default 'Scatter Plot'

        size : int or str, optional
            Size of the circle, can either be a number
            or a column name to scale the size, by default 8

        fill_color : color value, optional
            Colour or Colour palette to set fill colour

        line_color : color value, optional
            Colour or Colour palette to set line colour

        output_file : str, optional
            Output html file name for image

        **scatterplot_kwargs : optional
            See above links for list of possible scatterplot options.
        """

        scatterplot(x_col, y_col, self._data_properties.x_train, title=title, category=category, size=size, output_file=output_file, **scatterplot_kwargs)

    def visualize_lineplot(self, x_col: str, *y_cols, title='Line Plot', output_file='', **lineplot_kwargs):
        """
        Plots a lineplot for the given x and y columns provided using Bokeh.

        For a list of possible lineplot_kwargs please check out the following links:

        https://github.com/PatrikHlobil/Pandas-Bokeh#lineplot

        https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.line 
        
        Parameters
        ----------
        x_col : str
            X column name

        y_cols : str or str(s)
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
        """

        lineplot(x_col, list(y_cols), self._data_properties.x_train, title=title, output_file=output_file, **lineplot_kwargs)
