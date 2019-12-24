import copy
import os
import re

import aethos
import ipywidgets as widgets
import numpy as np
import pandas as pd
import pandas_profiling
from aethos.cleaning.clean import Clean
from aethos.config import shell
from aethos.feature_engineering.feature import Feature
from aethos.preprocessing.preprocess import Preprocess
from aethos.reporting.report import Report
from aethos.stats.stats import Stats
from aethos.util import (
    CLEANING_CHECKLIST,
    DATA_CHECKLIST,
    ISSUES_CHECKLIST,
    MULTI_ANALYSIS_CHECKLIST,
    PREPARATION_CHECKLIST,
    UNI_ANALYSIS_CHECKLIST,
    _get_columns,
    _set_item,
    label_encoder,
    split_data,
)
from aethos.visualizations.visualizations import Visualizations
from IPython import get_ipython
from IPython.display import HTML, display
from ipywidgets import Layout
from pandas.io.json import json_normalize
from pandas_summary import DataFrameSummary


class Data(Clean, Preprocess, Feature, Visualizations, Stats):
    """
    Data class thats run analytical techniques.

    Parameters
    -----------
    x_train: pd.DataFrame
        Training data or aethos data object

    x_test: pd.DataFrame
        Test data, by default None

    split: bool
        True to split your training data into a train set and a test set

    test_split_percentage: float
        Percentage of data to split train data into a train and test set.
        Only used if `split=True`

    target_field: str
        For supervised learning problems, the name of the column you're trying to predict.

    report_name: str
        Name of the report to generate, by default None
    """

    def __init__(
        self,
        x_train,
        x_test=None,
        split=True,
        test_split_percentage=0.2,
        target_field="",
        report_name=None,
    ):

        self.x_train = x_train
        self.x_test = x_test
        self.split = split
        self.target_field = target_field
        self.report_name = report_name
        self.test_split_percentage = test_split_percentage
        self.target_mapping = None

        if split and x_test is None:
            # Generate train set and test set.
            self.x_train, self.x_test = split_data(self.x_train, test_split_percentage)
            self.x_train.reset_index(drop=True, inplace=True)
            self.x_test.reset_index(drop=True, inplace=True)

        if report_name is not None:
            self.report = Report(report_name)
            self.report_name = self.report.filename
        else:
            self.report = None
            self.report_name = None

        Visualizations.__init__(self, x_train)

    def __repr__(self):

        return self.x_train.to_string()

    def _repr_html_(self):  # pragma: no cover

        return self.x_train.head().to_html(show_dimensions=True, notebook=True)

    def __getitem__(self, column):

        try:
            return self.x_train[column]

        except Exception as e:
            raise AttributeError(e)

    def __setitem__(self, column, value):

        if not self.split:
            self.x_train[column] = value

            return self.x_train.head()
        else:
            x_train_length = self.x_train.shape[0]
            x_test_length = self.x_test.shape[0]

            if isinstance(value, (list, np.ndarray)):
                ## If the number of entries in the list does not match the number of rows in the training or testing
                ## set raise a value error
                if len(value) != x_train_length and len(value) != x_test_length:
                    raise ValueError(
                        f"Length of list: {str(len(value))} does not equal the number rows as the training set or test set."
                    )

                self.x_train, self.x_test = _set_item(
                    self.x_train,
                    self.x_test,
                    column,
                    value,
                    x_train_length,
                    x_test_length,
                )

            elif isinstance(value, tuple):
                for data in value:
                    if len(data) != x_train_length and len(data) != x_test_length:
                        raise ValueError(
                            f"Length of list: {str(len(value))} does not equal the number rows as the training set or test set."
                        )

                    (self.x_train, self.x_test,) = _set_item(
                        self.x_train,
                        self.x_test,
                        column,
                        data,
                        x_train_length,
                        x_test_length,
                    )

            else:
                self.x_train[column] = value
                self.x_test[column] = value

            return self.x_train.head()

    def __getattr__(self, key):

        if key in self.__dict__:
            return getattr(self, key)

        if key in self.x_train.columns:
            return self.x_train[key]
        else:
            if hasattr(self.x_train, key):
                return getattr(self.x_train, key)
            else:
                raise AttributeError(f"Name does not have attribute {key}.")

    def __setattr__(self, item, value):

        if item not in self.__dict__ or hasattr(
            self, item
        ):  # any normal attributes are handled normally
            dict.__setattr__(self, item, value)
        else:
            self.__setitem__(item, value)

    def __deepcopy__(self, memo):

        new_inst = type(self)(
            x_train=self.x_train,
            x_test=self.x_test,
            split=self.split,
            test_split_percentage=self.test_split_percentage,
            target_field=self.target_field,
            report_name=self.report_name,
        )

        return new_inst

    @property
    def y_train(self):
        """
        Property function for the training predictor variable
        """

        return self.x_train[self.target_field] if self.target_field else None

    @y_train.setter
    def y_train(self, value):
        """
        Setter function for the training predictor variable
        """

        if self.target_field:
            self.x_train[self.target_field] = value
        else:
            self.target_field = "label"
            self.x_train["label"] = value
            print('Added a target (predictor) field (column) named "label".')

    @property
    def y_test(self):
        """
        Property function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target_field:
                return self.x_test[self.target_field]
            else:
                return None
        else:
            return None

    @y_test.setter
    def y_test(self, value):
        """
        Setter function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target_field:
                self.x_test[self.target_field] = value
            else:
                self.target_field = "label"
                self.x_test["label"] = value
                print('Added a target (predictor) field (column) named "label".')

    @y_test.setter
    def y_test(self, value):
        """
        Setter function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target_field:
                self.x_test[self.target_field] = value
            else:
                self.target_field = "label"
                self.x_test["label"] = value
                print('Added a target (predictor) field (column) named "label".')

    @property
    def columns(self):
        """
        Property to return columns in the dataset.
        """

        return self.x_train.columns.tolist()

    @property
    def missing_values(self):
        """
        Property function that shows how many values are missing in each column.
        """

        dataframes = list(
            filter(lambda x: x is not None, [self.x_train, self.x_test,],)
        )

        missing_df = []
        for ind, dataframe in enumerate(dataframes):
            caption = (
                "Train set missing values." if ind == 0 else "Test set missing values."
            )

            if not dataframe.isnull().values.any():
                print("No missing values!")  # pragma: no cover
            else:
                total = dataframe.isnull().sum().sort_values(ascending=False)
                percent = (
                    dataframe.isnull().sum() / dataframe.isnull().count()
                ).sort_values(ascending=False)
                missing_data = pd.concat(
                    [total, percent], axis=1, keys=["Total", "Percent"]
                )

                missing_df.append(
                    missing_data.style.format({"Percent": "{:.2%}"}).set_caption(
                        caption
                    )
                )

        def multi_table(table_list):
            """ Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
            """
            return HTML(
                '<table><tr style="background-color:white;">'
                + "".join(
                    [
                        "<td style='padding-right:25px'>"
                        + table._repr_html_()
                        + "</td>"
                        for table in table_list
                    ]
                )
                + "</tr></table>"
            )

        if shell == "ZMQInteractiveShell":
            display(multi_table(missing_df))
        else:
            for table in missing_df:
                print(table)

    def copy(self):
        """
        Returns deep copy of object.
        
        Returns
        -------
        Object
            Deep copy of object
        """

        return copy.deepcopy(self)

    def log(self, msg: str):
        """
        Logs notes to the report file
        """

        self.report.log(msg)

    def standardize_column_names(self):
        """
        Utility function that standardizes all column names to lowercase and underscores for spaces.

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.standardize_column_names()
        """

        new_column_names = {}
        pattern = re.compile("\W+")

        for name in self.x_train.columns:
            new_column_names[name] = re.sub(pattern, "_", name.lower())

        self.col_mapping = new_column_names

        self.x_train.rename(columns=new_column_names, inplace=True)

        if self.x_test is not None:
            self.x_test.rename(columns=new_column_names, inplace=True)

        return self.copy()

    def expand_json_column(self, col):
        """
        Utility function that expands a column that has JSON elements into columns, where each JSON key is a column. 

        Parameters
        ----------
        cols: str
            Column in the data that has the nested data.

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.expand_json_column('col1')
        """

        df = json_normalize(self.x_train[col], sep="_")
        self.x_train.drop(col, axis=1, inplace=True)
        self.x_train = pd.concat([self.x_train, df], axis=1)

        if self.x_test is not None:
            df = json_normalize(self.x_test[col], sep="_")
            self.x_test.drop(col, axis=1, inplace=True)
            self.x_test = pd.concat([self.x_test, df], axis=1)

        return self.copy()

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

        Returns
        -------
        Data or DataFrame:
            Returns a deep copy of the Data object or a DataFrame if replace is False.

        Examples
        --------
        >>> data.search('aethos')
        """

        # TODO: Refactor this to take in boolean expressions

        if not values:
            return ValueError("Please provide value to search.")

        if replace:
            if not_equal:
                self.x_train = self.x_train[self.x_train.isin(list(values)).any(axis=1)]

                if self.x_test is not None:
                    self.x_test = self.x_test[
                        self.x_test.isin(list(values)).any(axis=1)
                    ]
            else:
                self.x_train = self.x_train[self.x_train.isin(list(values)).any(axis=1)]

                if self.x_test is not None:
                    self.x_test = self.x_test[
                        self.x_test.isin(list(values)).any(axis=1)
                    ]

            return self.copy()
        else:
            data = self.x_train.copy()

            if not not_equal:
                data = data[data.isin(list(values))].dropna(how="all")
            else:
                data = data[~data.isin(list(values))].dropna(how="all")

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
        '''This translates to your data where col2 is equal to 3 and col 3 is equal to 4 and column 4 is equal to 1, 2 or 3.
        The col1 specifies that this this is the only column you want to see at the output.'''
        >>> data.where('col1', col2=3, col3=4, col4=[1,2,3])
        """

        filtered_data = self.x_train.copy()

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
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.groupby('col1', 'col2')
        """

        if not groupby:
            return ValueError("Please provided columns to groupby.")

        if replace:
            self.x_train = self.x_train.groupby(list(groupby))

            if self.x_test is not None:
                self.x_test = self.x_test.groupby(list(groupby))

            return self.copy()
        else:
            data = self.x_train.copy()

            return data.groupby(list(groupby))

    def groupby_analysis(self, groupby: list, *cols, data_filter=None):
        """
        TODO: Refactor this to a class method.

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

        Examples
        --------
        >>> data.groupby_analysis()
        >>> data.groupby_analysis('col1', 'col2')
        """

        analysis = {}
        numeric_analysis = [
            "count",
            "min",
            "max",
            "mean",
            "std",
            "var",
            "median",
            ("most_common", lambda x: pd.Series.mode(x)[0]),
            "sum",
            "mad",
            "nunique",
        ]
        other_analysis = [
            "count",
            ("most_common", lambda x: pd.Series.mode(x)[0]),
            "nunique",
        ]

        list_of_cols = _get_columns(list(cols), self.x_train)

        if isinstance(data_filter, pd.DataFrame):
            data = data_filter
        else:
            data = self.x_train.copy()

        for col in list_of_cols:
            if col not in groupby:
                # biufc - bool, int, unsigned, float, complex
                if data[col].dtype.kind in "biufc":
                    analysis[col] = numeric_analysis
                else:
                    analysis[col] = other_analysis

        analyzed_data = data.groupby(groupby).agg(analysis)

        return analyzed_data

    def data_report(self, title="Profile Report", output_file="", suppress=False):
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

        Examples
        --------
        >>> data.data_report()
        >>> data.data_report(title='Titanic EDA', output_file='titanic.html')
        """

        if shell == "ZMQInteractiveShell":  # pragma : no cover
            report = self.x_train.profile_report(
                title=title, style={"full_width": True}
            )
        else:
            report = self.x_train.profile_report(title=title)

        if output_file:
            report.to_file(output_file=output_file)

        if not suppress:
            return report

    def describe(self, dataset="train"):
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

        Examples
        ---------
        >>> data.describe()
        """

        if dataset == "train":
            x_train_summary = DataFrameSummary(self.x_train)

            return x_train_summary.summary()
        else:
            x_test_summary = DataFrameSummary(self.x_test)

            return x_test_summary.summary()

    def column_info(self, dataset="train"):
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

        Examples
        ---------
        >>> data.column_info()
        """

        if dataset == "train":
            x_train_summary = DataFrameSummary(self.x_train)

            return x_train_summary.columns_stats
        else:
            x_test_summary = DataFrameSummary(self.x_test)

            return x_test_summary.columns_stats

    def describe_column(self, column, dataset="train"):
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
            
        Examples
        --------
        >>> data.describe_column('col1')
        """

        if dataset == "train":
            x_train_summary = DataFrameSummary(self.x_train)

            return x_train_summary[column]
        else:
            x_test_summary = DataFrameSummary(self.x_test)

            return x_test_summary[column]

    def drop(self, *drop_columns, keep=[], regexp="", reason=""):
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
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.drop('A', 'B', reason="Columns were unimportant")
        >>> data.drop('col1', keep=['col2'], regexp=r"col*") # Drop all columns that start with "col" except column 2
        >>> data.drop(keep=['A']) # Drop all columns except column 'A'
        >>> data.drop(regexp=r'col*') # Drop all columns that start with 'col'       
        """

        if not isinstance(keep, list):
            raise TypeError("Keep parameter must be a list.")

        # Handles if columns do not exist in the dataframe
        data_columns = self.x_train.columns
        regex_columns = []

        if regexp:
            regex = re.compile(regexp)
            regex_columns = list(filter(regex.search, data_columns))

        drop_columns = set(drop_columns).union(regex_columns)

        # If there are columns to be dropped, exclude the ones in the keep list
        # If there are no columns to be dropped, drop everything except the keep list
        if drop_columns:
            drop_columns = list(drop_columns.difference(keep))
        else:
            keep = set(data_columns).difference(keep)
            drop_columns = list(drop_columns.union(keep))

        self.x_train = self.x_train.drop(drop_columns, axis=1)

        if self.x_test is not None:
            self.x_test = self.x_test.drop(drop_columns, axis=1)

        if self.report is not None:
            self.report.log(f'Dropped columns: {", ".join(drop_columns)}. {reason}')

        return self.copy()

    def encode_target(self):
        """
        Encodes target variables with value between 0 and n_classes-1.

        Running this function will automatically set the corresponding mapping for the target variable mapping number to the original value.

        Note that this will not work if your test data will have labels that your train data does not.        

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.encode_target()
        """

        if not self.target_field:
            raise ValueError(
                "Please set the `target_field` field variable before encoding."
            )

        (self.x_train, self.x_test, self.target_mapping,) = label_encoder(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=[self.target_field],
            target=True,
        )

        if self.report is not None:
            self.report.log("Encoded the target variable as numeric values.")

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

        Examples
        --------
        >>> data.to_csv('titanic')
        """

        index = kwargs.pop("index", index)
        chunksize = kwargs.pop("chunksize", 10000)

        self.x_train.to_csv(
            name + "_train.csv", index=index, chunksize=chunksize, **kwargs
        )

        if self.x_test is not None:
            self.x_test.to_csv(
                name + "_test.csv", index=index, chunksize=chunksize, **kwargs
            )

    def checklist(self):
        """
        Displays a checklist dashboard with reminders for a Data Science project.

        Examples
        --------
        >>> data.checklist()
        """

        data_checkboxes = []
        clean_checkboxes = []
        analysis_checkboxes = [
            [widgets.Label(value="Univariate Analysis")],
            [widgets.Label(value="Multivariate Analysis")],
            [widgets.Label(value="Timeseries Analysis")],
        ]
        issue_checkboxes = []
        preparation_checkboxes = []

        for item in DATA_CHECKLIST:
            data_checkboxes.append(
                widgets.Checkbox(description=item, layout=Layout(width="100%"))
            )
        data_box = widgets.VBox(data_checkboxes)

        for item in CLEANING_CHECKLIST:
            clean_checkboxes.append(
                widgets.Checkbox(description=item, layout=Layout(width="100%"))
            )
        clean_box = widgets.VBox(clean_checkboxes)

        for item in UNI_ANALYSIS_CHECKLIST:
            analysis_checkboxes[0].append(
                widgets.Checkbox(description=item, layout=Layout(width="100%"))
            )
        uni_box = widgets.VBox(analysis_checkboxes[0])

        for item in MULTI_ANALYSIS_CHECKLIST:
            analysis_checkboxes[1].append(
                widgets.Checkbox(description=item, layout=Layout(width="100%"))
            )

        multi_box = widgets.VBox(analysis_checkboxes[1])

        analysis_box = widgets.HBox([uni_box, multi_box])

        for item in ISSUES_CHECKLIST:
            issue_checkboxes.append(
                widgets.Checkbox(description=item, layout=Layout(width="100%"))
            )
        issue_box = widgets.VBox(issue_checkboxes)

        for item in PREPARATION_CHECKLIST:
            preparation_checkboxes.append(
                widgets.Checkbox(description=item, layout=Layout(width="100%"))
            )
        prep_box = widgets.VBox(preparation_checkboxes)

        tab_list = [data_box, clean_box, analysis_box, issue_box, prep_box]

        tab = widgets.Tab()
        tab.children = tab_list
        tab.set_title(0, "Data")
        tab.set_title(1, "Cleaning")
        tab.set_title(2, "Analysis")
        tab.set_title(3, "Issues")
        tab.set_title(4, "Preparation")

        display(tab)

    def to_df(self):
        """
        Return Dataframes for x_train and x_test if it exists.

        Returns
        -------
        Dataframe, *Dataframe
            Transformed dataframe with rows with a missing values in a specific column are missing

            Returns 2 Dataframes test if x_test is provided.

        Examples
        --------
        >>> data.to_df()
        """

        if self.x_test is None:
            return self.x_train
        else:
            return self.x_train, self.x_test
