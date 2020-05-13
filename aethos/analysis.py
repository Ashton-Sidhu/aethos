import copy
import os
import re

import ipywidgets as widgets
import numpy as np
import pandas as pd

from aethos.config import shell
from aethos.stats.stats import Stats
from aethos.util import (
    CLEANING_CHECKLIST,
    DATA_CHECKLIST,
    ISSUES_CHECKLIST,
    MULTI_ANALYSIS_CHECKLIST,
    PREPARATION_CHECKLIST,
    UNI_ANALYSIS_CHECKLIST,
    _get_columns,
    _get_attr_,
    _get_item_,
    _interpret_data,
    label_encoder,
)
from aethos.visualizations.visualizations import Visualizations
from IPython import get_ipython
from IPython.display import HTML, display
from ipywidgets import Layout


class Analysis(Visualizations, Stats):
    """
    Core class thats run analytical techniques.

    Parameters
    -----------
    x_train: pd.DataFrame
        Training data or aethos data object

    x_test: pd.DataFrame
        Test data, by default None

    target: str
        For supervised learning problems, the name of the column you're trying to predict.
    """

    def __init__(
        self, x_train, x_test=None, target="",
    ):

        self.x_train = x_train
        self.x_test = x_test
        self.target = target
        self.target_mapping = None

    def __repr__(self):
        return self.x_train.to_string()

    def _repr_html_(self):  # pragma: no cover

        if self.target:
            cols = self.features + [self.target]
        else:
            cols = self.features

        return self.x_train[cols].head()._repr_html_()

    def __getitem__(self, column):
        return _get_item_(self, column)

    def __getattr__(self, key):
        return _get_attr_(self, key)

    def __deepcopy__(self, memo):

        x_test = self.x_test.copy() if self.x_test is not None else None

        new_inst = type(self)(
            x_train=self.x_train.copy(), x_test=x_test, target=self.target,
        )

        new_inst.target_mapping = self.target_mapping

        return new_inst

    @property
    def features(self):
        """Features for modelling"""

        cols = self.x_train.columns.tolist()

        if self.target:
            cols.remove(self.target)

        return cols

    @property
    def y_test(self):
        """
        Property function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target:
                return self.x_test[self.target]
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
            if self.target:
                self.x_test[self.target] = value
            else:
                self.target = "label"
                self.x_test["label"] = value
                print('Added a target (predictor) field (column) named "label".')

    @property
    def y_train(self):
        """
        Property function for the training predictor variable
        """

        return self.x_train[self.target] if self.target else None

    @y_train.setter
    def y_train(self, value):
        """
        Setter function for the training predictor variable
        """

        if self.target:
            self.x_train[self.target] = value
        else:
            self.target = "label"
            self.x_train["label"] = value
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

        if self.target is not None:
            self.target = re.sub(pattern, "_", self.target.lower())

        self.col_mapping = new_column_names

        self.x_train.rename(columns=new_column_names, inplace=True)

        if self.x_test is not None:
            self.x_test.rename(columns=new_column_names, inplace=True)

        return self

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

        from pandas.io.json import json_normalize

        df = json_normalize(self.x_train[col], sep="_")
        self.x_train.drop(col, axis=1, inplace=True)
        self.x_train = pd.concat([self.x_train, df], axis=1)

        if self.x_test is not None:
            df = json_normalize(self.x_test[col], sep="_")
            self.x_test.drop(col, axis=1, inplace=True)
            self.x_test = pd.concat([self.x_test, df], axis=1)

        return self

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

        import pandas_profiling

        if shell == "ZMQInteractiveShell":  # pragma : no cover
            report = self.x_train.profile_report(
                title=title, html={"style": {"full_width": True}}
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

        from pandas_summary import DataFrameSummary

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

        from pandas_summary import DataFrameSummary

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

        from pandas_summary import DataFrameSummary

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

        return self

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

        fig = self._viz.viz_correlation_matrix(
            self.x_train.corr(),
            data_labels=data_labels,
            hide_mirror=hide_mirror,
            output_file=output_file,
            **kwargs,
        )

        return fig

    def predictive_power(
        self, col=None, data_labels=False, hide_mirror=False, output_file="", **kwargs
    ):
        """
        Calculated the Predictive Power Score of each feature.

        If a column is provided, it will calculate it in regards to the target variable.

        Credits go to Florian Wetschorek - https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598

        Parameters
        ----------
        col : str
            Column in the dataframe

        data_labels : bool, optional
            True to display the correlation values in the plot, by default False

        hide_mirror : bool, optional
            Whether to display the mirroring half of the correlation plot, by default False

        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)

        Examples
        --------
        >>> data.predictive_power(data_labels=True)
        >>> data.predictive_power(col='col1')
        """

        import ppscore as pps
        import seaborn as sns

        if col:
            return pps.score(self.x_train, col, self.target)
        else:
            pp_df = pps.matrix(self.x_train)

            fig = self._viz.viz_correlation_matrix(
                pp_df,
                data_labels=data_labels,
                hide_mirror=hide_mirror,
                output_file=output_file,
                **kwargs,
            )

            return fig

    def autoviz(self, max_rows=150000, max_cols=30, verbose=0):  # pragma: no cover
        """
        Auto visualizes and analyzes your data to help explore your data.

        Credits go to AutoViMl - https://github.com/AutoViML/AutoViz

        Parameters
        ----------
        max_rows : int, optional
            Max rows to analyze, by default 150000

        max_cols : int, optional
            Max columns to analyze, by default 30

        verbose : {0, 1, 2}, optional
            0 - it does not print any messages and goes into silent mode
            1 - print messages on the terminal and also display
                charts on terminal
            2 - it will print messages but will not display charts,
                it will simply save them.
        """

        from autoviz.AutoViz_Class import AutoViz_Class

        target = self.target if self.target else ""

        AV = AutoViz_Class()

        dft = AV.AutoViz(
            "",
            dfte=self.x_train,
            depVar=target,
            max_cols_analyzed=max_cols,
            max_rows_analyzed=max_rows,
            verbose=verbose,
        )

    def interpret_data(self, show=True):
        """
        Interpret your data using MSFT Interpret dashboard.
        """

        if self.target:
            _interpret_data(
                self.x_train.drop(self.target, axis=1), self.y_train, show=show
            )
        else:
            return "Unsupported without a target variable."

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

        if not self.target:
            raise ValueError("Please set the `target` field variable before encoding.")

        (self.x_train, self.x_test, self.target_mapping,) = label_encoder(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=[self.target],
            target=True,
        )

        for k, v in self.target_mapping.items():
            print(f"{k}: {v}")

        return self

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
