import pandas as pd
from aethos.cleaning import util
from aethos.cleaning.categorical import *
from aethos.cleaning.numeric import *
from aethos.cleaning.util import replace_missing_fill
from aethos.config import technique_reason_repo
from aethos.util import _input_columns, _numeric_input_conditions


class Clean(object):
    def drop_column_missing_threshold(self, threshold: float):
        """
        Remove columns from the dataframe that have greater than or equal to the threshold value of missing values.
        Example: Remove columns where >= 50% of the data is missing.

        This function exists in `clean/utils.py`
        
        Parameters
        ----------
        threshold : float
            Value between 0 and 1 that describes what percentage of a column can be missing values.
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.drop_column_missing_threshold(0.5)
        """

        report_info = technique_reason_repo["clean"]["general"]["remove_columns"]

        original_columns = set(list(self.x_train.columns))

        (self.x_train, self.x_test,) = util.remove_columns_threshold(
            x_train=self.x_train, x_test=self.x_test, threshold=threshold,
        )

        if self.report is not None:
            new_columns = original_columns.difference(self.x_train.columns)
            self.report.report_technique(report_info, new_columns)

        return self.copy()

    def drop_constant_columns(self):
        """
        Remove columns from the data that only have one unique value.

        This function exists in `clean/utils.py`
                
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.drop_constant_columns()
        """

        report_info = technique_reason_repo["clean"]["general"][
            "remove_constant_columns"
        ]

        original_columns = set(list(self.x_train.columns))

        (self.x_train, self.x_test,) = util.remove_constant_columns(
            x_train=self.x_train, x_test=self.x_test
        )

        if self.report is not None:
            new_columns = original_columns.difference(self.x_train.columns)
            self.report.report_technique(report_info, new_columns)

        return self.copy()

    def drop_unique_columns(self):
        """
        Remove columns from the data that only have one unique value.

        This function exists in `clean/utils.py`
                
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.drop_unique_columns()
        """

        report_info = technique_reason_repo["clean"]["general"]["remove_unique_columns"]

        original_columns = set(list(self.x_train.columns))

        (self.x_train, self.x_test,) = util.remove_unique_columns(
            x_train=self.x_train, x_test=self.x_test
        )

        if self.report is not None:
            new_columns = original_columns.difference(self.x_train.columns)
            self.report.report_technique(report_info, new_columns)

        return self.copy()

    def drop_rows_missing_threshold(self, threshold: float):
        """
        Remove rows from the dataframe that have greater than or equal to the threshold value of missing rows.
        Example: Remove rows where > 50% of the data is missing.

        This function exists in `clean/utils.py`.

        Parameters
        ----------
        threshold : float
            Value between 0 and 1 that describes what percentage of a row can be missing values.
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.drop_rows_missing_threshold(0.5)    
        """

        report_info = technique_reason_repo["clean"]["general"]["remove_rows"]

        (self.x_train, self.x_test,) = util.remove_rows_threshold(
            x_train=self.x_train, x_test=self.x_test, threshold=threshold,
        )

        # Write to report
        if self.report is not None:
            self.report.report_technique(report_info)

        return self.copy()

    def replace_missing_mean(self, *list_args, list_of_cols=[]):
        """
        Replaces missing values in every numeric column with the mean of that column.

        If no columns are supplied, missing values will be replaced with the mean in every numeric column.

        Mean: Average value of the column. Effected by outliers.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/numeric.py` as `replace_missing_mean_median_mode`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to

        list_of_cols : list, optional
            Specific columns to apply this technique to, by default []
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_mean('col1', 'col2')
        >>> data.replace_missing_mean(['col1', 'col2'])
        """

        report_info = technique_reason_repo["clean"]["numeric"]["mean"]

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = replace_missing_mean_median_mode(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            strategy="mean",
        )

        if self.report is not None:
            if list_of_cols:
                self.report.report_technique(report_info, list_of_cols)
            else:
                list_of_cols = _numeric_input_conditions(list_of_cols, self.x_train)
                self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def replace_missing_median(self, *list_args, list_of_cols=[]):
        """
        Replaces missing values in every numeric column with the median of that column.

        If no columns are supplied, missing values will be replaced with the mean in every numeric column.

        Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/numeric.py` as `replace_missing_mean_median_mode`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            Specific columns to apply this technique to., by default []
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_median('col1', 'col2')
        >>> data.replace_missing_median(['col1', 'col2'])
        """

        report_info = technique_reason_repo["clean"]["numeric"]["median"]

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = replace_missing_mean_median_mode(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            strategy="median",
        )

        if self.report is not None:
            if list_of_cols:
                self.report.report_technique(report_info, list_of_cols)
            else:
                list_of_cols = _numeric_input_conditions(list_of_cols, self.x_train)
                self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def replace_missing_mostcommon(self, *list_args, list_of_cols=[]):
        """
        Replaces missing values in every numeric column with the most common value of that column

        Mode: Most common value.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/numeric.py` as `replace_missing_mean_median_mode`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_mostcommon('col1', 'col2')
        >>> data.replace_missing_mostcommon(['col1', 'col2'])
        """

        report_info = technique_reason_repo["clean"]["numeric"]["mode"]

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = replace_missing_mean_median_mode(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            strategy="most_frequent",
        )
        if self.report is not None:
            if list_of_cols:
                self.report.report_technique(report_info, list_of_cols)
            else:
                list_of_cols = _numeric_input_conditions(list_of_cols, self.x_train)
                self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def replace_missing_constant(
        self, *list_args, list_of_cols=[], constant=0, col_mapping=None
    ):
        """
        Replaces missing values in every numeric column with a constant.

        If no columns are supplied, missing values will be replaced with the mean in every numeric column.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/numeric.py` as `replace_missing_constant`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        constant : int or float, optional
            Numeric value to replace all missing values with , by default 0

        col_mapping : dict, optional
            Dictionary mapping {'ColumnName': `constant`}, by default None
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_constant(col_mapping={'a': 1, 'b': 2, 'c': 3})
        >>> data.replace_missing_constant('col1', 'col2', constant=2)
        >>> data.replace_missing_constant(['col1', 'col2'], constant=3)
        """

        report_info = technique_reason_repo["clean"]["numeric"]["constant"]

        if col_mapping:
            col_to_constant = col_mapping
        else:
            # If a list of columns is provided use the list, otherwise use arguemnts.
            col_to_constant = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = replace_missing_constant(
            x_train=self.x_train,
            x_test=self.x_test,
            col_to_constant=col_to_constant,
            constant=constant,
        )

        if self.report is not None:
            if not col_to_constant:
                self.report.report_technique(report_info, self.x_train.columns)
            else:
                self.report.report_technique(report_info, list(col_to_constant))

        return self.copy()

    def replace_missing_new_category(
        self, *list_args, list_of_cols=[], new_category=None, col_mapping=None
    ):
        """
        Replaces missing values in categorical column with its own category. The categories can be autochosen
        from the defaults set.

        For numeric categorical columns default values are: -1, -999, -9999
        For string categorical columns default values are: "Other", "Unknown", "MissingDataCategory"

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/categorical.py` as `replace_missing_new_category`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        new_category : str, int, or float, optional
            Category to replace missing values with, by default None

        col_mapping : dict, optional
           Dictionary mapping {'ColumnName': `constant`}, by default None
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_new_category(col_mapping={'col1': "Green", 'col2': "Canada", 'col3': "December"})
        >>> data.replace_missing_new_category('col1', 'col2', 'col3', new_category='Blue')
        >>> data.replace_missing_new_category(['col1', 'col2', 'col3'], new_category='Blue')
        """

        report_info = technique_reason_repo["clean"]["categorical"]["new_category"]

        # If dictionary mapping is provided, use that otherwise use column
        if col_mapping:
            col_to_category = col_mapping
        else:
            # If a list of columns is provided use the list, otherwise use arguemnts.
            col_to_category = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = replace_missing_new_category(
            x_train=self.x_train,
            x_test=self.x_test,
            col_to_category=col_to_category,
            constant=new_category,
        )

        if self.report is not None:
            if not col_to_category:
                self.report.report_technique(report_info, self.x_train.columns)
            else:
                self.report.report_technique(report_info, list(col_to_category))

        return self.copy()

    def replace_missing_remove_row(self, *list_args, list_of_cols=[]):
        """
        Remove rows where the value of a column for those rows is missing.

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/categorical.py` as `replace_missing_remove_row`.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_remove_row('col1', 'col2')
        >>> data.replace_missing_remove_row(['col1', 'col2'])
        """

        report_info = technique_reason_repo["clean"]["categorical"]["remove_rows"]

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = replace_missing_remove_row(
            x_train=self.x_train, x_test=self.x_test, cols_to_remove=list_of_cols,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def drop_duplicate_rows(self, *list_args, list_of_cols=[]):
        """
        Remove rows from the data that are exact duplicates of each other and leave only 1.
        This can be used to reduce processing time or performance for algorithms where
        duplicates have no effect on the outcome (i.e DBSCAN)

        If a list of columns is provided use the list, otherwise use arguemnts.

        This function exists in `clean/util.py` as `remove_duplicate_rows`.
       
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
       
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.drop_duplicate_rows('col1', 'col2') # Only look at columns 1 and 2
        >>> data.drop_duplicate_rows(['col1', 'col2'])
        >>> data.drop_duplicate_rows()
        """

        report_info = technique_reason_repo["clean"]["general"]["remove_duplicate_rows"]

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = util.remove_duplicate_rows(
            x_train=self.x_train, x_test=self.x_test, list_of_cols=list_of_cols,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def drop_duplicate_columns(self):
        """
        Remove columns from the data that are exact duplicates of each other and leave only 1.
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.drop_duplicate_columns()
        """

        report_info = technique_reason_repo["clean"]["general"][
            "remove_duplicate_columns"
        ]

        (self.x_train, self.x_test,) = util.remove_duplicate_columns(
            x_train=self.x_train, x_test=self.x_test
        )

        if self.report is not None:
            self.report.report_technique(report_info)

        return self.copy()

    def replace_missing_random_discrete(self, *list_args, list_of_cols=[]):
        """
        Replace missing values in with a random number based off the distribution (number of occurences) 
        of the data.

        For example if your data was [5, 5, NaN, 1, 2]
        There would be a 50% chance that the NaN would be replaced with a 5, a 25% chance for 1 and a 25% chance for 2.

        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
            
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_random_discrete('col1', 'col2')
        >>> data.replace_missing_random_discrete(['col1', 'col2'])
        """

        report_info = technique_reason_repo["clean"]["general"]["random_discrete"]

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = util.replace_missing_random_discrete(
            x_train=self.x_train, x_test=self.x_test, list_of_cols=list_of_cols,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def replace_missing_knn(self, k=5, **knn_kwargs):
        """
        Replaces missing data with data from similar records based off a distance metric.

        For more info see: https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer
        
        Parameters
        ----------
        missing_values : number, string, np.nan or None, default=`np.nan`
            The placeholder for the missing values. All occurrences of missing_values will be imputed.

        k : int, default=5
            Number of neighboring samples to use for imputation.

        weights : {‘uniform’, ‘distance’} or callable, default=’uniform’
            Weight function used in prediction. Possible values:

                ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.

                ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

                callable : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

        metric : {‘nan_euclidean’} or callable, default=’nan_euclidean’
            Distance metric for searching neighbors. Possible values:

                ‘nan_euclidean’

                callable : a user-defined function which conforms to the definition of _pairwise_callable(X, Y, metric, **kwds).
                The function accepts two arrays, X and Y, and a missing_values keyword in kwds and returns a scalar distance value.

        add_indicator : bool, default=False
            If True, a MissingIndicator transform will stack onto the output of the imputer’s transform.
            This allows a predictive estimator to account for missingness despite imputation.
            If a feature has no missing values at fit/train time, the feature won’t appear on the missing indicator even if there are missing values at transform/test time.

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_knn(k=8)
        """

        report_info = technique_reason_repo["clean"]["general"]["knn"]

        (self.x_train, self.x_test,) = util.replace_missing_knn(
            x_train=self.x_train, x_test=self.x_test, n_neighbors=k, **knn_kwargs
        )

        if self.report is not None:
            self.report.report_technique(report_info)

        return self.copy()

    def replace_missing_interpolate(
        self, *list_args, list_of_cols=[], method="linear", **inter_kwargs
    ):
        """
        Replaces missing values with an interpolation method and possible extrapolation.

        The possible interpolation methods are:
           
            - 'linear': Ignore the index and treat the values as equally spaced. This is the only method supported on MultiIndexes.
            - 'time': Works on daily and higher resolution data to interpolate given length of interval.
            - 'index', ‘values’: use the actual numerical values of the index.
            - 'pad': Fill in NaNs using existing values.
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline', ‘barycentric’, ‘polynomial’: Passed to scipy.interpolate.interp1d.
                - These methods use the numerical values of the index. Both ‘polynomial’ and ‘spline’ require that you also specify an order (int), e.g. df.interpolate(method='polynomial', order=5).
            - 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima': Wrappers around the SciPy interpolation methods of similar names.
            - 'from_derivatives': Refers to scipy.interpolate.BPoly.from_derivatives which replaces ‘piecewise_polynomial’ interpolation method in scipy 0.18.

        For more information see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html or https://docs.scipy.org/doc/scipy/reference/interpolate.html.

        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
            
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        method : str, optional
            Interpolation method, by default 'linear'

        limit : int, optional
            Maximum number of consecutive NaNs to fill. Must be greater than 0.

        limit_area : {None, ‘inside’, ‘outside’}, default None
            If limit is specified, consecutive NaNs will be filled with this restriction.

            - None: No fill restriction.
            - ‘inside’: Only fill NaNs surrounded by valid values (interpolate).
            - ‘outside’: Only fill NaNs outside valid values (extrapolate).
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_interpolate('col1', 'col2')
        >>> data.replace_missing_interpolate(['col1', 'col2'])
        >>> data.replace_missing_interpolate('col1', 'col2', method='pad', limit=3)
        """

        report_info = technique_reason_repo["clean"]["general"]["interpolate"]
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = util.replace_missing_interpolate(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            method=method,
            **inter_kwargs
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def replace_missing_backfill(self, *list_args, list_of_cols=[], **extra_kwargs):
        """
        Replaces missing values in a column with the next known data point.

        This is useful when dealing with timeseries data and you want to replace data in the past with data from the future.

        For more info view the following link: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
            
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_backfill('col1', 'col2')
        >>> data.replace_missing_backfill(['col1', 'col2'])
        """

        report_info = technique_reason_repo["clean"]["general"]["bfill"]
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = util.replace_missing_fill(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            method="bfill",
            **extra_kwargs
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def replace_missing_forwardfill(self, *list_args, list_of_cols=[], **extra_kwargs):
        """
        Replaces missing values in a column with the last known data point.

        This is useful when dealing with timeseries data and you want to replace future missing data with the past.

        For more info view the following link: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
            
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_forwardfill('col1', 'col2')
        >>> data.replace_missing_forwardfill(['col1', 'col2'])
        """

        report_info = technique_reason_repo["clean"]["general"]["ffill"]
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = util.replace_missing_fill(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            method="ffill",
            **extra_kwargs
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def replace_missing_indicator(
        self,
        *list_args,
        list_of_cols=[],
        missing_indicator=1,
        valid_indicator=0,
        keep_col=True
    ):
        """
        Adds a new column describing whether data is missing for each record in a column.

        This is useful if the missing data has meaning, aka not random.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
            
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        missing_indicator : int, optional
            Value to indicate missing data, by default 1

        valid_indicator : int, optional
            Value to indicate non missing data, by default 0

        keep_col : bool, optional
            True to keep column, False to replace it, by default False
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.replace_missing_indicator('col1', 'col2')
        >>> data.replace_missing_indicator(['col1', 'col2'])
        >>> data.replace_missing_indicator(['col1', 'col2'], missing_indicator='missing', valid_indicator='not missing', keep_col=False)
        """

        report_info = technique_reason_repo["clean"]["general"]["indicator"]
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = util.replace_missing_indicator(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            missing_indicator=missing_indicator,
            valid_indicator=valid_indicator,
            keep_col=keep_col,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()
