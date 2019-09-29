import pandas as pd

from pyautoml import technique_reason_repo
from pyautoml.base import MethodBase
from pyautoml.cleaning.categorical import *
from pyautoml.cleaning.numeric import *
from pyautoml.cleaning.util import *
from pyautoml.util import (_contructor_data_properties, _input_columns,
                           _numeric_input_conditions)


class Clean(MethodBase):

    
    def __init__(self, step=None, x_train=None, x_test=None, _data_properties=None, test_split_percentage=0.2, split=True, target_field="", report_name=None):   
        
        _data_properties = _contructor_data_properties(step)

        if _data_properties is None:        
            super().__init__(x_train=x_train, x_test=x_test, test_split_percentage=test_split_percentage,
                        split=split, target_field=target_field, target_mapping=None, report_name=report_name)
        else:
            super().__init__(x_train=_data_properties.x_train, x_test=_data_properties.x_test, test_split_percentage=test_split_percentage,
                        split=_data_properties.split, target_field=_data_properties.target_field, target_mapping=_data_properties.target_mapping, report_name=_data_properties.report_name)
        
        if self._data_properties.report is not None:
            self.report.write_header("Cleaning")


    def remove_columns(self, threshold: float):
        """
        Remove columns from the dataframe that have more than the threshold value of missing columns.
        Example: Remove columns where > 50% of the data is missing.

        This function exists in `clean/utils.py`
        
        Parameters
        ----------
        threshold : float
            Value between 0 and 1 that describes what percentage of a column can be missing values.
        
        Returns
        -------
        Clean:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['general']['remove_columns']

        
        original_columns = set(list(self._data_properties.x_train.columns))

        self._data_properties.x_train, self._data_properties.x_test = remove_columns_threshold(x_train=self._data_properties.x_train,
                                                                                                x_test=self._data_properties.x_test,
                                                                                                threshold=threshold)

        if self.report is not None:
            new_columns = original_columns.difference(self._data_properties.x_train.columns)
            self.report.report_technique(report_info, new_columns)

        return self.copy()


    def remove_rows(self, threshold: float):
        """
        Remove rows from the dataframe that have more than the threshold value of missing rows.
        Example: Remove rows where > 50% of the data is missing.

        This function exists in `clean/utils.py`.

        Parameters
        ----------
        threshold : float
            Value between 0 and 1 that describes what percentage of a row can be missing values.
        
        Returns
        -------
        Clean:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['general']['remove_rows']

        self._data_properties.x_train, self._data_properties.x_test = remove_rows_threshold(x_train=self._data_properties.x_train,
                                                                                                x_test=self._data_properties.x_test,
                                                                                                threshold=threshold)

        #Write to report
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
        Clean:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['numeric']['mean']
        
        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)
 
        self._data_properties.x_train, self._data_properties.x_test = replace_missing_mean_median_mode(x_train=self._data_properties.x_train,
                                                                                                        x_test=self._data_properties.x_test,
                                                                                                        list_of_cols=list_of_cols,
                                                                                                        strategy="mean",                                                                                                            
                                                                                                        )
            
        if self.report is not None:
            if list_of_cols:
                self.report.report_technique(report_info, list_of_cols)
            else:
                list_of_cols = _numeric_input_conditions(list_of_cols, self._data_properties.x_train)
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
        Clean:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['numeric']['median']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        self._data_properties.x_train, self._data_properties.x_test = replace_missing_mean_median_mode(x_train=self._data_properties.x_train,
                                                                                                        x_test=self._data_properties.x_test,
                                                                                                        list_of_cols=list_of_cols,
                                                                                                        strategy="median",                                                                                                            
                                                                                                        )

        if self.report is not None:
            if list_of_cols:
                self.report.report_technique(report_info, list_of_cols)
            else:
                list_of_cols = _numeric_input_conditions(list_of_cols, self._data_properties.x_train)
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
        Clean:
            Returns a deep copy of the Clean object.
        """
       
        report_info = technique_reason_repo['clean']['numeric']['mode']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        self._data_properties.x_train, self._data_properties.x_test = replace_missing_mean_median_mode(x_train=self._data_properties.x_train,
                                                                                                        x_test=self._data_properties.x_test,
                                                                                                        list_of_cols=list_of_cols,
                                                                                                        strategy="most_frequent",                                                                                                            
                                                                                                        )
        if self.report is not None:
            if list_of_cols:
                self.report.report_technique(report_info, list_of_cols)
            else:
                list_of_cols = _numeric_input_conditions(list_of_cols, self._data_properties.x_train)
                self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def replace_missing_constant(self, *list_args, list_of_cols=[], constant=0, col_mapping=None):
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
        Clean:
            Returns a deep copy of the Clean object.

        Examples
        --------
        >>> replace_missing_constant(col_mapping={'a': 1, 'b': 2, 'c': 3})
        >>> replace_missing_constant('col1', 'col2', constant=2)
        """

        report_info = technique_reason_repo['clean']['numeric']['constant']

        if col_mapping:
            col_to_constant = col_mapping
        else:
            # If a list of columns is provided use the list, otherwise use arguemnts.
            col_to_constant = _input_columns(list_args, list_of_cols)

        
        self._data_properties.x_train, self._data_properties.x_test = replace_missing_constant(x_train=self._data_properties.x_train,
                                                                                                x_test=self._data_properties.x_test,
                                                                                                col_to_constant=col_to_constant,
                                                                                                constant=constant,                                                                                                    
                                                                                                )

        if self.report is not None:
            if not col_to_constant:
                self.report.report_technique(report_info, self._data_properties.x_train.columns)
            else:
                self.report.report_technique(report_info, list(col_to_constant))

        return self.copy()


    def replace_missing_new_category(self, *list_args, list_of_cols=[], new_category=None, col_mapping=None):
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
        Clean:
            Returns a deep copy of the Clean object.

        Examples
        --------
        >>> replace_missing_new_category(col_mapping={'col1': "Green", 'col2': "Canada", 'col3': "December"})
        >>> replace_missing_new_category('col1', 'col2', 'col3', new_category='Blue')
        """

        report_info = technique_reason_repo['clean']['categorical']['new_category']
        
        # If dictionary mapping is provided, use that otherwise use column
        if col_mapping:
            col_to_category = col_mapping
        else:
            # If a list of columns is provided use the list, otherwise use arguemnts.
            col_to_category = _input_columns(list_args, list_of_cols)

        self._data_properties.x_train, self._data_properties.x_test = replace_missing_new_category(x_train=self._data_properties.x_train,
                                                                                                    x_test=self._data_properties.x_test,
                                                                                                    col_to_category=col_to_category,
                                                                                                    constant=new_category,                                                                                                                                                                                                            
                                                                                                    )

        if self.report is not None:
            if not col_to_category:
                self.report.report_technique(report_info, self._data_properties.x_train.columns)
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
        Clean:
            Returns a deep copy of the Clean object.
        """

        report_info = technique_reason_repo['clean']['categorical']['remove_rows']

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        self._data_properties.x_train, self._data_properties.x_test = replace_missing_remove_row(x_train=self._data_properties.x_train,
                                                                                                x_test=self._data_properties.x_test,
                                                                                                cols_to_remove=list_of_cols                                                                                           
                                                                                                )                                                                                        

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()


    def remove_duplicate_rows(self, *list_args, list_of_cols=[]):
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
        Clean:
            Returns a deep copy of the Clean object.
        """
    
        report_info = technique_reason_repo['clean']['general']['remove_duplicate_rows']
        
        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)
   
        self._data_properties.x_train, self._data_properties.x_test = remove_duplicate_rows(x_train=self._data_properties.x_train,
                                                                                            x_test=self._data_properties.x_test,
                                                                                            list_of_cols=list_of_cols,
                                                                                            )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()


    def remove_duplicate_columns(self):
        """
        Remove columns from the data that are exact duplicates of each other and leave only 1.
        
        Returns
        -------
        Clean:
            Returns a deep copy of the Clean object.
        """
    
        report_info = technique_reason_repo['clean']['general']['remove_duplicate_columns']
    
        self._data_properties.x_train, self._data_properties.x_test = remove_duplicate_columns(x_train=self._data_properties.x_train,
                                                                                                x_test=self._data_properties.x_test)

        if self.report is not None:
            self.report.report_technique(report_info)

        return self.copy()


    def replace_missing_random_discrete(self, *list_args, list_of_cols=[]):
        """
        Replace missing values in with a random number based off the distribution (number of occurences) 
        of the data.

        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
            
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Clean:
            Returns a deep copy of the Clean object.

        Examples
        --------
        >>> For example if your data was [5, 5, NaN, 1, 2]
        >>> There would be a 50% chance that the NaN would be replaced with a 5, a 25% chance for 1 and a 25% chance for 2.
        """
    
        report_info = technique_reason_repo['clean']['general']['random_discrete']
        
        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)
        
        self._data_properties.x_train, self._data_properties.x_test = replace_missing_random_discrete(x_train=self._data_properties.x_train,
                                                                                                    x_test=self._data_properties.x_test,
                                                                                                    list_of_cols=list_of_cols,
                                                                                                    )
    
        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()
