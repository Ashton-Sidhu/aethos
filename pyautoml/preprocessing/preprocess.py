import pandas as pd
from pyautoml import *
from pyautoml.base import MethodBase
from pyautoml.config import technique_reason_repo
from pyautoml.preprocessing.categorical import *
from pyautoml.preprocessing.numeric import *
from pyautoml.preprocessing.text import *
from pyautoml.util import _input_columns, _numeric_input_conditions, label_encoder


class Preprocess(MethodBase):
    def __init__(
        self,
        x_train=None,
        x_test=None,
        test_split_percentage=0.2,
        split=True,
        target_field="",
        report_name=None,
    ):

        if isinstance(x_train, pd.DataFrame):
            super().__init__(
                x_train=x_train,
                x_test=x_test,
                split=split,
                target_field=target_field,
                target_mapping=None,
                report_name=report_name,
                test_split_percentage=test_split_percentage,
            )
        else:
            step = x_train
            super().__init__(
                x_train=step.x_train,
                x_test=step.x_test,
                test_split_percentage=step.test_split_percentage,
                split=step.split,
                target_field=step.target_field,
                target_mapping=step.target_mapping,
                report_name=step.report_name,
            )

        if self.report is not None:
            self.report.write_header("Preprocessing")

    def normalize_numeric(
        self, *list_args, list_of_cols=[], keep_col=True, **normalize_params
    ):
        """
        Function that normalizes all numeric values between 2 values to bring features into same domain.
        
        If `list_of_cols` is not provided, the strategy will be applied to all numeric columns.

        If a list of columns is provided use the list, otherwise use arguments.

        For more info please see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler 

        This function can be found in `preprocess/numeric.py`     
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        feature_range : tuple(int or float, int or float), optional
            Min and max range to normalize values to, by default (0, 1)

        normalize_params : dict, optional
            Parmaters to pass into MinMaxScaler() constructor from Scikit-Learn
        
        Returns
        -------
        Preprocess:
            Returns a deep copy of the Preprocess object.
        """

        report_info = technique_reason_repo["preprocess"]["numeric"]["standardize"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train, self.x_test = scale(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            method="minmax",
            keep_col=keep_col,
            **normalize_params,
        )

        if self.report is not None:
            if list_of_cols:
                self.report.report_technique(report_info, list_of_cols)
            else:
                list_of_cols = _numeric_input_conditions(list_of_cols, self.x_train)
                self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def normalize_quantile_range(
        self, *list_args, list_of_cols=[], keep_col=True, **robust_params
    ):
        """
        Scale features using statistics that are robust to outliers.

        This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range).
        The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

        Standardization of a dataset is a common requirement for many machine learning estimators.
        Typically this is done by removing the mean and scaling to unit variance.
        However, outliers can often influence the sample mean / variance in a negative way.
        In such cases, the median and the interquartile range often give better results.
        
        If `list_of_cols` is not provided, the strategy will be applied to all numeric columns.

        If a list of columns is provided use the list, otherwise use arguments.

        For more info please see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler

        This function can be found in `preprocess/numeric.py`     
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        with_centering : boolean, True by default
            If True, center the data before scaling.
            This will cause transform to raise an exception when attempted on sparse matrices,
            because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.
        
        with_scaling : boolean, True by default
            If True, scale the data to interquartile range.

        quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
            Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR Quantile range used to calculate scale_.

        robust_params : dict, optional
            Parmaters to pass into MinMaxScaler() constructor from Scikit-Learn
        
        Returns
        -------
        Preprocess:
            Returns a deep copy of the Preprocess object.
        """

        report_info = technique_reason_repo["preprocess"]["numeric"]["robust"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train, self.x_test = scale(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            method="robust",
            keep_col=keep_col,
            **robust_params,
        )

        if self.report is not None:
            if list_of_cols:
                self.report.report_technique(report_info, list_of_cols)
            else:
                list_of_cols = _numeric_input_conditions(list_of_cols, self.x_train)
                self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def normalize_log(self, *list_args, list_of_cols=[], base=1):
        """
        Scales data logarithmically.

        Options are 1 for natural log, 2 for base2, 10 for base10.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        base : str, optional
            Base to logarithmically scale by, by default ''
        
        Returns
        -------
        Preprocess:
            Returns a deep copy of the Preprocess object.
        """

        report_info = technique_reason_repo["preprocess"]["numeric"]["log"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train, self.x_test = log_scale(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            base=base,
        )

        if self.report is not None:
            if list_of_cols:
                self.report.report_technique(report_info, list_of_cols)
            else:
                list_of_cols = _numeric_input_conditions(list_of_cols, self.x_train)
                self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def split_sentences(self, *list_args, list_of_cols=[], new_col_name="_sentences"):
        """
        Splits text data into sentences and saves it into another column for analysis.

        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_sentences`

        Returns
        -------
        Preprocess:
            Returns a deep copy of the Preprocess object.
        """

        report_info = technique_reason_repo["preprocess"]["text"]["split_sentence"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train, self.x_test = split_sentences(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def stem_nltk(
        self, *list_args, list_of_cols=[], stemmer="porter", new_col_name="_stemmed"
    ):
        """
        Transforms text to their word stem, base or root form. 
        For example:
            dogs --> dog
            churches --> church
            abaci --> abacus
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        stemmer : str, optional
            Type of NLTK stemmer to use, by default porter

            Current stemming implementations:
                - Porter

            For more information please refer to the NLTK stemming api https://www.nltk.org/api/nltk.stem.html

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_stemmed`
        
        Returns
        -------
        Preprocess
            Copy of preprocess object
        """

        report_info = technique_reason_repo["preprocess"]["text"]["stem"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train, self.x_test = nltk_stem(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            stemmer=stemmer,
            new_col_name=new_col_name,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def split_words_nltk(
        self, *list_args, list_of_cols=[], regexp="", new_col_name="_tokenized"
    ):
        """
        Splits text into its words using nltk punkt tokenizer by default. 
    
        Default is by spaces and punctuation but if a regex expression is provided, it will use that.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        regexp : str, optional
            Regex expression used to define what a word is.

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_tokenized`
        
        Returns
        -------
        Preprocess
            Copy of preprocess object
        """

        report_info = technique_reason_repo["preprocess"]["text"]["split_words"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = nltk_word_tokenizer(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            regexp=regexp,
            new_col_name=new_col_name,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def remove_stopwords_nltk(
        self, *list_args, list_of_cols=[], custom_stopwords=[], new_col_name="_rem_stop"
    ):
        """
        Removes stopwords following the nltk English stopwords list.

        A list of custom words can be provided as well, usually for domain specific words.

        Stop words are generally the most common words in a language
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        custom_stop_words : list, optional
            Custom list of words to also drop with the stop words, must be LOWERCASE, by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_rem_stop`
        
        Returns
        -------
        Preprocess
            Copy of preprocess object
        """

        report_info = technique_reason_repo["preprocess"]["text"]["remove_stopwords"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = nltk_remove_stopwords(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            custom_stopwords=custom_stopwords,
            new_col_name=new_col_name,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def remove_punctuation(
        self,
        *list_args,
        list_of_cols=[],
        regexp="",
        exceptions=[],
        new_col_name="_rem_punct",
    ):
        """
        Removes punctuation from every string entry.

        Defaults to removing all punctuation, but if regex of punctuation is provided, it will remove them.
        
        An example regex would be:

        (\w+\.|\w+)[^,] - Include all words and words with periods after them but don't include commas.
        (\w+\.)|(\w+), would also achieve the same result

        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []

        regexp : str, optional
            Regex expression used to define what to include.
            
        exceptions : list, optional
            List of punctuation to include in the text, by default []

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_rem_punct`
        
        Returns
        -------
        Preprocess
            Copy of preprocess object
        """

        report_info = technique_reason_repo["preprocess"]["text"]["remove_punctuation"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = remove_punctuation(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            regexp=regexp,
            exceptions=exceptions,
            new_col_name=new_col_name,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()

    def encode_labels(self, *list_args, list_of_cols=[]):
        """
        Encode labels with value between 0 and n_classes-1.

        Running this function will automatically set the corresponding mapping for the target variable mapping number to the original value.

        Note that this will not work if your test data will have labels that your train data does not.

        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Preprocess
            Copy of preprocess object
        """

        report_info = technique_reason_repo["preprocess"]["categorical"]["label_encode"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train, self.x_test, _ = label_encoder(
            x_train=self.x_train, x_test=self.x_test, list_of_cols=list_of_cols,
        )

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()
