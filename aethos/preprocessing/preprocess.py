import pandas as pd
from aethos import *
from aethos.config import technique_reason_repo
from aethos.preprocessing.categorical import *
from aethos.preprocessing.numeric import *
from aethos.preprocessing.text import *
from aethos.util import _input_columns, _numeric_input_conditions, label_encoder


class Preprocess(object):
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
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.normalize_numeric('col1')
        >>> data.normalize_numeric(['col1', 'col2'])
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
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.normalize_quantile_range('col1')
        >>> data.normalize_quantile_range(['col1', 'col2'])
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
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.normalize_log('col1')
        >>> data.normalize_log(['col1', 'col2'], base=10)
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
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.split_sentences('col1')
        >>> data.split_sentences(['col1', 'col2'])
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
                - porter
                - snowball

            For more information please refer to the NLTK stemming api https://www.nltk.org/api/nltk.stem.html

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_stemmed`
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.stem_nltk('col1')
        >>> data.stem_nltk(['col1', 'col2'], stemmer='snowball')
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
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.split_words_nltk('col1')
        >>> data.split_words_nltk(['col1', 'col2'])
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
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.remove_stopwords_nltk('col1')
        >>> data.remove_stopwords_nltk(['col1', 'col2'])
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
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.remove_punctuation('col1')
        >>> data.remove_punctuation(['col1', 'col2'])
        >>> data.remove_punctuation('col1', regexp=r'(\w+\.)|(\w+)') # Include all words and words with periods after.
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

    def clean_text(
        self,
        *list_args,
        list_of_cols=[],
        lower=True,
        punctuation=True,
        stopwords=True,
        stemmer=True,
        new_col_name="_clean",
    ):
        """
        Function that takes text and does the following:

        - Casts it to lowercase
        - Removes punctuation
        - Removes stopwords
        - Stems the text
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.

        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        lower : bool, optional
            True to cast all text to lowercase, by default True
        
        punctuation : bool, optional
            True to remove punctuation, by default True
        
        stopwords : bool, optional
            True to remove stop words, by default True
        
        stemmer : bool, optional
            True to stem the data, by default True

        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_clean`            
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.clean_text('col1')
        >>> data.clean_text(['col1', 'col2'], lower=False)
        >>> data.clean_text(lower=False, stopwords=False, stemmer=False)
        """

        report_info = technique_reason_repo["preprocess"]["text"]["clean_text"]

        list_of_cols = _input_columns(list_args, list_of_cols)

        for col in list_of_cols:
            self.x_train[col + new_col_name] = [
                process_text(text) for text in self.x_train[col]
            ]

            if self.x_test is not None:
                self.x_test[col + new_col_name] = [
                    process_text(text) for text in self.x_test[col]
                ]

        if self.report is not None:
            self.report.report_technique(report_info, list_of_cols)

        return self.copy()
