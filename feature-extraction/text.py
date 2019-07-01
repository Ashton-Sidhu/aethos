import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from base import FeatureBase
from data.util import DropAndReplaceColumns, GetListOfCols

#TODO: Add customization to BoW and TF-IDF through the parameters of the constructor

class FeatureText(FeatureBase):

    def FeatureBagOfWords(self, custom_cols=[], override=False):
        """Creates a matrix of how many times a word appears in a document.
        
        Keyword Arguments:
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                  Example: if custom_cols is provided and override is true, the technique will only be applied
                                  to the the columns in custom_cols (default: {False})
        """

        enc = CountVectorizer()
        list_of_cols = GetListOfCols("text", self.data_properties.field_types, override, custom_cols)
        train_data = self.data_properties.train_data
        test_data = self.data_properties.test_data

        if self.data_properties.use_full_dataset:
            for col in list_of_cols:
                enc_data = enc.fit_transform(self.df[col])
                enc_df = pd.DataFrame(enc_data, columns=enc_data.get_feature_names)
                self.df = DropAndReplaceColumns(self.df, col, enc_df)

        else:
            for col in list_of_cols:
                enc_train_data = enc.fit_transform(train_data[col])
                enc_train_df = pd.DataFrame(enc_train_data, columns=enc_train_data.get_feature_names)
                self.data_properties.train_data = DropAndReplaceColumns(train_data, col, enc_train_df)

                enc_test_data = enc.transform(test_data[col])
                enc_test_df = pd.DataFrame(enc_test_data, columns=enc_test_data.get_features_names)
                self.data_properties.test_data = DropAndReplaceColumns(test_data, col, enc_test_df)

    
    def FeatureTFIDF(self, custom_cols=[], override=False):
        """Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.
        
        Keyword Arguments:
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                  Example: if custom_cols is provided and override is true, the technique will only be applied
                                  to the the columns in custom_cols (default: {False})
        """

        enc = TfidfTransformer()
        list_of_cols = GetListOfCols("text", self.data_properties.field_types, override, custom_cols)
        train_data = self.data_properties.train_data
        test_data = self.data_properties.test_data

        if self.data_properties.use_full_dataset:
            for col in list_of_cols:
                enc_data = enc.fit_transform(self.df[col])
                enc_df = pd.DataFrame(enc_data, columns=enc_data.get_feature_names)
                self.df = DropAndReplaceColumns(self.df, col, enc_df)

        else:
            for col in list_of_cols:
                enc_train_data = enc.fit_transform(train_data[col])
                enc_train_df = pd.DataFrame(enc_train_data, columns=enc_train_data.get_feature_names)
                self.data_properties.train_data = DropAndReplaceColumns(train_data, col, enc_train_df)

                enc_test_data = enc.transform(test_data[list_of_cols])
                enc_test_df = pd.DataFrame(enc_test_data, columns=enc_test_data.get_features_names)
                self.data_properties.test_data = DropAndReplaceColumns(test_data, col, enc_test_df)
