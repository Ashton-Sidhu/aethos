import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from base import FeatureBase
from data.util import DropAndReplaceColumns, GetListOfCols

#TODO: Add customization to BoW and TF-IDF through the parameters of the constructor

class FeatureText(FeatureBase):

    def FeatureBagOfWords(self, custom_cols=[], override=False):

        enc = CountVectorizer()
        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "text")
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

        enc = TfidfTransformer()
        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "text")
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
