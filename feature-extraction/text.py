import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from base import FeatureBase
from data.util import DropAndReplaceColumns, GetListOfCols


class FeatureText(FeatureBase):

    def FeatureBagOfWords(self, custom_cols=[], override=False):

        enc = CountVectorizer()
        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "text")
        train_data = self.data_properties.train_data
        test_data = self.data_properties.test_data

        if self.data_properties.use_full_dataset:
            enc_data = enc.fit_transform(self.df[list_of_cols])
            enc_df = pd.DataFrame(enc_data, columns=enc_data.get_feature_names)
            self.df = DropAndReplaceColumns(self.df, list_of_cols, enc_df)

        else:
            enc_train_data = enc.fit_transform(train_data[list_of_cols])
            enc_train_df = pd.DataFrame(enc_train_data, columns=enc_train_data.get_feature_names)
            self.data_properties.train_data = DropAndReplaceColumns(train_data, list_of_cols, enc_train_df)

            enc_test_data = enc.transform(test_data[list_of_cols])
            enc_test_df = pd.DataFrame(enc_test_data, columns=enc_test_data.get_features_names)
            self.data_properties.test_data = DropAndReplaceColumns(test_data, list_of_cols, enc_test_df)
