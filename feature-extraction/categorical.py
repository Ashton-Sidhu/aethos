import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from base import FeatureBase
from data.util import DropAndReplaceColumns, GetListOfCols


class FeatureCategorical(FeatureBase):

    def FeatureOneHotEncode(self, handle_unknown='ignore', custom_cols=[], override=False):
        """[summary]
        
        Keyword Arguments:
            handle_unknown {str} -- Parameter to pass into OneHotEncoder constructor to specify how to deal with values
            it has not seen before. (default: {'ignore'})
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                  Example: if custom_cols is provided and override is true, the technique will only be applied
                                  to the the columns in custom_cols (default: {False})
        """

        enc = OneHotEncoder(handle_unknown=handle_unknown)
        list_of_cols = GetListOfCols("categorical", self.data_properties.field_types, override, custom_cols)
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
