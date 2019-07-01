from sklearn.preprocessing import MinMaxScaler

from base import PreprocessBase
from data.util import DropAndReplaceColumns, GetListOfCols


class PreprocessNumeric(PreprocessBase):

    def PreprocessNormalize(self, custom_cols=[], override=False):
        """Function that normalizes all numeric values between 0 and 1 to bring features into same domain.
        
        Keyword Arguments:
            use_full_dataset {bool} -- True if you want to scale the your data by the entire dataset and not training data (default: {False})
            custom_cols {list} -- A list of specific columns to apply this technique to. (default: {[]})
            override {boolean} -- True or False depending on whether the custom_cols overrides the columns in field_types
                                  Example: if custom_cols is provided and override is true, the technique will only be applied
                                  to the the columns in custom_cols (default: {False})
        """
        scaler = MinMaxScaler()
        list_of_cols = GetListOfCols("numeric", self.data_properties.field_types, override, custom_cols)
        train_data = self.data_properties.train_data
        test_data = self.data_properties.test_data

        if self.data_properties.use_full_dataset:
            scaled_data = scaler.fit_transform(self.df[list_of_cols])
            self.df = DropAndReplaceColumns(self.df, list_of_cols, scaled_data)
        
        else:
            scaled_train_data = scaler.fit_transform(train_data)
            self.data_properties.train_data = DropAndReplaceColumns(train_data, list_of_cols, scaled_train_data)

            scaled_test_data = scaler.transform(self.data_properties.test_data)
            self.data_properties.test_data = DropAndReplaceColumns(test_data, list_of_cols, scaled_test_data)
