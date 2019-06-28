from sklearn.preprocessing import MinMaxScaler

from base import PreprocessBase
from data.util import GetListOfCols


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
        list_of_cols = GetListOfCols(custom_cols, self.data_properties.field_types, override, "numeric")
        
        if self.data_properties.use_full_dataset:
            scaled_data = scaler.fit_transform(self.df[list_of_cols])
            self.df.drop(self.df[list_of_cols], inplace=True)
            self.df.concat(scaled_data, axis=1, inplace=True)
        else:
            scaled_train_data = scaler.fit_transform(self.data_properties.train_data)
            self.data_properties.train_data(self.data_properties.train_data[list_of_cols], inplace=True)
            self.data_properties.train_data.concat(scaled_train_data, axis=1, inplace=True)

            scaled_test_data = scaler.transform(self.data_properties.test_data)
            self.data_properties.test_data(self.data_properties.test_data[list_of_cols], inplace=True)
            self.data_properties.test_data.concat(scaled_test_data, axis=1, inplace=True)
