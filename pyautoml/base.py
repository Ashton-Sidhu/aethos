import pandas as pd
from IPython.display import display
from pyautoml.data.data import Data
from pyautoml.util import SplitData, _FunctionInputValidation


class MethodBase():

    def __init__(self, **kwargs):

        data = kwargs.pop('data')
        train_data = kwargs.pop('train_data')
        test_data = kwargs.pop('test_data')
        use_full_data = kwargs.pop('use_full_data')
        target_field = kwargs.pop('target_field')
        report_name = kwargs.pop('report_name')
        test_split_percentage = kwargs.pop('test_split_percentange')

        if not _FunctionInputValidation(data, train_data, test_data):
            raise ValueError("Error initialzing constructor, please provide one of either data or train_data and test_data, not both.")

        self.data_properties = Data(data, train_data, test_data, use_full_data=use_full_data, target_field=target_field, report_name=report_name)

        if data is not None:
            self.data_properties.data = data
            self.data_properties.train_data, self.data_properties.test_data = SplitData(self.data_properties.data, test_split_percentage)
        else:
            # Override user input for safety.
            self.data_properties.use_full_data = False       

        if self.data_properties.report is None:
            self.report = None
        else:
            self.report = self.data_properties.report
    
    def __repr__(self):

        if self.data_properties.data is not None:
            return display(self.data_properties.data)
        else:
            return display(self.train_data)


    @property
    def data(self):
        """
        Property function for the entire dataset.
        """

        return self.data_properties.data

    @property
    def train_data(self):
        """
        Property function for the training dataset.
        """

        return self.data_properties.train_data

        
    @property
    def test_data(self):
        """
        Property function for the entire dataset.
        """

        return self.data_properties.test_data

    @property
    def missing_data():
        """
        Property function that shows how many values are missing in each column.
        """

        dataframes = list(filter(lambda x: x is not None, [self.data_properties.data, self.data_properties.train_data, self.data_properties.test_data]))
        
        for dataframe in dataframes:
            if not dataframe.isnull().values.any():            
                print("No missing values!")
            else:
                total = dataframe.isnull().sum().sort_values(ascending=False)
                percent = (dataframe.isnull().sum()/dataframe.isnull().count()).sort_values(ascending=False)
                missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

                display(missing_data)
