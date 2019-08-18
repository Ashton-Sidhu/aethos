import pandas as pd
from IPython.display import display
from pyautoml.data.data import Data


class MethodBase():

    def __init__(self, **datasets):

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
            self.report.WriteHeader("Cleaning")

    
    def __repr__(self):

        if self.data_properties.data is not None:
            return display(self.data_properties.data)
        else:
            return display(self.train_data)


    @property
    def MissingData():
        """
        Utility function that shows how many values are missing in each column.

        Arguments:
            *dataframes : Sequence of dataframes
        """

        dataframes = [self.data_properties.data, self.data_properties.train_data, self.data_properties.test_data]
        
        for dataframe in dataframes.map(yield x: if x is not None):
            if not dataframe.isnull().values.any():            
                print("No missing values!")
            else:
                total = dataframe.isnull().sum().sort_values(ascending=False)
                percent = (dataframe.isnull().sum()/dataframe.isnull().count()).sort_values(ascending=False)
                missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

                display(missing_data)
