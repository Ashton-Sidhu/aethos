
from pyautoml import base


class Model(MethodBase):

    def __init__(self, step=None, data=None, train_data=None, test_data=None, test_split_percentage=0.2, split=True, target_field="", report_name=None):
        
        _data_properties = _contructor_data_properties(step)

        if _data_properties is None:        
            super().__init__(data=data, train_data=train_data, test_data=test_data, test_split_percentage=test_split_percentage,
                        split=split, target_field=target_field, report_name=report_name)
        else:
            super().__init__(data=_data_properties.data, train_data=_data_properties.train_data, test_data=_data_properties.test_data, test_split_percentage=test_split_percentage,
                        split=_data_properties.split, target_field=_data_properties.target_field, report_name=_data_properties.report_name)
                        
        if self._data_properties.report is not None:
            self.report.write_header("Feature Engineering")

        if target_field:
            if split:
                self._train_target_data = self._data_properties.train_data[self._data_properties.target_field]
                self._test_target_data = self._data_properties.test_data[self._data_properties.test_field]
                self._data_properties.train_data = self._data_properties.train_data.drop([self._data_properties.target_field], axis=1)
                self._data_properties.test_data = self._data_properties.test_data.drop([self._data_properties.target_field], axis=1)
            else:
                self._target_data = self._data_properties.data[self._data_properties.target_field]
                self._data_properties.data = self._data_properties.data.drop([self._data_properties.target_field], axis=1)

    @property
    def target_data(self):
        """
        Property function for the target data.
        """
        
        if self._data_properties.data is None:
            raise AttributeError("There seems to be nothing here. Try .train_data or .test_data")
        
        return self._target_data

    @target_data.setter
    def target_data(self, value):
        """
        Setter function for the target data.
        """

        self._target_data = value


    @property
    def train_target_data(self):
        """
        Property function for the training target data.
        """
        
        if self._data_properties.train_data is None:
            raise AttributeError("There seems to be nothing here. Try .data")

        return self._train_target_data

    @train_target_data.setter
    def train_target_data(self, value):
        """
        Setter function for the training target data.
        """

        self._train_target_data = value
        
    @property
    def test_target_data(self):
        """
        Property function for the test target data.
        """
        if self._data_properties.train_data is None:
            raise AttributeError("There seems to be nothing here. Try .data")

        return self._test_target_data

    @test_data.setter
    def test_target_data(self, value):
        """
        Setter for the test target data.
        """

        self._test_target_data = value
