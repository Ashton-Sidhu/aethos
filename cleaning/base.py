from data.data import Data


class CleanBase():

    #TODO: Create Cleaning test cases
    
    def __init__(self, data, target_field):
        
        self.data_properties = Data(data, target_field)
        self.df = data_properties.StandardizeData(data)
