class PreprocessingBase():

    def __init__(self, data_properties, data, target_field):

        #If a user is starting from the preprocessing stage (i.e all their data has been cleaned) |
        #  perform data standardization
        if data_properties is None:
            self.data_properties = Data(data, target_field)
            self.df = data_properties.StandardizeData(data)

        else:
            self.data_properties = data_properties
            self.df = data
