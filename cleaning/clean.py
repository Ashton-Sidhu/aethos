import pandas as pd

from categorical import CleanCategorical
from numeric import CleanNumeric
from text import CleanText


class Clean(CleanNumeric, CleanCategorical, CleanText):

    def __init__(self, data, target_field=""):
        CleanBase.__init__(self, data, target_field)

    def RemoveColumns(self, threshold):
        """Remove columns from the dataframe that have more than the threshold value of missing columns.
        Example: Remove columns where > 50% of the data is missing
        
        Arguments:
            threshold {[float]} -- Value between 0 and 1 that describes what percentage of a column can be missing values.
        """
        
        criteria_meeting_columns = self.df.columns[self.df.isnull().mean() < threshold]
        self.df = self.df[criteria_meeting_columns]

    def RemoveRows(self, threshold):
        """Remove rows from the dataframe that have more than the threshold value of missing rows.
        Example: Remove rows where > 50% of the data is missing
        
        Arguments:
            threshold {[float]} -- Value between 0 and 1 that describes what percentage of a row can be missing values.
        """

        self.df.dropna(thresh=int(self.df.shape[1] * threshold), inplace=True, axis=0)    

    def GenerateCode(self):
        return
