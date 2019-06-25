import pandas as pd

from numeric import *
from utils import *


class Clean(CleanUtil, CleanNumeric):

    def __init__(self, data):
        CleanBase.__init__(self, data)

    def RemoveColumns(self, threshold):
        """Removed columns from the dataframe that have more than the threshold value of missing columns.
        Example: Remove columns where > 50% of the data is missing
        
        Arguments:
            threshold {[float]} -- Value between 0 and 1 that describes what percentage of a column can be missing values.
        """
        
        criteria_meeting_columns = self.df.columns[self.df.isnull().mean() < threshold]
        self.df = self.df[criteria_meeting_columns]      

    def CleanMissingData(self, custom_cols={}, analysis="time_agnostic"):
        return

    def CleanData(self):
        return

    def GenerateCode(self):
        return
