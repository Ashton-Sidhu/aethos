import pandas as pd

from utils import *


class Clean(CleanUtil):

    def __init__(self, data):
        CleanBase.__init__(self, data)        

    def CleanMissingData(self, custom_cols={}, analysis="time_agnostic"):
        return

    def CleanData(self):
        return

    def GenerateCode(self):
        return


#For testing
def main():
    df = pd.read_csv("./datasets/train.csv")

    clean = Clean(df)
    clean.NormalizeColNames()
    clean.GetInputTypes()
    clean.ReduceData()
    print(clean.df.head())


if __name__ == "__main__":
    main()
