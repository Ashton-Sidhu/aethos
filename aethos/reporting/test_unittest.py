import os
import unittest

import numpy as np
import pandas as pd

from aethos import Data
from aethos.reporting.report import Report


class TestReport(unittest.TestCase):
    def test_report_writing(self):
        header = "Test"
        contents = "This is a test."

        report = Report("test")
        report.write_header(header)
        report.write_contents(contents)

        with open(report.filename) as f:
            content = f.read()

        os.remove(report.filename)

        self.assertTrue(True)

    def test_report_cleaning_technique(self):

        int_missing_data = np.array([(1, 0, 0), (0, None, None), (None, None, None)])
        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean.drop_column_missing_threshold(0.5)

        with open(clean.report.filename) as f:
            content = f.read()
        validate = "col2" in content and "col3" in content

        os.remove(clean.report.filename)

        self.assertTrue(validate)

    def test_report_cleaning_new_category(self):

        missing_data = [[1.0, "Green", 2], [1.0, "Other", 1], [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(missing_data, columns=columns)

        clean = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        clean_data = clean.replace_missing_new_category()

        with open(clean.report.filename) as f:
            content = f.read()
        validate = "col1" in content and "col2" in content and "col3" in content

        os.remove(clean.report.filename)

        self.assertTrue(validate)

    def test_report_preprocessing_standardize(self):

        unnormal_data = [[5.0, 3, 1], [2.0, 2, 1], [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(unnormal_data, columns=columns)

        preprocess = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        preprocess.normalize_numeric()

        with open(preprocess.report.filename) as f:
            content = f.read()
        validate = "col1" in content and "col2" in content and "col3" in content

        os.remove(preprocess.report.filename)

        self.assertTrue(validate)

    def test_report_feature_bow(self):

        list_of_sentences = ["Hi my name is pyml", "Hi name pyml"]

        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        feature.bag_of_words()

        with open(feature.report.filename) as f:
            content = f.read()
        validate = "representation" in content

        os.remove(feature.report.filename)

        self.assertTrue(validate)


if __name__ == "__main__":
    unittest.main()
