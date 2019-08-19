import os
import unittest

import numpy as np
import pandas as pd
from pyautoml import Clean, Feature, Preprocess
from pyautoml.reporting.report import Report


class TestReport(unittest.TestCase):

    def test_report_writing(self):
        header = "Test"
        contents = "This is a test."

        report = Report("test")
        report.write_header(header)
        report.write_contents(contents)

        with open("pyautoml_reports/test.txt") as f:
            content = f.read()

        os.remove("pyautoml_reports/test.txt")

        self.assertEqual(content, "Test\n\nThis is a test.")

    def test_report_cleaning_technique(self):

        int_missing_data = np.array([(1, 0, 0),
                                 (0, None, None),
                                 (None, None, None)])
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Clean(data, test_split_percentage=0.5, use_full_data=True, report_name="test")
        clean.remove_columns(0.5)

        with open("pyautoml_reports/test.txt") as f:
            content = f.read()
        validate = "col2" in content and "col3" in content

        os.remove("pyautoml_reports/test.txt")

        self.assertTrue(validate)

    def test_report_cleaning_new_category(self):

        missing_data = [[1.0, "Green", 2],
                        [1.0, "Other", 1],
                        [np.nan, np.nan, np.nan]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(missing_data, columns=columns)

        clean = Clean(data, test_split_percentage=0.5, use_full_data=True, report_name="test")
        clean_data = clean.replace_missing_new_category()

        with open("pyautoml_reports/test.txt") as f:
            content = f.read()
        validate = "col1" in content and "col2" in content and "col3" in content

        os.remove("pyautoml_reports/test.txt")

        self.assertTrue(validate)

    def test_report_preprocessing_standardize(self):

        unnormal_data = [[5.0, 3, 1],
                        [2.0, 2, 1],
                        [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(unnormal_data, columns=columns)

        preprocess = Preprocess(data, test_split_percentage=0.5, use_full_data=False, report_name="test")
        normal_data, test = preprocess.normalize_numeric()
        
        with open("pyautoml_reports/test.txt") as f:
            content = f.read()
        validate = "col1" in content and "col2" in content and "col3" in content

        os.remove("pyautoml_reports/test.txt")

        self.assertTrue(validate)

    def test_report_feature_bow(self):

        list_of_sentences = ['Hi my name is pyml',
                            'Hi name pyml']

        feature = Feature(list_of_sentences, test_split_percentage=0.5, use_full_data=True, report_name="test")
        transform_data = feature.bag_of_words()

        with open("pyautoml_reports/test.txt") as f:
            content = f.read()
        validate = "representation" in content

        os.remove("pyautoml_reports/test.txt")

        self.assertTrue(validate)

if __name__ == "__main__":
    unittest.main()
