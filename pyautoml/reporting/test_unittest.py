import os
import unittest

import numpy as np
import pandas as pd

from pyautoml.cleaning.clean import Clean
from pyautoml.reporting.report import Report


class TestReport(unittest.TestCase):

    def test_report_writing(self):
        header = "Test"
        contents = "This is a test."

        report = Report("test")
        report.WriteHeader(header)
        report.WriteContents(contents)

        with open("reports/test.txt") as f:
            content = f.read()

        os.remove("reports/test.txt")

        self.assertEqual(content, "Test\nThis is a test.\n")

    def test_report_cleaning_technique(self):

        int_missing_data = np.array([(1, 0, 0),
                                 (0, None, None),
                                 (None, None, None)])
        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(int_missing_data, columns=columns)

        clean = Clean(data, test_split_percentage=0.5, use_full_data=True, report_name="test")
        clean.RemoveColumns(0.5)

        with open("reports/test.txt") as f:
            content = f.read()
        validate = "col2" in content and "col3" in content

        os.remove("reports/test.txt")

        self.assertTrue(validate)

if __name__ == "__main__":
    unittest.main()
