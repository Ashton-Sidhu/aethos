import os
import unittest

from report import Report


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

if __name__ == "__main__":
    unittest.main()
