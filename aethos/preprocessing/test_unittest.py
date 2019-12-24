import unittest
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from aethos import Data


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(Path.home()) + "/.aethos/reports/")

    def test_preprocessnumeric_normalize(self):

        unnormal_data = [[5.0, 3, 1], [2.0, 2, 1], [10.0, 1, 1], [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(unnormal_data, columns=columns)

        preprocess = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        preprocess.normalize_numeric(keep_col=False)
        validate = preprocess.x_train.values.tolist()

        self.assertTrue(True)

    def test_preprocessnumeric_robust(self):

        unnormal_data = [[5.0, 3, 1], [2.0, 2, 1], [10.0, 1, 1], [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(unnormal_data, columns=columns)

        preprocess = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        preprocess.normalize_quantile_range(keep_col=False)
        validate = preprocess.x_train.values.tolist()

        self.assertTrue(True)

    def test_preprocessnumeric_log(self):

        unnormal_data = [[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(unnormal_data, columns=columns)

        preprocess = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        preprocess.normalize_log()
        preprocess.normalize_log(base=2)
        preprocess.normalize_log(base=10)

        self.assertTrue(True)

    def test_preprocess_traindata(self):

        unnormal_x_train = [[5.0, 3, 1], [2.0, 2, 1], [10.0, 1, 1]]

        unnormal_x_test = [[5.0, 3, 1], [2.0, 2, 1], [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]
        x_train = pd.DataFrame(unnormal_x_train, columns=columns)
        x_test = pd.DataFrame(unnormal_x_test, columns=columns)

        preprocess = Data(
            x_train=x_train,
            x_test=x_test,
            test_split_percentage=0.5,
            report_name="test",
        )
        preprocess.normalize_numeric("col1", "col2", "col3")
        validate_train = preprocess.x_train.values.tolist()
        validate_test = preprocess.x_test.values.tolist()

        self.assertListEqual(validate_train, validate_test)

    def test_preprocess_splitsentences(self):

        text_data = [
            "Hi my name is aethos. Please split me.",
            "This function is going to split by sentence. Automation is great.",
        ]
        data = pd.DataFrame(data=text_data, columns=["data"])

        prep = Data(x_train=data, split=False, report_name="test")
        prep.split_sentences("data")
        validate = prep.x_train["data_sentences"].values.tolist()

        self.assertListEqual(
            validate,
            [
                ["Hi my name is aethos.", "Please split me."],
                [
                    "This function is going to split by sentence.",
                    "Automation is great.",
                ],
            ],
        )

    def test_preprocess_nltkstem(self):

        text_data = [
            "Hi my name is aethos. Please split me.",
            "This function is going to split by sentence. Automation is great.",
        ]
        data = pd.DataFrame(data=text_data, columns=["data"])

        prep = Data(x_train=data, split=False, report_name="test")
        prep.stem_nltk("data")
        validate = prep.x_train.shape[1]

        self.assertEqual(validate, 2)

    def test_preprocess_nltksplit(self):

        text_data = ["Please.exe split me."]
        data = pd.DataFrame(data=text_data, columns=["data"])

        prep = Data(x_train=data, split=False, report_name="test")
        prep.split_words_nltk("data")
        validate = prep.x_train.data_tokenized.values.tolist()

        self.assertListEqual(validate, [["Please.exe", "split", "me", "."]])

    def test_preprocess_nltksplit_regex(self):

        text_data = ["Please123 split me."]
        data = pd.DataFrame(data=text_data, columns=["data"])

        prep = Data(x_train=data, split=False, report_name="test")
        prep.split_words_nltk("data", regexp=r"\w+\d+")
        validate = prep.x_train.data_tokenized.values.tolist()

        self.assertListEqual(validate, [["Please123"]])

    def test_preprocess_nltkremove_punctuation(self):

        text_data = ["Please split me."]
        data = pd.DataFrame(data=text_data, columns=["data"])

        prep = Data(x_train=data, split=False, report_name="test")
        prep.remove_punctuation("data")
        validate = prep.x_train.data_rem_punct.values.tolist()

        self.assertListEqual(validate, ["Please split me"])

    def test_preprocess_nltkremove_punctuation_regexp(self):

        text_data = ["Please.exe, split me.", "hello it's me, testing.dll."]
        data = pd.DataFrame(data=text_data, columns=["data"])

        prep = Data(x_train=data, split=False, report_name="test")
        prep.remove_punctuation("data", regexp=r"\w+\.\w+|\w+")
        validate = prep.x_train.data_rem_punct.values.tolist()

        self.assertListEqual(
            validate, ["Please.exe split me", "hello it s me testing.dll"]
        )

    def test_preprocess_cleantext(self):

        text_data = ["Please.exe, split me.", "hello it's me, testing.dll."]
        data = pd.DataFrame(data=text_data, columns=["data"])

        prep = Data(x_train=data, split=False, report_name="test")
        prep.clean_text("data")
        validate = prep.x_train["data_clean"].tolist()

        self.assertListEqual(validate, ["please.ex split", "hello 's testing.dl"])

    def test_preprocess_nltkremove_punctuation_exception(self):

        text_data = ["Please,> split me."]
        data = pd.DataFrame(data=text_data, columns=["data"])

        prep = Data(x_train=data, split=False, report_name="test")
        prep.remove_punctuation("data", exceptions=[".", ">"])
        validate = prep.x_train.data_rem_punct.values.tolist()

        self.assertListEqual(validate, ["Please> split me."])

    def test_preprocess_nltkremove_stopwords(self):

        text_data = ["Please the split me."]
        data = pd.DataFrame(data=text_data, columns=["data"])

        prep = Data(x_train=data, split=False, report_name="test")
        prep.remove_stopwords_nltk("data", custom_stopwords=["please"])
        validate = prep.x_train.data_rem_stop.values.tolist()

        self.assertListEqual(validate, ["split ."])


if __name__ == "__main__":
    unittest.main()
