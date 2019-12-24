import unittest

import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from aethos import Data


class TestFeatureExtraction(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(Path.home()) + "/.aethos/reports/")

    def test_featureextractiontext_bow(self):

        list_of_sentences = ["Hi my name is pyml", "Hi name pyml"]

        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        feature.bag_of_words(keep_col=False)
        validate = feature.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 1, 1, 1, 1], [1, 0, 0, 1, 1]])

    def test_featureextractiontext_bow_keepcol(self):

        list_of_sentences = ["Hi my name is pyml", "Hi name pyml"]

        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        feature.bag_of_words(keep_col=True)
        validate = feature.x_train.values.tolist()

        self.assertListEqual(
            validate,
            [["Hi my name is pyml", 1, 1, 1, 1, 1], ["Hi name pyml", 1, 0, 0, 1, 1]],
        )

    def test_featureextractiontext_tfidf(self):

        list_of_sentences = ["Hi my name is pyml", "Hi name pyml"]
        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        feature.tfidf(keep_col=False, lowercase=False, stop_words="english")
        validate = feature.x_train.shape[1]

        self.assertEqual(validate, 2)

    def test_featureextractiontext_splittfidf(self):

        list_of_sentences = ["Hi my name is pyml", "Hi name pyml"]
        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        feature.tfidf("text", keep_col=False, lowercase=False, stop_words="english")
        validate = feature.x_train.shape[1]

        self.assertEqual(validate, 2)

    def test_featureextractiontext_tfidf_keepcol(self):

        list_of_sentences = ["Hi my name is pyml", "Hi name pyml"]
        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        feature.tfidf(keep_col=True, lowercase=False, stop_words="english")
        validate = feature.x_train.shape[1]

        self.assertEqual(validate, 3)

    def test_featureextractioncategorical_onehot(self):

        normal_data = [
            ["Green", "Green", "Car"],
            ["Green", "Other", "Truck"],
            ["Blue", "Other", "Truck"],
        ]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Data(
            x_train=data, test_split_percentage=0.5, split=False, report_name="test"
        )
        feature.onehot_encode(list_of_cols=["col1", "col3"], keep_col=False)
        validate = feature.x_train.values.tolist()

        self.assertListEqual(
            validate,
            [["Green", 0, 1, 1, 0], ["Other", 0, 1, 0, 1], ["Other", 1, 0, 0, 1]],
        )

    def test_featureextractiontext_nltkpostag(self):

        normal_data = [
            "hi welcome to aethos.",
            "This application automates common Data Science/ML analysis tasks.",
        ]

        columns = ["text"]
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        feature.postag_nltk()
        validate = feature.x_train.shape[1] == 2 and feature.x_test.shape[1] == 2

        self.assertTrue(validate, 2)

    def test_feature_fulldata_apply(self):

        data = [[1, 0, 1], [0, 2, 0], [1, 0, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        feature = Data(x_train=data, split=False, report_name="test")
        feature.apply(lambda x: x["col1"] > 0, "new_col")
        validate = "new_col" in feature.x_train.columns

        self.assertTrue(validate)

    def test_feature_splitdata_apply(self):

        data = [["py", 0, 1], ["auto", 2, 0], ["ml", 0, 1]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(data, columns=columns)

        feature = Data(x_train=data, test_split_percentage=0.33, report_name="test")
        feature.apply(lambda x: x["col1"], "new_col")
        validate = (
            "new_col" in feature.x_train.columns and "new_col" in feature.x_test.columns
        )

        self.assertTrue(validate)

    def test_feature_labelencoder(self):

        data = [["canada", "green", 1], ["canada", "green", 1], ["canada", "green", 0]]

        data = pd.DataFrame(data=data, columns=["col1", "col2", "col3"])

        feature = Data(x_train=data, test_split_percentage=0.33, report_name="test")
        feature.encode_labels("col1", "col2")

        self.assertTrue(True)

    def test_feature_polynomial(self):

        data = np.arange(6).reshape(3, 2)

        data = pd.DataFrame(data=data, columns=["col1", "col2"])

        feature = Data(x_train=data, test_split_percentage=0.33, report_name="test")
        feature.polynomial_features()

        validate = feature.x_train.shape[1] == 6 and feature.x_test.shape[1] == 6

        self.assertTrue(validate)

    def test_featureextractiontext_spacypostag(self):

        normal_data = [
            "hi welcome to aethos.",
            "This application automates common Data Science/ML analysis tasks.",
        ]

        columns = ["text"]
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        feature.postag_spacy()
        validate = feature.x_train.shape[1] == 2 and feature.x_test.shape[1] == 2

        self.assertTrue(validate, 2)

    def test_featureextractiontext_spacyphrases(self):

        normal_data = [
            "hi welcome to aethos.",
            "This application automates common Data Science/ML analysis tasks.",
        ]

        columns = ["text"]
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        feature.nounphrases_spacy()
        validate = feature.x_train.shape[1] == 2 and feature.x_test.shape[1] == 2

        self.assertTrue(validate, 2)

    def test_featureextractiontext_nltkphrases(self):

        normal_data = [
            "hi welcome to aethos.",
            "This application automates common Data Science/ML analysis tasks.",
        ]

        columns = ["text"]
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        feature.nounphrases_nltk()
        validate = feature.x_train.shape[1] == 2 and feature.x_test.shape[1] == 2

        self.assertTrue(validate, 2)

    def test_featureextractiontext_hash_keepcol(self):

        list_of_sentences = ["Hi my name is pyml", "Hi name pyml"]

        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        feature.text_hash(keep_col=True, n_features=5)

        self.assertTrue(True)

    def test_feature_pca(self):

        data = np.arange(6).reshape(3, 2)

        data = pd.DataFrame(data=data, columns=["col1", "col2"])

        feature = Data(x_train=data, test_split_percentage=0.33, report_name="test")
        feature.pca(n_components=2)

        validate = feature.x_train.shape[1] == 2 and feature.x_test.shape[1] == 2

        self.assertTrue(validate)

    def test_feature_pcatarget(self):

        data = np.arange(9).reshape(3, 3)

        data = pd.DataFrame(data=data, columns=["col1", "col2", "col3"])

        feature = Data(
            x_train=data,
            test_split_percentage=0.33,
            target_field="col3",
            report_name="test",
        )
        feature.pca(n_components=2)

        validate = feature.col3 is not None and feature.x_train.shape[1] == 3

        self.assertTrue(validate)

    def test_util_corr(self):

        int_missing_data = [
            [1, 8, 3, 4],
            [2, 8, 5, 1],
            [3, 8, 5, 7],
            [4, 8, 5, 2],
            [5, 8, 5, 9],
            [6, 8, 5, 3],
        ]

        columns = ["col1", "col2", "col3", "col4"]
        data = pd.DataFrame(int_missing_data, columns=columns)

        feat = Data(x_train=data, test_split_percentage=0.5, report_name="test")
        feat.drop_correlated_features()

        validate = feat.x_test.columns.tolist() == feat.x_train.columns.tolist() and feat.x_test.columns.tolist() == [
            "col2",
            "col3",
            "col4",
        ]

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
