import unittest

import pandas as pd
from pyautoml.feature_engineering.feature import Feature


class TestFeatureExtraction(unittest.TestCase):

    def test_featureextractiontext_bow(self):

        list_of_sentences = ['Hi my name is pyml',
                             'Hi name pyml']

        feature = Feature(list_of_sentences,
                          test_split_percentage=0.5, use_full_data=True)
        transform_data = feature.BagofWords()
        validate = transform_data.toarray().tolist()

        self.assertListEqual(validate, [[1, 1, 1, 1, 1],
                                        [1, 0, 0, 1, 1]])

    def test_featureextractiontext_tfidf(self):

        list_of_sentences = ['Hi my name is pyml',
                             'Hi name pyml']

        feature = Feature(list_of_sentences,
                          test_split_percentage=0.5, use_full_data=True)
        transform_data = feature.TFIDF(params={"lowercase": False})
        validate = transform_data.shape[1]

        self.assertEqual(validate, 5)

    def test_featureextractioncategorical_onehot(self):

        normal_data = [["Green", "Green", "Car"],
                       ["Green", "Other", "Truck"],
                       ["Blue", "Other", "Truck"]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Feature(data, test_split_percentage=0.5, use_full_data=True)
        transform_data = feature.OneHotEncode(["col1", "col3"], data=data)
        validate = transform_data.values.tolist()

        self.assertListEqual(validate, [["Green", 0, 1, 1, 0],
                                        ["Other", 0, 1, 0, 1],
                                        ["Other", 1, 0, 0, 1]])

    def test_featureextractiontext_postag(self):

        normal_data = ["hi welcome to PyAutoML.",
                       "This application automates common Data Science/ML analysis tasks."]

        columns = ["text"]
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Feature(data, test_split_percentage=0.5, use_full_data=True)
        transform_data = feature.PoSTag()
        validate = len(transform_data.columns)

        self.assertTrue(validate, 2)


if __name__ == "__main__":
    unittest.main()
