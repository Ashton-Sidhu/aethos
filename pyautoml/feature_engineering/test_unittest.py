import unittest

import pandas as pd
from pyautoml import Feature


class TestFeatureExtraction(unittest.TestCase):

    def test_featureextractiontext_bow(self):

        list_of_sentences = ['Hi my name is pyml',
                             'Hi name pyml']

        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Feature(data=data,
                          test_split_percentage=0.5, split=False)
        feature.bag_of_words(keep_col=False)
        validate = feature.x_train.values.tolist()

        self.assertListEqual(validate, [[1, 1, 1, 1, 1],
                                        [1, 0, 0, 1, 1]])

    def test_featureextractiontext_bow_keepcol(self):

        list_of_sentences = ['Hi my name is pyml',
                             'Hi name pyml']

        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Feature(data=data,
                          test_split_percentage=0.5, split=False)
        feature.bag_of_words(keep_col=True)
        validate = feature.x_train.values.tolist()

        self.assertListEqual(validate, [['Hi my name is pyml', 1, 1, 1, 1, 1],
                                        ['Hi name pyml', 1, 0, 0, 1, 1]])

    def test_featureextractiontext_tfidf(self):

        list_of_sentences = ['Hi my name is pyml',
                             'Hi name pyml']
        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Feature(data=data,
                          test_split_percentage=0.5, split=False)
        feature.tfidf(keep_col=False, lowercase=False, stop_words='english')
        validate = feature.x_train.shape[1]

        self.assertEqual(validate, 2)

    def test_featureextractiontext_splittfidf(self):

        list_of_sentences = ['Hi my name is pyml',
                             'Hi name pyml']
        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Feature(data=data,
                          test_split_percentage=0.5)
        feature.tfidf('text', keep_col=False, lowercase=False, stop_words='english')
        validate = feature.x_train.shape[1]

        self.assertEqual(validate, 2)

    def test_featureextractiontext_tfidf_keepcol(self):

        list_of_sentences = ['Hi my name is pyml',
                             'Hi name pyml']
        columns = ["text"]
        data = pd.DataFrame(list_of_sentences, columns=columns)

        feature = Feature(data=data,
                          test_split_percentage=0.5, split=False)
        feature.tfidf(keep_col=True, lowercase=False, stop_words='english')
        validate = feature.x_train.shape[1]

        self.assertEqual(validate, 3)

    def test_featureextractioncategorical_onehot(self):

        normal_data = [["Green", "Green", "Car"],
                       ["Green", "Other", "Truck"],
                       ["Blue", "Other", "Truck"]]

        columns = ["col1", "col2", "col3"]
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Feature(data=data, test_split_percentage=0.5, split=False)
        feature.onehot_encode(list_of_cols=["col1", "col3"], keep_col=False)
        validate = feature.x_train.values.tolist()

        self.assertListEqual(validate, [["Green", 0, 1, 1, 0],
                                        ["Other", 0, 1, 0, 1],
                                        ["Other", 1, 0, 0, 1]])

    def test_featureextractiontext_postag(self):

        normal_data = ["hi welcome to PyAutoML.",
                       "This application automates common Data Science/ML analysis tasks."]

        columns = ["text"]
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Feature(data=data, test_split_percentage=0.5, split=False)
        feature.postag_nltk()
        validate = len(feature.x_train.columns)

        self.assertTrue(validate, 2)


    def test_feature_fulldata_apply(self):

        data = [[1, 0, 1],
                [0, 2, 0],
                [1, 0, 1]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(data, columns=columns)

        feature = Feature(data=data, split=False)
        feature.apply(lambda x: x['col1'] > 0, 'new_col')
        validate = 'new_col' in feature.x_train.columns

        self.assertTrue(validate)

    def test_feature_splitdata_apply(self):

        data = [["py", 0, 1],
                ["auto", 2, 0],
                ["ml", 0, 1]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(data, columns=columns)

        feature = Feature(data=data, test_split_percentage=0.33)
        feature.apply(lambda x: x['col1'], 'new_col')
        validate = 'new_col' in feature.x_train.columns and 'new_col' in feature.x_test.columns

        self.assertTrue(validate)

    def test_feature_labelencoder(self):

        data = [["canada", "green", 1],
                ["canada", "green", 1],
                ["canada", "green", 0]]

        data = pd.DataFrame(data=data, columns=['col1', 'col2', 'col3'])

        feature = Feature(data=data, test_split_percentage=0.33, report_name="test")
        feature.encode_labels('col1', 'col2')
        
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
