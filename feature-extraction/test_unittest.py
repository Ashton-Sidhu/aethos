import unittest

from categorical import *
from feature import *
from text import *


class TestFeatureExtraction(unittest.TestCase):

    def test_featureextractiontext_bow(self):

        list_of_sentences = ['Hi my name is pyml',
                            'Hi name pyml']

        feature = Feature(list_of_sentences)
        transform_data = feature.BagofWords()
        validate = transform_data.toarray().tolist()

        self.assertListEqual(validate, [[1, 1, 1, 1, 1],
                                        [1, 0, 0, 1, 1]] )

    def test_featureextractiontext_tfidf(self):

        list_of_sentences = ['Hi my name is pyml',
                            'Hi name pyml']

        feature = Feature(list_of_sentences)
        transform_data = feature.TFIDF(data=list_of_sentences, tfidf_kwargs={"lowercase": False})
        validate = transform_data.shape[1]

        self.assertEqual(validate, 5)

    def test_featureextractioncategorical_onehot(self):
        
        normal_data = [["Green", "Green", "Car"],
                        ["Green", "Other", "Truck"],
                        ["Blue", "Other", "Truck"]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(normal_data, columns=columns)

        feature = Feature(data)
        transform_data = feature.OneHotEncode(["col1", "col3"], data=data)
        validate = transform_data.values.tolist()

        self.assertListEqual(validate, [["Green", 0, 1, 1, 0],
                                        ["Other", 0, 1, 0, 1],
                                        ["Other", 1, 0, 0, 1]])

if __name__ == "__main__":
    unittest.main()
