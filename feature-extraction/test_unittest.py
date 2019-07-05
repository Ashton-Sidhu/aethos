import unittest

from text import *


class TestFeatureExtraction(unittest.TestCase):

    def test_featureextractiontext_bow(self):

        list_of_sentences = ['Hi my name is pyml',
                            'Hi name pyml']

        transform_data = FeatureBagOfWords(data=list_of_sentences)
        validate = transform_data.toarray().tolist()

        self.assertListEqual(validate, [[1, 1, 1, 1, 1],
                                        [1, 0, 0, 1, 1]] )

    def test_featureextractiontext_tfidf(self):

        list_of_sentences = ['Hi my name is pyml',
                            'Hi name pyml']

        transform_data = FeatureTFIDF(data=list_of_sentences)
        validate = transform_data.shape[1]

        self.assertEqual(validate, 5)

if __name__ == "__main__":
    unittest.main()
