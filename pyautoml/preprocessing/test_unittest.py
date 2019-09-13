import unittest

import numpy as np
import pandas as pd
from pyautoml import Preprocess


class TestPreprocessing(unittest.TestCase):

    def test_preprocessnumeric_normalize(self):

        unnormal_data = [[5.0, 3, 1],
                        [2.0, 2, 1],
                        [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]        
        data = pd.DataFrame(unnormal_data, columns=columns)

        preprocess = Preprocess(data=data, test_split_percentage=0.5, split=False)
        preprocess.normalize_numeric()
        validate = preprocess.data.values.tolist()

        self.assertListEqual(validate, [[.375, 1.0, 0.0],
                                        [0, 0.5, 0.0],
                                        [1.0, 0, 0.0]])

    def test_preprocess_traindata(self):

        unnormal_train_data = [[5.0, 3, 1],
                            [2.0, 2, 1],
                            [10.0, 1, 1]]

        unnormal_test_data = [[5.0, 3, 1],
                            [2.0, 2, 1],
                            [10.0, 1, 1]]

        columns = ["col1", "col2", "col3"]        
        train_data = pd.DataFrame(unnormal_train_data, columns=columns)
        test_data = pd.DataFrame(unnormal_test_data, columns=columns)

        preprocess = Preprocess(train_data=train_data, test_data=test_data, test_split_percentage=0.5)
        preprocess.normalize_numeric("col1", "col2", "col3")
        validate_train = preprocess.train_data.values.tolist()
        validate_test = preprocess.test_data.values.tolist()

        self.assertListEqual(validate_train, validate_test)

    def test_preprocess_splitsentences(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]
        data = pd.DataFrame(data=text_data, columns=['data'])

        prep = Preprocess(data=data, split=False)
        prep.split_sentences('data')
        validate = prep.data['data_sentences'].values.tolist()

        self.assertListEqual(validate, [["Hi my name is PyAutoML.", "Please split me."],
                                        ["This function is going to split by sentence.", "Automation is great."]])

    def test_preprocess_nltkstem(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]
        data = pd.DataFrame(data=text_data, columns=['data'])

        prep = Preprocess(data=data, split=False)
        prep.stem_nltk('data')
        validate = prep.data.shape[1]

        self.assertEquals(validate, 2)

    def test_preprocess_nltksplit(self):

        text_data = [
                    "Please.exe split me.",
                    ]
        data = pd.DataFrame(data=text_data, columns=['data'])

        prep = Preprocess(data=data, split=False)
        prep.split_words_nltk('data')
        validate = prep.data.data_tokenized.values.tolist()

        self.assertListEqual(validate, [["Please.exe", "split", "me", "."]])

    def test_preprocess_nltksplit_regex(self):

        text_data = [
                    "Please123 split me.",
                    ]
        data = pd.DataFrame(data=text_data, columns=['data'])

        prep = Preprocess(data=data, split=False)
        prep.split_words_nltk('data', regexp=r'\w+\d+')
        validate = prep.data.data_tokenized.values.tolist()

        self.assertListEqual(validate, [["Please123"]])

    def test_preprocess_nltkremove_punctuation(self):

        text_data = [
                    "Please split me.",
                    ]
        data = pd.DataFrame(data=text_data, columns=['data'])

        prep = Preprocess(data=data, split=False)
        prep.remove_punctuation('data')
        validate = prep.data.data_rem_punct.values.tolist()

        self.assertListEqual(validate, ["Please split me"])

    def test_preprocess_nltkremove_punctuation_regexp(self):

        text_data = [
                    "Please.exe, split me.",
                    "hello it's me, testing.dll."
                    ]
        data = pd.DataFrame(data=text_data, columns=['data'])

        prep = Preprocess(data=data, split=False)
        prep.remove_punctuation('data', regexp=r'\w+\.\w+|\w+')
        validate = prep.data.data_rem_punct.values.tolist()

        self.assertListEqual(validate, ["Please.exe split me", "hello it s me testing.dll"])

    def test_preprocess_nltkremove_punctuation_exception(self):

        text_data = [
                    "Please,> split me.",
                    ]
        data = pd.DataFrame(data=text_data, columns=['data'])

        prep = Preprocess(data=data, split=False)
        prep.remove_punctuation('data', exceptions=['.',  '>'])
        validate = prep.data.data_rem_punct.values.tolist()

        self.assertListEqual(validate, ["Please> split me."])

    def test_preprocess_nltkremove_stopwords(self):

        text_data = [
                    "Please the split me.",
                    ]
        data = pd.DataFrame(data=text_data, columns=['data'])

        prep = Preprocess(data=data, split=False)
        prep.remove_stopwords_nltk('data', custom_stopwords=['please'])
        validate = prep.data.data_rem_stop.values.tolist()

        self.assertListEqual(validate, ["split ."])  

if __name__ == "__main__":
    unittest.main()
