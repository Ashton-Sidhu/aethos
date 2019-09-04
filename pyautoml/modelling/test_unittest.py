import unittest

import pandas as pd
from pyautoml import Model


class TestModelling(unittest.TestCase):

    def test_text_gensim_summarize(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(data=data, split=False)
        model.summarize_gensim('data', ratio=0.5)
        validate = model.data_summarized is not None

        self.assertTrue(validate)

    def test_text_gensim_keywords(self):

        text_data = [
                    "Hi my name is PyAutoML. Please split me.",
                    "This function is going to split by sentence. Automation is great."
                    ]

        data = pd.DataFrame(data=text_data, columns=['data'])

        model = Model(data=data, split=False)
        model.extract_keywords_gensim('data', ratio=0.5)
        validate = model.data_extracted_keywords is not None

        self.assertTrue(validate)

if __name__ == "__main__":
    unittest.main()
