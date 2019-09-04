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

        model = Model(data=text_data, split=False)
        model.summarize_gensim_textrank('data')
        validate = model.data_summarized is not None
        print(model)

        self.assertTrue(validate)

if __name__ == "__main__":
    unittest.main()
