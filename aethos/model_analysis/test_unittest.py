import pandas as pd
import unittest
import numpy as np

from aethos import Classification, Regression, Unsupervised

class TestModelAnalysis(unittest.TestCase):
    def test_plot_predicted_actual(self):

        data = np.random.randint(0, 2, size=(500, 3))

        data = pd.DataFrame(data=data, columns=["col1", "col2", "col3"])

        model = Regression(x_train=data, target="col3", test_split_percentage=0.5,)
        m = model.LinearRegression(model_name="l1", run=True)

        m.plot_predicted_actual()

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
