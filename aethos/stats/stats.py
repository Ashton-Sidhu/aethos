import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import swifter
from aethos.config import technique_reason_repo
from scipy.stats.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tqdm import tqdm


class Stats(object):
    def predict_data_sample(self):
        """
        Identifies how similar the train and test set distribution are by trying to predict whether each sample belongs
        to the train or test set using Random Forest, 10 Fold Stratified Cross Validation.

        The lower the F1 score, the more similar the distributions are as it's harder to predict which sample belongs to which distribution.

        Credit: https://www.kaggle.com/nanomathias/distribution-of-test-vs-training-data#1.-t-SNE-Distribution-Overview

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.

        Examples
        --------
        >>> data.predict_data_sample()
        """

        if self.x_test is None or not self.target_field:
            raise ValueError(
                "Test data or target field must be set. They can be set by assigning values to the `target_field` or the `x_test` variable."
            )

        report_info = technique_reason_repo["stats"]["dist_compare"]["predict"]

        x_train = self.x_train.drop(self.target_field, axis=1)
        x_test = self.x_test.drop(self.target_field, axis=1)

        x_train["label"] = 1
        x_test["label"] = 0

        data = pd.concat([x_train, x_test], axis=0)
        label = data["label"].tolist()

        predictions = cross_val_predict(
            ExtraTreesClassifier(n_estimators=100),
            data.drop(columns=["label"]),
            label,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        )

        if self.report is not None:
            self.report.report_technique(report_info, [])

        print(classification_report(data["label"].tolist(), predictions))

        return self.copy()

    def ks_feature_distribution(self, threshold=0.1, show_plots=True):
        """
        Uses the Kolomogorov-Smirnov test see if the distribution in the training and test sets are similar.
        
        Credit: https://www.kaggle.com/nanomathias/distribution-of-test-vs-training-data#1.-t-SNE-Distribution-Overview

        Parameters
        ----------
        threshold : float, optional
            KS statistic threshold, by default 0.1

        show_plots : bool, optional
            True to show histograms of feature distributions, by default True

        Returns
        -------
        DataFrame
            Columns that are significantly different in the train and test set.

        Examples
        --------
        >>> data.ks_feature_distribution()
        >>> data.ks_feature_distribution(threshold=0.2)
        """

        if self.x_test is None:
            raise ValueError(
                "Data must be split into train and test set. Please set the `x_test` variable."
            )

        report_info = technique_reason_repo["stats"]["dist_compare"]["ks"]

        diff_data = []
        diff_df = None

        for col in tqdm(self.x_train.columns):
            statistic, pvalue = ks_2samp(
                self.x_train[col].values, self.x_test[col].values
            )

            if pvalue <= 0.05 and np.abs(statistic) > threshold:
                diff_data.append(
                    {
                        "feature": col,
                        "p": np.round(pvalue, 5),
                        "statistic": np.round(np.abs(statistic), 2),
                    }
                )

        if diff_data:
            diff_df = pd.DataFrame(diff_data).sort_values(
                by=["statistic"], ascending=False
            )

            if show_plots:
                n_cols = 4
                n_rows = int(len(diff_df) / n_cols) + 1

                _, ax = plt.subplots(n_rows, n_cols, figsize=(40, 6 * n_rows))

                for i, (_, row) in enumerate(diff_df.iterrows()):
                    if i >= len(ax):
                        break
                    extreme = np.max(
                        np.abs(
                            self.x_train[row.feature].tolist()
                            + self.x_test[row.feature].tolist()
                        )
                    )
                    self.x_train.loc[:, row.feature].swifter.apply(np.log1p).hist(
                        ax=ax[i],
                        alpha=0.6,
                        label="Train",
                        density=True,
                        bins=np.arange(-extreme, extreme, 0.25),
                    )

                    self.x_test.loc[:, row.feature].swifter.apply(np.log1p).hist(
                        ax=ax[i],
                        alpha=0.6,
                        label="Train",
                        density=True,
                        bins=np.arange(-extreme, extreme, 0.25),
                    )

                    ax[i].set_title(f"Statistic = {row.statistic}, p = {row.p}")
                    ax[i].set_xlabel(f"Log({row.feature})")
                    ax[i].legend()

                plt.tight_layout()
                plt.show()

            if self.report is not None:
                self.report.report_technique(report_info, [])

        return diff_df
