import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import scipy as sc
import statsmodels.api as sm
import swifter
from aethos.config import technique_reason_repo
from aethos.visualizations import visualize as viz
from scipy.stats.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tqdm import tqdm
from collections import Counter
from typing import Union
from aethos.stats.util import run_2sample_ttest


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

                _, ax = plt.subplots(n_rows, n_cols, figsize=(40, 8 * n_rows))

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

    def most_common(self, col: str, n=15, plot=False, use_test=False):
        """
        Analyzes the most common values in the column and either prints them or displays a bar chart.
        
        Parameters
        ----------
        col : str
            Column to analyze

        n : int, optional
            Number of top most common values to display, by default 15

        plot : bool, optional
            True to plot a bar chart, by default False

        use_test : bool, optional
            True to analyze the test set, by default False

        Examples
        --------
        >>> data.most_common('col1', plot=True)
        >>> data.most_common('col1', n=50, plot=True)
        >>> data.most_common('col1', n=50)
        """

        if use_test:
            data = self.x_test[col].tolist()
        else:
            data = self.x_train[col].tolist()

        test_sample = data[0]

        if isinstance(test_sample, list):
            data = itertools.chain(*map(list, data))
        elif isinstance(test_sample, str):
            data = map(str.split, data)
            data = itertools.chain(*data)

        counter = Counter(data)
        most_common = dict(counter.most_common(n))

        if plot:
            from aethos.visualizations.visualize import barplot

            df = pd.DataFrame(list(most_common.items()), columns=["Word", "Count"])

            barplot(
                x="Word", y="Count", data=df,
            )
        else:
            for k, v in most_common.items():
                print(f"{k}: {v}")

    def ind_ttest(self, group1: str, group2: str, equal_var=True, output_file=None):
        """
        Performs an Independent T test.

        This is to be used when you want to compare the means of 2 groups.

        If group 2 column name is not provided and there is a test set, it will compare the same column in the train and test set.

        If there are any NaN's they will be omitted.
        
        Parameters
        ----------
        group1 : str
            Column for group 1 to compare.
        
        group2 : str, optional
            Column for group 2 to compare, by default None

        equal_var : bool, optional
            If True (default), perform a standard independent 2 sample test that assumes equal population variances.
            If False, perform Welch's t-test, which does not assume equal population variance, by default True

        output_file : str, optional
            Name of the file to output, by default None

        Returns
        -------
        list
            T test statistic, P value

        Examples
        --------
        >>> data.ind_ttest('col1', 'col2')
        >>> data.ind_ttest('col1', 'col2', output_file='ind_ttest.png')
        """

        res = run_2sample_ttest(
            group1, group2, self.x_train, "ind", output_file, equal_var=equal_var
        )

        return res

    def paired_ttest(self, group1: str, group2=None, output_file=None):
        """
        Performs a Paired t-test.

        This is to be used when you want to compare the means from the same group at different times.

        If group 2 column name is not provided and there is a test set, it will compare the same column in the train and test set.

        If there are any NaN's they will be omitted.
        
        Parameters
        ----------
        group1 : str
            Column for group 1 to compare.
        
        group2 : str, optional
            Column for group 2 to compare, by default None

        equal_var : bool, optional
            If True (default), perform a standard independent 2 sample test that assumes equal population variances.
            If False, perform Welch's t-test, which does not assume equal population variance, by default True

        output_file : str, optional
            Name of the file to output, by default None

        Returns
        -------
        list
            T test statistic, P value

        Examples
        --------
        >>> data.paired_ttest('col1', 'col2')
        >>> data.paired_ttest('col1', 'col2', output_file='pair_ttest.png')
        """

        # The implementation is the same as an independent t-test
        res = run_2sample_ttest(group1, group2, self.x_train, "pair", output_file)

        return res

    def onesample_ttest(self, group1: str, mean: Union[float, int], output_file=None):
        """
        Performs a One Sample t-test.

        This is to be used when you want to compare the mean of a single group against a known mean.

        If there are any NaN's they will be omitted.
        
        Parameters
        ----------
        group1 : str
            Column for group 1 to compare.
        
        mean : float, int, optional
            Sample mean to compare to.

        output_file : str, optional
            Name of the file to output, by default None

        Returns
        -------
        list
            T test statistic, P value

        Examples
        --------
        >>> data.onesample_ttest('col1', 1)
        >>> data.onesample_ttest('col1', 1, output_file='ones_ttest.png')
        """

        data_group1 = self.x_train[group1].tolist()

        results = sc.stats.ttest_1samp(data_group1, mean, nan_policy="omit")

        matrix = [
            ["", "Test Statistic", "p-value"],
            ["Sample Data", results[0], results[1]],
        ]

        viz.create_table(matrix, True, output_file)

        return results

    def anova(
        self,
        dep_var: str,
        num_variables=[],
        cat_variables=[],
        formula=None,
        verbose=False,
    ):
        """
        Runs an anova.

        Anovas are to be used when one wants to compare the means of a condition between 2+ groups.

        ANOVA tests if there is a difference in the mean somewhere in the model (testing if there was an overall effect), but it does not tell one where the difference is if the there is one.
        
        Parameters
        ----------
        dep_var : str
            Dependent variable you want to explore the relationship of

        num_variables : list, optional
            Numeric variable columns, by default []

        cat_variables : list, optional
            Categorical variable columns, by default []

        formula : str, optional
            OLS formula statsmodel lib, by default None

        verbose : bool, optional
            True to print OLS model summary and formula, by default False

        Returns
        -------


        Examples
        --------
        >>> data.anova('dep_col', num_variables=['col1', 'col2'], verbose=True)
        >>> data.anova('dep_col', cat_variables=['col1', 'col2'], verbose=True)
        >>> data.anova('dep_col', num_variables=['col1', 'col2'], cat_variables=['col3'] verbose=True)
        """

        from statsmodels.formula.api import ols

        assert (
            num_variables != [] or cat_variables != []
        ), "You must specify variables, either categorical or numerical."

        # Create the formula string to pass into OLS in the form of `dep_colname` ~ `num_col1` + C(`cat_col1`) + ...
        cat_variables = [f"C({var})" for var in cat_variables]
        join = "+" if cat_variables and num_variables else ""
        formula = (
            f'{dep_var} ~ {" + ".join(num_variables)} {join} {" + ".join(cat_variables)}'
            if not formula
            else formula
        )

        mod = ols(formula, data=self.x_train).fit()

        if verbose:
            print(formula)
            print(mod.summary())

        table = sm.stats.anova_lm(mod, typ=2)

        print(table)
