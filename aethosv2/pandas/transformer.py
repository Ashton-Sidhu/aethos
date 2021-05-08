from typing import Tuple, Union, List
import pandas as pd
import numpy as np
import yaml

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from aethosv2.core.transformer import CoreTransformer
from aethosv2.pipeline.constructor import PipelineConstructor
from aethosv2.code.build import CodeBuilder
from aethosv2.pandas.utils import DropColumns


class PipelineDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True

    @staticmethod
    def represent_estimators(self, data):
        return self.represent_str(str(data))


import sklearn

yaml.add_multi_representer(
    sklearn.base.BaseEstimator, PipelineDumper.represent_estimators
)


class PandasTransformer(CoreTransformer):

    # @property
    # def _constructor(self):
    #     return PandasTransformer

    def __init__(self, df: pd.DataFrame):
        super(PandasTransformer, self).__init__(df)

    def __repr__(self):

        return yaml.dump(
            self.column_transformer,
            Dumper=PipelineDumper,
            sort_keys=False,
            indent=4,
            default_flow_style=False,
        )

    def replace_missing_most_common(self, *columns, **kwargs):

        transformer = SimpleImputer(strategy="most_frequent", **kwargs)

        self._add_transformer(columns, transformer)

    def one_hot_encode(self, *columns, **kwargs):

        transformer = OneHotEncoder(**kwargs)

        self._add_transformer(columns, transformer)

    def replace_missing_mean(self, *columns, **kwargs):

        transformer = SimpleImputer(strategy="mean", **kwargs)

        self._add_transformer(columns, transformer)

    def _build_pipeline(self):

        end_to_end_steps = []

        # Create a pipeline for each column and add it to an e2e pipeline
        for column, transformers in self.column_transformer.items():
            if transformers:
                # Make sure operations on the same column are sequential
                steps = make_pipeline(*transformers)
                end_to_end_steps.append((column, steps, [column]))

        if end_to_end_steps:
            col_pipeline = ColumnTransformer(transformers=end_to_end_steps)
        else:
            raise RuntimeError("No steps added to pipeline!")

        return Pipeline(steps=[("preprocess", col_pipeline)])

    def drop(self, *columns):

        columns = list(columns)

        self.pipeline.add(make_column_transformer((DropColumns(columns), columns)))
