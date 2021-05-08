import pandas as pd

from aethosv2.code.build import CodeBuilder

from typing import Union

# Abstract class that is not backend specific
# Maybe this is the main entry point class?
class CoreTransformer(object):
    def __init__(self, df: Union[pd.DataFrame]):
        self.df = df
        # self.pipeline = PipelineConstructor(constructor=Pipeline)
        self.column_transformer = {col: [] for col in df.columns}
        self.code_builder = CodeBuilder()
        # self.pipeline_steps = self.pipeline.constructor.steps

    def run(self):
        pipe = self._build_pipeline()

        return pipe.fit(self.df).transform(self.df)

    def generate_code(self):
        self.code_builder.build(self.pipeline)

    def _add_transformer(self, columns, transformer):
        for column in list(columns):
            self.column_transformer[column].append(transformer)
