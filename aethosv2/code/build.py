import inspect
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import black


class CodeBuilder(object):
    def build(self, pipeline):

        import_dict = {
            "Pipeline": "from sklearn.pipeline import Pipeline",
            "ColumnTransformer": "from sklearn.compose import ColumnTransformer",
        }

        imports = self._create_imports(pipeline, import_dict)

        formatted_imports = "\n".join(list(imports.values()))

        return black.format_str(
            f"{formatted_imports}\npipe = {str(pipeline)}.fit(df)",
            mode=black.Mode(string_normalization=True, line_length=100, is_pyi=True),
        )

    def _create_imports(self, pipeline, import_dict):

        transformer_index = 1

        # If it's not a Pipeline or a Column Transformer return right away
        if isinstance(pipeline, Pipeline):
            list_of_steps = pipeline.steps
        elif isinstance(pipeline, ColumnTransformer):
            list_of_steps = pipeline.transformers
        else:
            return import_dict

        for step in list_of_steps:
            transformer = step[transformer_index]
            transformer_class_name = transformer.__class__.__name__

            if transformer_class_name not in import_dict:
                import_dict[
                    transformer_class_name
                ] = f"from {transformer.__module__} import {transformer_class_name}"

            self._create_imports(transformer, import_dict)

        return import_dict
