import pytest

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from aethosv2.code.build import CodeBuilder


@pytest.fixture(scope="session")
def pipeline():
    # fmt: off
    return Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers=[
                ("Fare", Pipeline(steps=[("simpleimputer", SimpleImputer(strategy="most_frequent"),)]), ["Fare"],),
                ("Embarked", Pipeline(steps=[
                    ("simpleimputer", SimpleImputer(strategy="most_frequent"),),
                    ("onehotencoder", OneHotEncoder()),
                    ]), 
                    ["Embarked"],
                ),]),
            )
        ]
    )
    # fmt: on


@pytest.fixture(scope="session")
def cb():
    return CodeBuilder()


def test_code_builder(cb: CodeBuilder, pipeline):

    expected_code = "\n".join(
        [
            "from sklearn.pipeline import Pipeline",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.impute._base import SimpleImputer",
            "from sklearn.preprocessing._encoders import OneHotEncoder",
            "",
            "pipe = Pipeline(",
            "    steps=[",
            "        (",
            '            "preprocess",',
            "            ColumnTransformer(",
            "                transformers=[",
            "                    (",
            '                        "Fare",',
            "                        Pipeline(",
            '                            steps=[("simpleimputer", SimpleImputer(strategy="most_frequent"))]',
            "                        ),",
            '                        ["Fare"],',
            "                    ),",
            "                    (",
            '                        "Embarked",',
            "                        Pipeline(",
            "                            steps=[",
            '                                ("simpleimputer", SimpleImputer(strategy="most_frequent")),',
            '                                ("onehotencoder", OneHotEncoder()),',
            "                            ]",
            "                        ),",
            '                        ["Embarked"],',
            "                    ),",
            "                ]",
            "            ),",
            "        )",
            "    ]",
            ").fit(df)",
            "",
        ]
    )

    pipeline_code = cb.build(pipeline)

    assert pipeline_code == expected_code


def test_create_imports(cb: CodeBuilder, pipeline):

    import_dict = {
        "Pipeline": "from sklearn.pipeline import Pipeline",
        "ColumnTransformer": "from sklearn.compose import ColumnTransformer",
        "SimpleImputer": "from sklearn.impute._base import SimpleImputer",
        "OneHotEncoder": "from sklearn.preprocessing._encoders import OneHotEncoder",
    }

    imports = cb._create_imports(pipeline, import_dict)

    assert imports == import_dict
