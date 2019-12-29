import os
import sys
from subprocess import check_call

from setuptools import find_packages, setup
from setuptools.command.install import install

VERSION = "1.0.0"

pkgs = [
    "numpy==1.17.4",
    "pandas>=0.25",
    "scikit-learn>=0.22",
    "textblob==0.15.3",
    "matplotlib<3.2.0",
    "pandas_summary",
    "pandas-bokeh==0.4.2",
    "ptitprince==0.1.5",
    "nltk==3.4.5",
    "ipython==7.10.1",
    "gensim==3.8.1",
    "pandas-profiling==2.3.0",
    "cookiecutter==1.6.0",
    "pathos==0.2.5",
    "shap==0.33.0",
    "interpret==0.1.20",
    "yellowbrick==1.0.1",
    "spacy==2.2.3",
    "xgboost==0.90",
    "ipywidgets==7.5.1",
    "qgrid==1.1.1",
    "python-dateutil<2.8.1",
    "itables==0.2.1",
    "selenium==3.141.0",
    "python-docx==0.8.10",
    "graphviz==0.13.2",
    "pyldavis==2.1.2",
    "swifter==0.297",
    "lightgbm==2.3.1",
    "catboost==0.20.1"
]


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != "v" + VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="aethos",
    url="https://github.com/Ashton-Sidhu/py-automl",
    packages=find_packages(),
    author="Ashton Sidhu",
    author_email="ashton.sidhu1994@gmail.com",
    install_requires=pkgs,
    version=VERSION,
    license="GPL-3.0",
    description="A library of data science and machine learning techniques to help automate your workflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="datascience, machinelearning, automation, analysis",
    data_files=[
        (
            str(os.path.join(os.path.expanduser("~"), ".aethos")),
            ["aethos/config/config.yml"],
        )
    ],
    python_requires=">= 3.6",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
    ],
    cmdclass={"verify": VerifyVersionCommand},
    entry_points={"console_scripts": ["aethos=aethos.__main__:main"]},
)
