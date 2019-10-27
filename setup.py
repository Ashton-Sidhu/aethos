import os
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install

VERSION = "0.6.0"

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != 'v' + VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='py-automl',
    url='https://github.com/Ashton-Sidhu/py-automl',
    packages=find_packages(),
    author='Ashton Sidhu',
    author_email='ashton.sidhu1994@gmail.com',
    install_requires=['numpy', 'pandas>=0.25', 'scikit-learn', 'textblob', 'matplotlib<3.2.0',
                      'pandas_summary', 'pandas-bokeh', 'ptitprince', 'nltk', 'ipython', 'gensim',
                      'pandas-profiling', 'cookiecutter', 'pathos', 'shap', 'interpret', 'yellowbrick',
                      'impyute', 'spacy', 'xgboost'],
    version=VERSION,
    license='GPL-3.0',
    description='A library of data science and machine learning techniques to help automate your workflow.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='datascience, machinelearning, automation, analysis',
    python_requires='>= 3.5',
    cmdclass={
        'verify': VerifyVersionCommand,
    },
    entry_points= {
        'console_scripts' : ['pyautoml=pyautoml.__main__:main']
    }
)
