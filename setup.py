from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='py-automl',
    url='https://github.com/Ashton-Sidhu/py-automl',
    packages=find_packages(),
    author='Ashton Sidhu',
    author_email='ashton.sidhu1994@gmail.com',
    install_requires=['numpy', 'pandas', 'scikit-learn', 'textblob', 'pandas_summary', 'pandas-bokeh', 'ptitprince', 'nltk', 'ipython'],
    version='0.3.1',
    license='GPL-3.0',
    description='A library of data science and machine learning techniques to help automate your workflow.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True
)
