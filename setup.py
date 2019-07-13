from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='py-automl',
    url='https://github.com/Ashton-Sidhu/py-automl',
    packages=find_packages(),
    author='Ashton Sidhu',
    author_email='ashton.sidhu1994@gmail.com',
    install_requires=['numpy', 'pandas', 'scikit-learn'],
    version='0.1.1',
    license='MIT',
    description='A library of data science and machine learning techniques to help automate workflow.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True
)
