from setuptools import find_packages, setup

setup(
    name='py_automl',
    url='https://github.com/Ashton-Sidhu/py-automl',
    packages=find_packages(),
    author='Ashton Sidhu',
    author_email='ashton.sidhu1994@gmail.com',
    install_requires=['numpy', 'pandas', 'scikit-learn'],
    version='0.1',
    license='MIT',
    description='A library of data science and machine learning techniques to help automate workflow.',
    long_description=open('README.md').read()
)
