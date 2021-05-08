import os
import sys
from subprocess import check_call

from setuptools import find_packages, setup
from setuptools.command.install import install

VERSION = "2.0.1"

pkgs = []

# extras = {"ptmodels": ["transformers==2.3.0", "tensorflow==2.1.0"]}
extras = {"spark": ["pyspark>=3.0.0"], "dask": [], "spark2": []}


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
    url="https://github.com/Ashton-Sidhu/aethos",
    packages=find_packages(),
    author="Ashton Sidhu",
    author_email="ashton.sidhu1994@gmail.com",
    install_requires=pkgs,
    extras_require=extras,
    version=VERSION,
    license="GPL-3.0",
    description="A library of data science and machine learning techniques to help automate your workflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="datascience, machinelearning, automation, analysis",
    python_requires=">= 3.6",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
    ],
    cmdclass={"verify": VerifyVersionCommand},
    entry_points={"console_scripts": ["aethos=aethos.__main__:main"]},
)
