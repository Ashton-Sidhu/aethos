import os

import click


@click.group()
def main():
    pass


@main.command()
def create():
    """
    Creates a Data Science folder structure and files.
    """
    os.system("cookiecutter https://github.com/drivendata/cookiecutter-data-science")


@main.command()
def enable_extensions():
    """
    Enables jupyter extensions such as qgrid.
    """
    os.system("jupyter nbextension enable --py --sys-prefix widgetsnbextension")
    os.system("jupyter nbextension enable --py --sys-prefix qgrid")


@main.command()
def install_corpora():
    """
    Installs the necessary corpora from spaCy and NLTK for NLP analysis.
    """
    os.system("python3 -m textblob.download_corpora")
    os.system("python3 -c 'import nltk; nltk.download(\"stopwords\")'")
    os.system("python3 -m spacy download en")
