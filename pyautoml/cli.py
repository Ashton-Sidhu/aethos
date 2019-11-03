import os

import click


## NOTE: Temporary for now until web app is developed and other functionality.
def create(ctx, param, value):
    os.system("cookiecutter https://github.com/drivendata/cookiecutter-data-science")

def install_extensions(ctx, param, value):
    os.system("jupyter nbextension enable --py --sys-prefix widgetsnbextension")
    os.system("jupyter nbextension enable --py --sys-prefix qgrid")

def install_corpora(ctx, param, value):
    os.system("python3 -m textblob.download_corpora")
    os.system("python3 -c 'import nltk; nltk.download('stopwords')'")
    os.system("python3 -m spacy download en")

@click.command(context_settings=dict(help_option_names=[u"-h", u"--help"]))
@click.argument(u"create", callback=create)
@click.argument(u"install_extensions", callback=install_extensions)
@click.argument(u"install_corpora", callback=install_corpora)
def main(create, install_extensions, install_corpora):
    pass
