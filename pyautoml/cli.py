import os

import click


## NOTE: Temporary for now until web app is developed and other functionality.
def create(ctx, param, value):
    os.system("cookiecutter https://github.com/drivendata/cookiecutter-data-science")
    ctx.exit()

def install_extensions(ctx, param, value):
    os.system("jupyter nbextension enable --py --sys-prefix widgetsnbextension")
    os.system("jupyter nbextension enable --py --sys-prefix qgrid")
    ctx.exit()

def install_corpora(ctx, param, value):
    os.system("python3 -m textblob.download_corpora")
    os.system("python3 -c 'import nltk; nltk.download(\"stopwords\")'")
    os.system("python3 -m spacy download en")
    ctx.exit()

@click.command(context_settings=dict(help_option_names=[u"-h", u"--help"]))
@click.option(u"-c", u"--create", is_flag=True, expose_value=True, callback=create)
@click.option(u"-ie", u"--install-extensions", is_flag=True, expose_value=True, callback=install_extensions)
@click.option(u"-ic", u"--install-corpora", is_flag=True, expose_value=True, callback=install_corpora)
def main(create, install_extensions, install_corpora):
    pass
