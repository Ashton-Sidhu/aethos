import os
import sys

import click


## NOTE: Temporary for now until web app is developed and other functionality.
def create(ctx, param, value):
    os.system("cookiecutter https://github.com/drivendata/cookiecutter-data-science")


@click.command(context_settings=dict(help_option_names=[u"-h", u"--help"]))
@click.argument(u"create", callback=create)
def main():

    sys.exit()
