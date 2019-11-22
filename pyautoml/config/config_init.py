import pyautoml.config.config as cf
from pyautoml.config.config import is_bool
from pyautoml.config import shell

interactive_df_doc = """
: bool
    Use QGrid library to interact with pandas dataframe.
    Default value is False
    Valid values: False, True
"""


def use_qgrid(key):
    import qgrid

    if shell == "ZMQInteractiveShell":
        if cf.get_option(key):
            qgrid.enable()
            qgrid.set_defaults(show_toolbar=True)
        else:
            qgrid.disable()


cf.register_option(
    "interactive_df",
    default=False,
    doc=interactive_df_doc,
    validator=is_bool,
    cb=use_qgrid,
)
