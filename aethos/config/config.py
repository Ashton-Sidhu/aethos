"""
Implemented pandas version of package wide configuration: https://github.com/pandas-dev/pandas/blob/f9f95c0762af2c114ed27d21288c9d25ab49634d/pandas/_config/config.py

Overview
========
This module supports the following requirements:
- options are referenced using keys in dot.notation, e.g. "x.y.option - z".
- keys are case-insensitive.
- functions should accept partial/regex keys, when unambiguous.
- options can be registered by modules at import time.
- options can be registered at init-time (via core.config_init)
- options have a default value, and (optionally) a description and
  validation function associated with them.
- options can be deprecated, in which case referencing them
  should produce a warning.
- deprecated options can optionally be rerouted to a replacement
  so that accessing a deprecated option reroutes to a differently
  named option.
- options can be reset to their default value.
- all option can be reset to their default value at once.
- you can register a callback to be invoked when the option value
  is set or reset. Changing the stored value is considered misuse, but
  is not verboten.

Implementation
==============

- "Registered options" and "Deprecated options" have metadata associated
  with them, which are stored in auxiliary dictionaries keyed on the
  fully-qualified key, e.g. "x.y.z.option".

- the config_init module is imported by the package's __init__.py file.
  placing any register_option() calls there will ensure those options
  are available as soon as pandas is loaded. If you use register_option
  in a module, it will only be available after that module is imported,
  which you should be aware of.

- `config_prefix` is a context_manager (for use with the `with` keyword)
  which can save developers some typing, see the docstring.
"""

import re
from collections import namedtuple
from contextlib import contextmanager

_registered_options = {}
_global_config = {}

RegisteredOption = namedtuple("RegisteredOption", "key default doc validator cb")


class OptionError(AttributeError, KeyError):
    """Exception for pandas.options, backwards compatible with KeyError
    checks
    """


def register_option(key, default, doc="", validator=None, cb=None):
    """Register an option in the package-wide pandas config object
    Parameters
    ----------
    key       - a fully-qualified key, e.g. "x.y.option - z".
    default    - the default value of the option
    doc       - a string description of the option
    validator - a function of a single argument, should raise `ValueError` if
                called with a value which is not a legal value for the option.
    cb        - a function of a single argument "key", which is called
                immediately after an option value is set/reset. key is
                the full name of the option.
    Returns
    -------
    Nothing.
    
    Raises
    ------
    ValueError if `validator` is specified and `default` is not a valid value.
    """
    import tokenize
    import keyword

    key = key.lower()

    if key in _registered_options:
        msg = "Option '{key}' has already been registered"
        raise OptionError(msg.format(key=key))

    # the default value should be legal
    if validator:
        validator(default)

    _global_config[key] = default  # initialize

    # save the option metadata
    _registered_options[key] = RegisteredOption(
        key=key, default=default, doc=doc, validator=validator, cb=cb
    )


def _select_options(pat):
    """returns a list of keys matching `pat`
    if pat=="all", returns all registered options
    """

    # short-circuit for exact key
    if pat in _registered_options:
        return [pat]

    # else look through all of them
    keys = sorted(_registered_options.keys())
    if pat == "all":  # reserved key
        return keys

    return [k for k in keys if re.search(pat, k, re.I)]


def _get_single_key(pat):
    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError("No such keys(s): {pat!r}".format(pat=pat))
    if len(keys) > 1:
        raise OptionError("Pattern matched multiple keys")
    key = keys[0]

    return key


def _get_registered_option(key):
    """
    Retrieves the option metadata if `key` is a registered option.

    Returns
    -------
    RegisteredOption (namedtuple) if key is deprecated, None otherwise
    """

    return _registered_options.get(key)


def _get_option(pat):
    key = _get_single_key(pat)

    return _global_config[key]


def _set_option(*args, **kwargs):
    # must at least 1 arg deal with constraints later
    nargs = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword arguments")

    if kwargs:
        msg = '_set_option() got an unexpected keyword argument "{kwarg}"'
        raise TypeError(msg.format(list(kwargs.keys())[0]))

    for k, v in zip(args[::2], args[1::2]):
        key = _get_single_key(k)

        o = _get_registered_option(key)
        if o and o.validator:
            o.validator(v)

        _global_config[key] = v

        if o.cb:
            o.cb(key)


def _build_option_description(k):
    """ Builds a formatted description of a registered option and prints it """

    o = _get_registered_option(k)

    s = "{k} ".format(k=k)

    if o.doc:
        s += "\n".join(o.doc.strip().split("\n"))
    else:
        s += "No description available."

    if o:
        s += "\n    [default: {default}] [currently: {current}]".format(
            default=o.default, current=_get_option(k)
        )

    return s


def _describe_option(pat="", _print_desc=True):

    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError("No such keys(s)")

    s = ""
    for k in keys:  # filter by pat
        s += _build_option_description(k)

    if _print_desc:
        print(s)
    else:
        return s


def _reset_option(pat):

    keys = _select_options(pat)

    if len(keys) == 0:
        raise OptionError("No such keys(s)")

    if len(keys) > 1 and len(pat) < 4 and pat != "all":
        raise ValueError(
            "You must specify at least 4 characters when "
            "resetting multiple keys, use the special keyword "
            '"all" to reset all the options to their default '
            "value"
        )

    for k in keys:
        _set_option(k, _registered_options[k].default)


def pp_options_list(keys, width=80, _print=False):
    """ Builds a concise listing of available options, grouped by prefix """

    from textwrap import wrap
    from itertools import groupby

    def pp(name, ks):
        pfx = "- " + name + ".[" if name else ""
        ls = wrap(
            ", ".join(ks),
            width,
            initial_indent=pfx,
            subsequent_indent="  ",
            break_long_words=False,
        )
        if ls and ls[-1] and name:
            ls[-1] = ls[-1] + "]"
        return ls

    ls = []
    singles = [x for x in sorted(keys) if x.find(".") < 0]
    if singles:
        ls += pp("", singles)
    keys = [x for x in keys if x.find(".") >= 0]

    for k, g in groupby(sorted(keys), lambda x: x[: x.rfind(".")]):
        ks = [x[len(k) + 1 :] for x in list(g)]
        ls += pp(k, ks)
    s = "\n".join(ls)
    if _print:
        print(s)
    else:
        return s


class DictWrapper:
    """ provide attribute-style access to a nested dict"""

    def __init__(self, d, prefix=""):
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "prefix", prefix)

    def __setattr__(self, key, val):
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        # you can't set new keys
        # can you can't overwrite subtrees
        if key in self.d and not isinstance(self.d[key], dict):
            _set_option(prefix, val)
        else:
            raise OptionError("You can only set the value of existing options")

    def __getattr__(self, key):
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        try:
            v = object.__getattribute__(self, "d")[key]
        except KeyError:
            raise OptionError("No such option")
        if isinstance(v, dict):
            return DictWrapper(v, prefix)
        else:
            return _get_option(prefix)

    def __dir__(self):
        return list(self.d.keys())


class CallableDynamicDoc:
    def __init__(self, func, doc_templ):
        self.__doc_tmpl__ = doc_templ
        self.__func__ = func

    def __call__(self, *args, **kwargs):
        return self.__func__(*args, **kwargs)

    @property
    def __doc__(self):
        opts_desc = _describe_option("all", _print_desc=False)
        opts_list = pp_options_list(list(_registered_options.keys()))

        return self.__doc_tmpl__.format(opts_desc=opts_desc, opts_list=opts_list)


_get_option_tmpl = """
get_option(pat)

Retrieves the value of the specified option.

Available options:

{opts_list}

Parameters
----------
pat : str
    Regexp which should match a single option.
    Note: partial matches are supported for convenience, but unless you use the
    full option name (e.g. x.y.z.option_name), your code may break in future
    versions if new options with similar names are introduced.

Returns
-------
result : the value of the option

Raises
------
OptionError : if no such option exists

Notes
-----
The available options with its descriptions:

{opts_desc}
"""

_set_option_tmpl = """
set_option(pat, value)

Sets the value of the specified option.

Available options:

{opts_list}

Parameters
----------
pat : str
    Regexp which should match a single option.
    Note: partial matches are supported for convenience, but unless you use the
    full option name (e.g. x.y.z.option_name), your code may break in future
    versions if new options with similar names are introduced.
value : object
    New value of option.

Returns
-------
None

Raises
------
OptionError if no such option exists

Notes
-----
The available options with its descriptions:

{opts_desc}
"""

_describe_option_tmpl = """
describe_option(pat, _print_desc=False)

Prints the description for one or more registered options.

Call with not arguments to get a listing for all registered options.

Available options:

{opts_list}

Parameters
----------
pat : str
    Regexp pattern. All matching keys will have their description displayed.
_print_desc : bool, default True
    If True (default) the description(s) will be printed to stdout.
    Otherwise, the description(s) will be returned as a unicode string
    (for testing).

Returns
-------
None by default, the description(s) as a unicode string if _print_desc
is False

Notes
-----
The available options with its descriptions:

{opts_desc}
"""

_reset_option_tmpl = """
reset_option(pat)

Reset one or more options to their default value.

Pass "all" as argument to reset all options.

Available options:

{opts_list}

Parameters
----------
pat : str/regex
    If specified only options matching `prefix*` will be reset.
    Note: partial matches are supported for convenience, but unless you
    use the full option name (e.g. x.y.z.option_name), your code may break
    in future versions if new options with similar names are introduced.

Returns
-------
None

Notes
-----
The available options with its descriptions:

{opts_desc}
"""

get_option = CallableDynamicDoc(_get_option, _get_option_tmpl)
set_option = CallableDynamicDoc(_set_option, _set_option_tmpl)
reset_option = CallableDynamicDoc(_reset_option, _reset_option_tmpl)
describe_option = CallableDynamicDoc(_describe_option, _describe_option_tmpl)
options = DictWrapper(_global_config)


@contextmanager
def config_prefix(prefix):
    """contextmanager for multiple invocations of API with a common prefix
    supported API functions: (register / get / set )__option
    Warning: This is not thread - safe, and won't work properly if you import
    the API functions into your module using the "from x import y" construct.
    Example:
    import pandas._config.config as cf
    with cf.config_prefix("display.font"):
        cf.register_option("color", "red")
        cf.register_option("size", " 5 pt")
        cf.set_option(size, " 6 pt")
        cf.get_option(size)
        ...
        etc'
    will register options "display.font.color", "display.font.size", set the
    value of "display.font.size"... and so on.
    """

    # Note: reset_option relies on set_option, and on key directly
    # it does not fit in to this monkey-patching scheme

    global register_option, get_option, set_option, reset_option

    def wrap(func):
        def inner(key, *args, **kwds):
            pkey = "{prefix}.{key}".format(prefix=prefix, key=key)
            return func(pkey, *args, **kwds)

        return inner

    _register_option = register_option
    _get_option = get_option
    _set_option = set_option
    set_option = wrap(set_option)
    get_option = wrap(get_option)
    register_option = wrap(register_option)
    yield None
    set_option = _set_option
    get_option = _get_option
    register_option = _register_option


def is_type_factory(_type):
    """
    Parameters
    ----------
    `_type` - a type to be compared against (e.g. type(x) == `_type`)
    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if type(x) is not equal to `_type`
    """

    def inner(x):
        if type(x) != _type:
            msg = "Value must have type '{typ!s}'"
            raise ValueError(msg.format(typ=_type))

    return inner


is_bool = is_type_factory(bool)
is_list = is_type_factory(list)
