"""
*arraylib* --- Library for array computing in Python
====================================================

Namespace functions
-------------------

.. autofunction:: arraylib.bind

Array functions
---------------

.. automodule:: arraylib.lib

"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = (
    "bind",
    "use",
)

from contextlib import contextmanager as _contextmanager
from . import lib


class _Namespace:
    def __init__(self, ns: dict, name=None) -> None:
        self.__name__ = name or "???"
        self.__dict__.update(ns)

    def __repr__(self) -> str:
        return f"arraylib({self.__name__})"


def bind(xp, /, **aliases):
    """Return a namespace with arraylib functions bound to *xp*.

    .. versionadded:: 0.0.0

    """
    from types import FunctionType
    # globals for the new module
    ns = {
        "__builtins__": __builtins__,
    }
    # load the available dtypes into the namespace
    dtypes = []
    for names in lib._dtypes:
        loaded = []
        for name in names:
            try:
                dtype = getattr(xp, name)
            except AttributeError:
                pass
            else:
                ns[name] = dtype
                loaded.append(dtype)
        dtypes.append(tuple(loaded))
    # store the dtypes summary information
    ns["_dtypes"] = type(lib._dtypes)(*dtypes)
    # load the library
    for name in lib.__all__:
        # skip arraylib meta-functions
        if name in ("bind",):
            continue
        # check explicit alias given
        if name in aliases:
            ns[name] = aliases[name]
            continue
        # find name in array namespace
        try:
            ns[name] = getattr(xp, name)
        except AttributeError:
            pass
        else:
            continue
        # get arraylib implementation
        obj = getattr(lib, name)
        # search for aliases
        for alias in getattr(obj, "al_alias", ()):
            try:
                ns[name] = getattr(xp, alias)
            except AttributeError:
                pass
            else:
                break
        # no implementation found, use arraylib's
        else:
            # modify functions to use ns as their globals
            if isinstance(obj, FunctionType):
                newobj = FunctionType(obj.__code__, ns, obj.__name__,
                                      obj.__defaults__, obj.__closure__)
                newobj.__kwdefaults__ = obj.__kwdefaults__
                ns[name] = newobj
            # copy everything else
            else:
                ns[name] = obj
    # construct and return the namespace object
    name = getattr(xp, "__name__", None)
    return _Namespace(ns, name)


@_contextmanager
def use(xp, /, **aliases):
    """Context manager for binding *xp* to the arraylib module.

    .. versionadded:: 0.0.0

    """
    glob = globals()
    ns = bind(xp, **aliases)
    try:
        glob.update(ns.__dict__)
        yield
    finally:
        for key in ns.__dict__:
            del glob[key]
