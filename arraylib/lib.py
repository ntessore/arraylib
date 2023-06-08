"""
.. autofunction::   abs
.. autofunction::   asinh
.. autofunction::   asarray
.. autofunction::   astype
.. autofunction::   ceil
.. autofunction::   choose
.. autofunction::   cos
.. autofunction::   exp
.. autofunction::   finfo
.. autofunction::   floor
.. autofunction::   frexp
.. autofunction::   invsinpi
.. autofunction::   isfinite
.. autofunction::   isinf
.. autofunction::   isnan
.. autofunction::   ldexp
.. autofunction::   log
.. autofunction::   log2
.. autofunction::   loggamma
.. autofunction::   logsinpi
.. autofunction::   max
.. autofunction::   min
.. autodata::       PI
.. autodata::       pi
.. autofunction::   round
.. autofunction::   sin
.. autofunction::   sinpi
.. autofunction::   sum
.. autofunction::   ulp
.. autofunction::   where

"""

from __future__ import annotations

__all__ = (
    "abs",
    "asinh",
    "asarray",
    "astype",
    "ceil",
    "choose",
    "cos",
    "exp",
    "finfo",
    "floor",
    "frexp",
    "invsinpi",
    "isfinite",
    "isinf",
    "isnan",
    "ldexp",
    "log",
    "log2",
    "loggamma",
    "logsinpi",
    "max",
    "min",
    "PI",
    "pi",
    "round",
    "sin",
    "sinpi",
    "sum",
    "ulp",
    "where",
)

from collections import namedtuple


# attempt to import these dtypes from the array namespace
_dtypes = namedtuple("_dtypes", "int, uint, float, complex")(
    ("int8", "int16", "int32", "int64"),
    ("uint8", "uint16", "uint32", "uint64"),
    ("float16", "float32", "float64", "float96", "float128"),
    ("complex32", "complex64", "complex96", "complex128", "complex192",
     "complex256")
)


def _alias(*names: str):
    """Decorator to give array namespace function aliases."""
    def decorator(func):
        setattr(func, "al_alias", names)
        return func
    return decorator


def _extern(func):
    """Decorator for functions to be provided by the array namespace."""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        __tracebackhide__ = __traceback_hide__ = True
        name = func.__name__
        exc = NotImplementedError(name)
        if hasattr(exc, "add_note"):
            exc.add_note(f"""
            The function `{name}` must be provided by the array namespace.
            """)
        raise exc
    return wrapper


def _notimplemented(func):
    """Decorator for functions without arraylib implementation."""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        __tracebackhide__ = __traceback_hide__ = True
        name = func.__name__
        exc = NotImplementedError(name)
        if hasattr(exc, "add_note"):
            exc.add_note(f"""
            The function `{name}` is missing from arraylib.
            Please open an issue in arraylib's GitHub repository.
            """)
        raise exc
    return wrapper


@_extern
def abs(z: float|complex, /) -> float|complex:
    """Absolute value *|z|*.

    .. versionadded:: 0.0.0

    """


@_alias("arcsinh")
@_extern
def asinh(z: float|complex, /) -> float|complex:
    """Inverse function of hyperbolic sine.

    .. versionadded:: 0.0.0

    """


@_extern
def asarray(obj: bool|int|float|complex, /, *, dtype: dtype|None = None
            ) -> bool|int|float|complex:
    """Create an array from the input.

    .. versionadded:: 0.0.0

    """


def astype(x: array, dtype: dtype, /, *, copy=True) -> array:
    """Copy the array *x* as dtype *dtype*.

    .. versionadded:: 0.0.0

    """
    return x.astype(dtype, copy=copy)


@_extern
def ceil(z: float|complex, /) -> float|complex:
    """Ceiling function *⌈z⌉*.

    .. versionadded:: 0.0.0

    """


@_extern
def choose(a: int, choices: list[bool|int|float|complex], /
           ) -> bool|int|float|complex:
    """Choose array entries from index array *a* and choices *choices*.

    .. versionadded:: 0.0.0

    """


@_extern
def cos(z: float|complex, /) -> float|complex:
    """Cosine function *cos(z)*.

    .. versionadded:: 0.0.0

    """


@_extern
def exp(z: float|complex, /) -> float|complex:
    """Exponential function *exp(z)*.

    .. versionadded:: 0.0.0

    """


@_extern
def finfo(obj: dtype|float|complex, /) -> finfo_type:
    """Return information about floating point types.

    .. versionadded:: 0.0.0

    """


@_extern
def floor(z: float|complex, /) -> float|complex:
    """Floor function *⌊z⌋*.

    .. versionadded:: 0.0.0

    """


@_notimplemented
def frexp(x: float, /) -> tuple[float, int]:
    """Decompose floating point numbers *x* into factor and exponent.

    .. versionadded:: 0.0.0

    """


# The reciprocal of sinpi, useful with gamma functions, due to the
# reflection formula Γ(z) = π/sin(πz)/Γ(1-z).
#
# References
# ----------
# [1] Johansson, F. (2021). Arbitrary-precision computation of the gamma
# function. arXiv, 2109.08392v1.
#
@_notimplemented
def invsinpi(z: float|complex, /) -> float|complex:
    """Inverse sine function *1/sin(πz)*.

    .. versionadded:: 0.0.0

    """


@_extern
def isfinite(z: float|complex, /) -> bool:
    """Return where the input array *z* is finite.

    .. versionadded:: 0.0.0

    """


@_extern
def isinf(z: float|complex, /) -> bool:
    """Return where the input array *z* is infinite.

    .. versionadded:: 0.0.0

    """


@_extern
def isnan(z: float|complex, /) -> bool:
    """Return where the input array *z* is not a number.

    .. versionadded:: 0.0.0

    """


@_notimplemented
def ldexp(x1: float, x2: int, /) -> float:
    """Load exponent *x2* of the floating points numbers *x1*.

    .. versionadded:: 0.0.0

    """


@_extern
def log(z: float|complex, /) -> float|complex:
    """Natural logarithm *ln(z)*.

    .. versionadded:: 0.0.0

    """


@_extern
def log2(z: float|complex, /) -> float|complex:
    """Logarithm in base 2 *log2(z)*.

    .. versionadded:: 0.0.0

    """


# The log-sine function ln sin(πz), useful with log-gamma functions, due
# to the reflection formula ln Γ(z) = ln π - ln Γ(1-z) - ln sin(πz).
#
# References
# ----------
# [1] Johansson, F. (2021). Arbitrary-precision computation of the gamma
# function. arXiv, 2109.08392v1.
#
@_notimplemented
def logsinpi(z, /):
    """Log-sine function *ln sin(πz)*.

    .. versionadded:: 0.0.0

    """


# References
# ----------
# [1] Johansson, F. (2021). Arbitrary-precision computation of the gamma
# function. arXiv, 2109.08392v1.
#
@_notimplemented
def loggamma(z, /):
    """Log-gamma function *ln Γ(z)*.

    .. versionadded:: 0.0.0

    """


@_extern
def max(x: bool|int|float, /, *, axis: int|tuple[int, ...]|None = None,
        keepdims: bool = False) -> bool|int|float:
    """Maximum element function *max(x)*.

    .. versionadded:: 0.0.0

    """


@_extern
def min(x: bool|int|float, /, *, axis: int|tuple[int, ...]|None = None,
        keepdims: bool = False) -> bool|int|float:
    """Minimum element function *min(x)*.

    .. versionadded:: 0.0.0

    """


# requires decimals to convert to the widest floating point type
PI: str = "3.14159265358979323846264338327950288419716939937510582097494"
"""Constant *π* as a high-precision string.

.. versionadded:: 0.0.0

"""


# float version of PI
pi: float = float(PI)
"""Constant *π* in float precision.

.. versionadded:: 0.0.0

"""


@_extern
def round(z: float|complex, /) -> float|complex:
    """Round to nearest integer.

    .. versionadded:: 0.0.0

    """


@_extern
def sin(z: float|complex, /) -> float|complex:
    """Sine function *sin(z)*.

    .. versionadded:: 0.0.0

    """


# The sinpi function, useful with gamma functions, due to the reflection
# formula 1/Γ(z) = (1/π) sin(πz) Γ(1-z).
#
def sinpi(z: float|complex, /) -> float|complex:
    """Sine function *sin(πz)*.

    .. versionadded:: 0.0.0

    """
    n = round(2*z.real)
    y = pi * (z - n/2)
    s = sin(y)
    c = cos(y)
    n = astype(n % 4, int8)
    return choose(n, [s, c, -s, -c])


@_extern
def sum(z: bool|int|float|complex, /, *,
        axis: int|tuple[int, ...]|None = None,
        dtype: dtype = None, keepdims: bool = False
        ) -> bool|int|float|complex:
    """Sum the input array.*.

    .. versionadded:: 0.0.0

    """


def ulp(z: float|complex, /) -> float:
    """Unit in the last place of input.

    .. versionadded:: 0.0.0

    """
    fp = finfo(z.dtype)
    one = asarray(1, dtype=fp.dtype)
    two = asarray(2, dtype=fp.dtype)
    xmin = two*two/fp.max
    xabs = abs(z)
    _, p = frexp(one/fp.eps)
    _, e = frexp(where(xmin > xabs, xmin, xabs))
    return where(isfinite(z), ldexp(one, e - p), xabs)


@_extern
def where(cond: bool, x1: bool|int|float|complex,
          x2: bool|int|float|complex, /) -> bool|int|float|complex:
    """Return entries from *x1* where *cond* is true, and from *x2* otherwise.

    .. versionadded:: 0.0.0

    """
