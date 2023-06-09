import operator
import random
import pytest
import arraylib

from mpmath import mp, mpf, mpc

xp = arraylib.bind(__import__('numpy'))


def _show(x):
    if x.imag:
        s = "-" if x.imag < 0 else "+"
        return f"{x.real!s} {s} I {abs(x.imag)!s}"
    else:
        return f"{x.real!s}"


def mpfloat(dtype, func=None, /):
    fp = xp.finfo(dtype)

    def decorator(func):
        @mp.workprec(fp.bits)
        def wrapper(x):
            with mp.extraprec(fp.bits):
                y = func(mpf(str(x)))
            return xp.asarray(str(y), dtype=dtype)
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def mpcomplex(dtype, func=None, /):
    fp = xp.finfo(dtype)

    def decorator(func):
        @mp.workprec(fp.bits)
        def wrapper(z):
            with mp.extraprec(fp.bits):
                w = func(mpc(str(z.real), str(z.imag)))
            return xp.asarray(str(w).replace(' ', ''), dtype=dtype)
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def flim(dtype):
    fp = xp.finfo(dtype)
    four = xp.asarray(4, dtype=fp.dtype)
    return four/fp.max, fp.max, fp.eps


def uniform(lower, upper, size):
    delta = upper - lower
    return xp.asarray([lower + delta * random.random()
                      for _ in range(size)], dtype=delta.dtype)


def loguniform(lower, upper, size, sign=True):
    r = xp.exp(uniform(xp.log(lower), xp.log(upper), size))
    if sign:
        s = xp.asarray([random.choice([-1, 1]) for _ in range(size)],
                       dtype=r.dtype)
    else:
        s = xp.asarray(1, dtype=r.dtype)
    return s*r


def assert_op(op, a, b):
    __tracebackhide__ = True
    result = op(a, b)
    if not xp.all(result):
        bad_n = xp.sum(~result, axis=None)
        bad_a = a[~result][0]
        bad_b = b[~result][0]
        msg = (f"a not {op.__name__} b [{bad_n}/{result.size}]\n"
               f"{_show(bad_a)} not {op.__name__} {_show(bad_b)}")
        raise AssertionError(msg)


def assert_eq(a, b):
    assert_op(operator.eq, a, b)


def assert_gt(a, b):
    assert_op(operator.gt, a, b)


def assert_fn_equal(name, z, got, expected, nulp=5):
    __tracebackhide__ = True
    err = xp.abs(got - expected)/xp.ulp(expected)
    max_err = xp.max(err)
    if not max_err <= nulp:
        bad = (err > nulp)
        n = xp.sum(err > nulp)
        msg = f"result not equal to {nulp} ulp [{n}/{z.size}]"
        for i in range(min(n, 3)):
            bad_z = z[bad][i]
            bad_got = got[bad][i]
            bad_expected = expected[bad][i]
            bad_err = err[bad][i]
            msg += (f"\n---"
                    f"\n{name}({_show(bad_z)})"
                    f"\n= {_show(bad_got)}"
                    f"\n≠ {_show(bad_expected)} [{bad_err:.0f} ulp]")
        raise AssertionError(msg)


@pytest.mark.parametrize("dtype", xp._dtypes.float)
def test_sinpi_float(dtype):
    fmin, fmax, feps = flim(dtype)
    one = xp.asarray(1, dtype=dtype)

    @mpfloat(dtype)
    def f(x):
        return mp.sinpi(x)

    assert xp.ulp(one/feps) == 1

    x = loguniform(fmin, one/feps, 1000)

    expected = xp.asarray([f(xi) for xi in x], dtype=dtype)

    assert x.dtype == dtype
    got = xp.sinpi(x)
    assert got.dtype == dtype

    assert_fn_equal("sinpi", x, got, expected)


@pytest.mark.parametrize("dtype", xp._dtypes.complex)
def test_sinpi_complex(dtype):
    fmin, fmax, feps = flim(dtype)
    one = xp.asarray(1, dtype=dtype)
    pi = xp.PI(dtype)
    j = xp.asarray(1j, dtype=dtype)

    @mpcomplex(dtype)
    def f(z):
        return mp.sinpi(z)

    assert xp.ulp(one/feps) == 1

    x = loguniform(fmin, one/feps, 1000)
    y = loguniform(fmin, xp.asinh(fmax)/pi, 1000)
    z = x + j*y

    expected = xp.asarray([f(zi) for zi in z], dtype=dtype)

    assert z.dtype == dtype
    got = xp.sinpi(z)
    assert got.dtype == dtype

    assert_fn_equal("sinpi", z, got, expected)


@pytest.mark.parametrize("dtype", xp._dtypes.float + xp._dtypes.complex)
def test_ulp(dtype):
    zero = xp.asarray(0, dtype=dtype)
    one = xp.asarray(1, dtype=dtype)
    two = xp.asarray(2, dtype=dtype)
    inf = xp.asarray("inf", dtype=dtype)
    nan = xp.asarray("nan", dtype=dtype)

    fp = xp.finfo(dtype)

    ulp_of_one = xp.ulp(one)
    assert ulp_of_one == fp.eps
    assert ulp_of_one.dtype == fp.dtype

    assert xp.ulp(zero) > 0

    assert xp.ulp(inf) == inf
    assert xp.ulp(-inf) == inf

    assert xp.isnan(xp.ulp(nan))

    x = loguniform(two*two/fp.max, fp.max, 1000)
    u = xp.ulp(x)

    assert_gt(x + u, x)
    assert_eq(x + u/two/two, x)


@pytest.mark.parametrize("dtype", [float, complex])
def test_ulp_pymath(dtype):
    import math
    import sys
    assert xp.ulp(xp.asarray(0, dtype)) == math.ulp(0)
    emax = math.log2(sys.float_info.max)
    emin = 1 - emax
    for _ in range(1000):
        y = random.choice([-1, 1]) * 2**random.uniform(emin, emax)
        x = xp.asarray(y, dtype=dtype)
        assert xp.ulp(x) == math.ulp(y)
