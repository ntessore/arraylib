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


def uniform(lower, upper, size, dtype):
    delta = upper - lower
    return xp.asarray([lower + delta * random.random()
                      for _ in range(size)], dtype=dtype)


def loguniform(lower, upper, size, dtype, sign=True):
    log_a, log_b = xp.log(lower), xp.log(upper)
    orign = (log_a + log_b)/2
    delta = log_b - log_a
    x = [xp.exp(orign + delta*random.uniform(-0.5, 0.5))
         for _ in range(size)]
    if sign:
        x = [random.choice([-xi, xi]) for xi in x]
    return xp.asarray(x, dtype=dtype)


def assert_fn_equal(name, z, got, expected, nulp=5):
    __tracebackhide__ = True
    err = xp.abs(got - expected)/xp.ulp(expected)
    min_err, max_err = xp.min(err), xp.max(err)
    if max_err > nulp:
        bad = (err > nulp)
        bad_n = xp.sum(bad)
        bad_z = z[bad][0]
        bad_got = got[bad][0]
        bad_expected = expected[bad][0]
        bad_err = err[bad][0]
        msg = (f"result not equal to {nulp} ulp [{bad_n}/{z.size}, "
               f"min={min_err:.0f}, max={max_err:.0f}]\n"
               f"{name}({_show(bad_z)}) \n"
               f"= {_show(bad_got)}\n"
               f"≠ {_show(bad_expected)} [{bad_err:.0f} ulp]")
        raise AssertionError(msg)


@pytest.mark.parametrize("dtype", xp._dtypes.float)
def test_sinpi_float(dtype):
    fp = xp.finfo(dtype)
    fmin, fmax = 4/fp.max, fp.max

    @mp.workprec(fp.bits*2)
    def f(x):
        return mp.sinpi(mpf(str(x)))

    xmin, xmax = fmin, fmax/xp.pi

    x = loguniform(xmin, xmax, 1000, dtype)

    expected = xp.asarray(list(map(f, x)), dtype=dtype)
    got = xp.sinpi(x)
    assert_fn_equal("sinpi", x, got, expected)


@pytest.mark.parametrize("dtype", xp._dtypes.complex)
def test_sinpi_complex(dtype):
    fp = xp.finfo(dtype)
    fmin, fmax = 4/fp.max, fp.max

    @mp.workprec(fp.bits*2)
    def f(x):
        return mp.sinpi(mpc(str(x.real), str(x.imag)))

    xmin, xmax = fmin, fmax/xp.pi
    ymin, ymax = fmin, xp.asinh(fmax)/xp.pi

    x = loguniform(xmin, xmax, 1000, dtype)
    y = loguniform(ymin, ymax, 1000, dtype)
    z = x + xp.asarray(1j, dtype=dtype)*y

    expected = xp.asarray(list(map(f, z)), dtype=dtype)
    got = xp.sinpi(z)
    assert_fn_equal("sinpi", z, got, expected)


@pytest.mark.parametrize("dtype", xp._dtypes.float + xp._dtypes.complex)
def test_ulp(dtype):
    zero = xp.asarray(0, dtype)
    one = xp.asarray(1, dtype)
    inf = xp.asarray("inf", dtype=dtype)
    nan = xp.asarray("nan", dtype=dtype)

    fp = xp.finfo(dtype)

    ulp_of_one = xp.ulp(one)
    assert ulp_of_one == fp.eps
    assert ulp_of_one.dtype == fp.dtype

    assert xp.ulp(zero) > 0

    assert xp.isinf(xp.ulp(inf))
    assert xp.isinf(xp.ulp(-inf))

    assert xp.isnan(xp.ulp(nan))


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
