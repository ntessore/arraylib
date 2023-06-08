*arraylib* — Library for array computing in Python
==================================================

The *arraylib* package aims to provide a standardised interface to array
implementations such as *NumPy*, *JAX*, *PyTorch*, etc.

The standard array functions are modelled on the *Array API*, and should be
compatible with it.

The goal of the *arraylib* package is to extend the standard with many of the
special functions which are necessary for scientific computing.

Usage
-----

Install the *arraylib* package and an array implementation, for example
*NumPy*.  Load both, then obtain an *arraylib* namespace by binding
the `numpy` array namespace:

```py
>>> import arraylib
>>> import numpy as np
>>> xp = arraylib.bind(np)
>>> xp.log
<ufunc 'log'>
>>> xp.log is np.log
True
>>> xp.sinpi(2.5)
1.0
```
