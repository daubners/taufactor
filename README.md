<p align="center">
<a href="https://pypi.python.org/pypi/taufactor">
        <img src="https://img.shields.io/pypi/v/taufactor.svg"
            alt="PyPI"></a>
<a href="https://taufactor.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/taufactor/badge/?version=latest"
            alt="ReadTheDocs"></a>
<a href="https://pyup.io/repos/github/tldr-group/taufactor/">
        <img src="https://pyup.io/repos/github/tldr-group/taufactor/shield.svg"
            alt="PyUp"></a>
<a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg"
            alt="MIT LICENSE"></a>
</p>

# TauFactor
TauFactor is an application for calculating tortuosity factors from tomographic data. TauFactor uses CuPy_ which is an implementation of NumPy-compatible multi-dimensional array on CUDA.

* Free software: MIT license
* Documentation: https://taufactor.readthedocs.io.


<img src="docs/tau_example.png" width="324" height="324">

# Requirements
**You will need an NVIDIA GPU to use this distribution of taufactor.** <br />
Before installing taufactor, download the most recent version of CuPy:
https://docs.cupy.dev/en/stable/install.html

# Usage
A basic example for taufactor:
```python
import taufactor as tau

# load image
img = tifffile.imread('path/filename')
# ensure 1s for conductive phase and 0s otherwise.
# here we perform an example segmentation on a grayscale img
img[img > 0.7] = 1
img[img != 1] = 0
# create a solver object
s = tau.Solver(img)
# call solve function
D_rel = s.solve()
```

We can also use the periodic solver

```python
import taufactor as tau

# create a periodic solver object and set an iteration limit
s = tau.PeriodicSolver(img, iter_limit=1000)
# call solve function
D_rel = s.solve()
```
# Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
