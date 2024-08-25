from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    # name="c_breakout",
    include_package_data=True,
    ext_modules=cythonize(
        ["breakout/c_breakout.pyx", "c_gae.pyx"],
        # annotate=True,
    ),
    include_dirs=[numpy.get_include()],
    packages=["breakout"],
)
