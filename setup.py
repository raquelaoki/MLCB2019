#python setup.py build_ext --inplace
from distutils.core import setup#, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='simulations',
    ext_modules = cythonize("simulations.pyx",language_level=3,language='c++'), #nthreads
    include_dirs=[numpy.get_include()]
)