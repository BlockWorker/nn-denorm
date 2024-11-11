from setuptools import Extension, setup
from Cython.Build import cythonize
import sys

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'


ext_modules = [
    Extension(
        "*",
        ["fastconv.py"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    ),
    Extension(
        "*",
        ["fastconv.py"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    )
]


setup(
    ext_modules=cythonize(ext_modules),
)