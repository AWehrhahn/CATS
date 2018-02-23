"""
Build Cython modules
"""
import os
from os import path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

def clean(ext):
    for pyx in ext.sources:
        if pyx.endswith('.pyx'):
            c = pyx[:-4] + '.c'
            cpp = pyx[:-4] + '.cpp'
            so = pyx[:-4] + '.so'
            if os.path.exists(so):
                os.unlink(so)
            if os.path.exists(c):
                os.unlink(c)
            elif os.path.exists(cpp):
                os.unlink(cpp)


extensions = [
    Extension("solution", ["solution.py"]),
    Extension("intermediary", ["intermediary.pyx"]),
    Extension("marcs", ["marcs.py"]),
    Extension("synthetic", ["synthetic.py"]),
    Extension("idl", ["idl.py"]),
    Extension("harps", ["harps.py"]),
    Extension("dataset", ["dataset.py"]),
    Extension("data_module_interface", ["data_module_interface.py"]),

]

setup(
    name="ExoSpectrum",
    ext_modules=cythonize(extensions,
    compiler_directives={"infer_types": True})
)
