#!/usr/bin/env python

import numpy as np
import setuptools
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension('jsonvectorizer.jsontype', ['jsonvectorizer/jsontype.pyx']),
    Extension('jsonvectorizer.lil', ['jsonvectorizer/lil.pyx']),
    Extension('jsonvectorizer.schema', ['jsonvectorizer/schema.pyx']),
    Extension(
        'jsonvectorizer.jsonvectorizer', ['jsonvectorizer/jsonvectorizer.pyx']
    )
]
for extension in extensions:
    extension.cython_directives = {'embedsignature': True}

with open('requirements.txt') as f:
    install_requires = f.read()

setuptools.setup(
    name='jsonvectorizer',
    version='0.1.0',
    packages=setuptools.find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    package_data={'jsonvectorizer': ['*.pxd']},
    install_requires=install_requires
)
