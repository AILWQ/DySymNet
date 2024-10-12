#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree


import setuptools


# Package meta-data.
NAME = 'DySymNet'
DESCRIPTION = 'This package contains the official Pytorch implementation for the paper "A Neural-Guided Dynamic Symbolic Network for Exploring Mathematical Expressions from Data" accepted by ICML\'24.'
URL = 'https://github.com/AILWQ/DySymNet'
EMAIL = 'liwenqiang2021@gmail.com'
AUTHOR = 'Wenqiang Li'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.2.0'

# What packages are required for this module to be executed?
REQUIRED = [
    'scikit-learn==1.5.2',
    'numpy==1.26.4',
    'sympy==1.13.3',
    'torch==2.2.2',
    'matplotlib==3.9.2',
    'tqdm==4.66.5',
    'pandas==2.2.3',
    'pip==24.2',
    'scipy==1.13.1'
]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Where the magic happens:
setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(),
    install_requires=REQUIRED,
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",

    ]
)