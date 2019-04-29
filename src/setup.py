# -*- coding: utf-8 -*-
r"""
    src.setup
    ~~~~~~~~~

    Setup needed for continuous integration.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""

from setuptools import setup


setup(
    name="stratego",
    version="0.0.0",
    author="Zayd Hammoudeh",
    packages=['stratego'],
    install_requires=['sty']
)
