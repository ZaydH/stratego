import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "strategp",
    version = "0.0.0",
    author = "Zayd Hammoudeh",
    packages=['stratego']
)
