import doctest
import os
import sys
from pathlib import Path


def test_docs():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    for py_pth in Path("../stratego/").iter_dir():
        if not py_pth.is_file() or py_pth.suffix != ".py":
            continue
        doctest.testfile(py_pth)

