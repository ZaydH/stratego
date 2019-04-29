# -*- coding: utf-8 -*-
r"""
    tests.test_location
    ~~~~~~~~~~~~~~~~~~~

    Verify the basic functions of the Stratego \p Location class.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""

from stratego.location import Location


def test_directions():
    r""" Verify the four direction operations """
    l = Location(1, 1)
    assert l.up() == l.relative(row_diff=-1)
    assert l.right() == l.relative(col_diff=1)
    assert l.down() == l.relative(row_diff=1)
    assert l.left() == l.relative(col_diff=-1)


def test_neighbors():
    r""" Verify the neighbors set contains each direction """
    l = Location(1, 1)
    for direct in [l.up(), l.right(), l.down(), l.left()]:
        assert direct in l.neighbors()
