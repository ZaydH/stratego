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
    for dir in [l.up(), l.right(), l.down(), l.left()]:
        assert dir in l.neighbors()
