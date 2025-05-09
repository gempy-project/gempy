import gempy.core.color_generator as cg


def test_color_generator():

    # create an instance of ColorsGenerator
    color_iterator: iter = cg.ColorsGenerator()

    # generate color dictionary (with optional seaborn palettes if you want)

    # get the generator from next_color method

    # now you can get colors one by one
    print(next(color_iterator))  # prints first color
    print(next(color_iterator))  # prints second color

    left_colors = color_iterator._gempy_default_colors[2:]

    assert left_colors == [
        color for color, _ in zip(color_iterator, left_colors)
    ]

    # get a random color
    assert color_iterator.up_next() == next(color_iterator)

    # override the random color generator
    cg.np.random.randint = lambda a, b: 0x00ff42

    assert color_iterator._random_hexcolor() == "#00ff42"
    assert next(color_iterator) == "#00ff42"

    # trigger up_next method for the cache
    assert color_iterator.up_next() == "#00ff42"

    cg.np.random.randint = lambda a, b: 0x000000

    # ensure that the up_next method still returns the same color
    assert color_iterator.up_next() == "#00ff42"

    # trigger the cache
    assert next(color_iterator) == "#00ff42"
    assert next(color_iterator) == "#000000"

    # etc...
