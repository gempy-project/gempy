from gempy.core.color_generator import ColorsGenerator


def test_color_generator():

    # create an instance of ColorsGenerator
    color_iterator: iter = ColorsGenerator()

    # generate color dictionary (with optional seaborn palettes if you want)

    # get the generator from next_color method

    # now you can get colors one by one
    print(next(color_iterator))  # prints first color
    print(next(color_iterator))  # prints second color
    # etc...
