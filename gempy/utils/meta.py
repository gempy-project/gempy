
# region Auxiliary


def _setdoc(docstring):
    if type(docstring) is list:
        try:
            docstring = '-'.join(docstring)
        except TypeError:
            raise TypeError(str(docstring))

    def decor(func):

        if func.__doc__ is None:

            func.__doc__ = docstring
        else:
            func.__doc__ += '\n' + docstring

        return func

    return decor
# endregion
