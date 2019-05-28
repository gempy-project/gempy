
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


def _setdoc_pro(docstring):
    if type(docstring) is not list:
        docstring = [docstring]
        # try:
        #     docstring = '-'.join(docstring)
        # except TypeError:
        #     raise TypeError(str(docstring))

    def decor(func):

        if func.__doc__ is None:

            func.__doc__ = docstring
        else:
            for e, i in enumerate(docstring):
                # Find the location on the target method to insert comment
                marker = '[s'+str(e)+']'
                loc_0 = func.__doc__.find(marker) + len(marker)
                # Find the first paragraph of the docstring
                # look for \n
                break_loc = i.find('\n')
                if break_loc == 0:
                    break_loc = i[:2].find('\n') + 2

                text = i[:break_loc]
                if text == -1:
                    text = 'No break found'
                func.__doc__ = func.__doc__[:loc_0] + '-(inserted)-' + text + func.__doc__[loc_0:]
        return func

    return decor
# endregion
