
# region Auxiliary


def setdoc(docstring, indent=True, position='end'):
    if type(docstring) is list:
        try:
            docstring = '----'.join(docstring)
        except TypeError:
            raise TypeError(str(docstring))

    def decor(func):

        if func.__doc__ is None:

            func.__doc__ = docstring
        else:
            if indent is True:
                aux = docstring.replace('\n', '\n\n        ')
                func.__doc__ += ' (inserted) \n        ' + aux
            else:
                if position == 'end':
                    func.__doc__ += ' (inserted) ' + docstring
                else:
                    func.__doc__ = ' (inserted) - ' + docstring +'\n'+ func.__doc__
        return func

    return decor


def setdoc_pro(docstring):
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
                if loc_0 == -1:
                    print('marker not found')
                # Remove \n if it is the first
                # if i[:1] == '\n':
                #     i = i[1:]
                # Find the first paragraph of the docstring
                # look for \n
                break_loc = i.find('\n\n')
                if break_loc == 0:
                    break_loc = i[2:].find('\n\n') + 2

                text = i[:break_loc]
                if text == -1:
                    text = 'No break found'
                text = text.replace('\n    ', '')
               # print(loc_0, text)
                func.__doc__ = func.__doc__[:loc_0] + ' -(inserted)- ' + text + func.__doc__[loc_0:]
        return func

    return decor
# endregion
