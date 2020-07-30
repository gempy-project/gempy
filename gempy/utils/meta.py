from . import docstring as doc_args
import re

# region Auxiliary


def _setdoc(docstring, indent=True, position='end'):
    """Copy the doc of a function or method into the decorated function.

    Args:
        docstring: docstring of to be added
        indent:
        position:

    Returns:

    """
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
                stop = aux.find('\nExamples\n-----')
                func.__doc__ += ' (inserted) \n        ' + aux[:stop]
            else:
                stop = docstring.find('\nExamples\n-----')
                if position == 'end':

                    func.__doc__ += docstring[:stop]
                else:
                    func.__doc__ = docstring[:stop] + '\n'+ func.__doc__
        return func

    return decor


def _setdoc_pro(docstring=[]):
    """This takes a list and places where it finds [s+0]"""
    if type(docstring) is not list:
        docstring = [docstring]

    def decor(func):

        if func.__doc__ is None:
            raise AttributeError('Add """"""" to the docstring')
           # print(func.__doc__)
        else:
            # Loop for the docstrings we pass looking for numbers
            for e, i in enumerate(docstring):
                # Find the location on the target method to insert comment
                marker = '[s'+str(e)+']'
                loc_0 = func.__doc__.find(marker) + len(marker)
                if loc_0 == -1:
                    print('marker not found')
                break_loc = i.find('\n\n')
                if break_loc == 0:
                    break_loc = i[2:].find('\n\n') + 2

                text = i[:break_loc]
                if text == -1:
                    text = 'No break found'
                text = text.replace('\n    ', '')
                func.__doc__ = func.__doc__[:loc_0] + ' '+ text + func.__doc__[loc_0:]

        # Find all the arg_string markers
        loc_1 = 0
        text = []
        marker = '[s_'
        for e in range(10):
            #print(len(func.__doc__), func.__doc__)
            #print('loc_1', loc_1)
            #print(func.__doc__)
            loc_0 = func.__doc__[loc_1:].find(marker)
            if loc_0 == -1:
                break
            else:
                loc_0 +=  loc_1
            #print('Here it is: ', func.__doc__[loc_1+len(text):loc_1+len(text)+50])
            #print(loc_0)

            end_marker = func.__doc__[loc_0:].find(']')
            loc_1 = loc_0 + end_marker
            arg_string = func.__doc__[loc_0 + 3:loc_1]
            #print('arg_string ',arg_string)
            try:
                text = getattr(doc_args, arg_string)
            except AttributeError as e:
                 print(e, func)

            # This 2 is to add the type to the string
            loc_1 = loc_0 + end_marker
            func.__doc__ = func.__doc__[:loc_0 - 2] + ' ' + text + func.__doc__[loc_1+1:]

        return func

    return decor
# endregion
