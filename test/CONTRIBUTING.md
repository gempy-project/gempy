# Contributing

We welcome and encoruage everyone to contribute to GemPy! 
Contributions can be questions, bug reports, feature requests and new code. 
Here is how to get started.

## Cloning the Source Repository

To get the latest release candidate of GemPy you can clone the master branch
from GitHub using Git:

```bash
git clone https://github.com/cgre-aachen/gempy.git
``` 
## Issues 

### Questions

For questions about GemPy, its applications and functionality and usage, please
create an issue in the GitHub repository 
[cgre-aachen/gempy](https://github.com/cgre-aachen/gempy). This way we can 
document any questions you may have, so other people with similar questions 
will be able to find the answers in the future.

### Reporting Bugs

If you happen to stumble upon one of the incredibly rare and utterly unlikely
bugs, crashes or concerning behaviour while using GemPy, please report it on 
the [issue page](https://github.com/cgre-aachen/gempy/issues) using the bug 
report template and appropriate labeling. The remplate will provide you with 
some default questions to answer, which will provide us with essential 
information to understand and solve your issue. When reporting a bug, please be 
overly descriptive so that we will be able to to reproduce and resolve the issue. 
Whenever possible, please provide tracebacks / error messages, screenshots and
sample code / files to help us address your issue.

### Feature Requests

We encourage users to submit ideas for improvements to the GemPy project. For
this please create an issue on the
 [issue page](https://github.com/cgre-aachen/gempy/issues) with the *Feature 
 Request* template and label. Please make sure to use a descriptive title and to
 provide ample background information to help us implement that functionality
 in the future.

## Contributing New Code

If you want to contribute code to the GemPy open-source project please start by
opening up an issue on the [issue page](https://github.com/cgre-aachen/gempy/issues)
so we can start discussions about implementation and provide you with any help 
you might need. 

**Once you are ready to start coding, please take a look at the following 
contribution guidelines.**

Any code contributions are welcome: if you want to fix a couple of typos, add
custom geomodel post-processing functionality or a new plotting function - your
efforts are welcome! 

We adhere to three general coding paradigms to ensure GemPy to grow as a valuable
and reliable community project:

1. **Write intuitive code.** Python allows for readable code - and we think that
any good code should be readable and self-explanatory. From adhering to code
formatting guidelines to intuitive naming conventions, applying good standards
will ensure the usability and maintainability of the code for users and 
developers alike.
2. **Document everything.** Functions, methods and classes need to have 
descriptive and helpful docstrings. Describe the *why* of your code. The *how*
should be left to intuitive code. Consider including inline comments in your 
code if you think it will make the *how* more intuitive. We also recommend 
providing a simple use case descriptiong within the docstring of a new feature.
3. **Untested code is broken code.** We aim to increase our test coverage and
keep it as high as possible and sensible. Thus, any new code contributed to 
GemPy needs to be tested.

### Licensing

All contributed code will be licensed under the LGPL-3 license found in the
[license file](https://github.com/cgre-aachen/gempy/blob/master/LICENSE) within
the GemPy repository.

If you did not write the code yourself, it is your responsibility to
ensure that the existing license is compatible and included in the contributed
files. In general we would like to discourage contributing third party code to
our project.

### Code Formatting

We reasonably adhere to the offical
 [Style Guide for Python Code (PEP8)](https://www.python.org/dev/peps/pep-0008/)
, which you should already be using in all of your code. Please make sure to 
format your code in a reasonable manner to make reviewing and using your code
as easy and intuitive as possible.

### Documentation

Any contribution to GemPy needs to be appropriately documented using docstrings.
This includes docstrings at the top of every module / Python file you contribute.
We adhere to the docstring format provided by the 
[Google Python Style Guidelines](https://github.com/cgre-aachen/gempy/issues) 
(Section 3.8) and we recommend you read them before starting development.

**Example function documentation:**
```python
def func(arg1: int, arg2: float) -> int:
    """A concise one line summary of the function.
    
    Additional information and description of the function, if necessary. This
    can be as long and verbose as you think is necessary for other users and 
    developers to understand your functionality.
    
    Args:
        arg1 (int): Description of the first argument.
        arg2 (float): Description of the second argument. Please use hanging 
            indentation for multi-line argument descriptions.
    
    Returns:
        (int) Description of the return value(s)
    """
    return 42
```

### Testing

To ensure that the GemPy release candidate (master branch) is working properly, 
we employ the code testing suite [pytest](https://docs.pytest.org/). If you are
unfamiliar with *pytest*, you should have a look at their documentation and get
familiar with using it before you contribute to GemPy.

If you contribute code that changes existing code (e.g. bug fixes), then please
run all tests locally before creating a pull request. You can do this by running
*pytest* via your terminal in your GemPy folder:

```bash
cd ./path/to/gempy
pytest
```

If you contribute new functionality to GemPy, we require you to cover your code
with tests, so that we can ensure it working properly.

All tests are located in the `test` folder of the GemPy repository. This is where
you need to add your own tests as a subfolder. 

### Pull Request Checklist

The following checklist *needs* to be completed before we will accept any pull
request to the GemPy codebase. 

1. Run all tests.
2. Check documentation
3. Check if your code adheres to PEP8

 
