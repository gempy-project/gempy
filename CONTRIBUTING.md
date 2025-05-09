# Contributing

We welcome and encoruage everyone to contribute to GemPy! 
Contributions can be questions, bug reports, feature requests and new code. 
Here is how to get started.

## Issues 

### Questions

For questions about GemPy (e.g. its applications, functionality, and usage), 
please [search the existing issues for related questions](https://github.com/cgre-aachen/gempy/issues).
If your question has not already been asked, then [make a new issue](https://github.com/cgre-aachen/gempy/issues/new/choose).

### Reporting Bugs

Please report bugs on the [issue page using the bug report template](https://github.com/cgre-aachen/gempy/issues/new?assignees=&labels=&template=bug_report.md&title=) and label the issue as a bug.
The template asks essential questions for you to answer so that we can to understand, reproduce, and fix the bug. 
Be verbose!
Whenever possible, provide tracebacks and/or error messages, screenshots, and sample code or other files.

### Feature Requests

We encourage users to submit ideas for improvements to the GemPy project. For
this please create an issue on the
 [issue page](https://github.com/cgre-aachen/gempy/issues) with the *Feature 
 Request* template and label. Please make sure to use a descriptive title and to
 provide ample background information to help us implement that functionality
 in the future.

## Contributing New Code

Any code contributions are welcome, 
whether fixing a typo or bug,
adding new post-processing/plotting functionality,
improve core functionality,
or anything that you think should be in the repository. 

Contributions should address an open issue (either a bug or a feature request).
If you have found a new bug 
or have an idea or a new feature, 
then please [open the issue](https://github.com/cgre-aachen/gempy/issues/new/choose) 
for discussion and link to that issue in your pull request.

### Python code guidelines
We aim to follow particular Python coding guidelines to improve the sustainability and positive impact of this community project:

- Follow [The Zen of Python](https://www.python.org/dev/peps/pep-0020/), most importantly "readability counts" when writing Python code.
- Adhere to the [Style Guide for Python Code (PEP8)](https://www.python.org/dev/peps/pep-0008/).
- Write thorough and effective documentation: 
Make a docstring for each module, function, class, and method, 
all following [PEP 257](https://www.python.org/dev/peps/pep-0257/) for high-level guidelines
and [Google Python Style Guidelines](http://google.github.io/styleguide/pyguide.html) for Syntax. 
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
- The code should explain the *what* and *how*. Add inline comments to explain the *why*.
If an inline comment seems to be needed, consider first making the code more readable.
For all comments, follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
- Test every line of code. Untested code is dead code.

### Licensing

All contributed code will be licensed under 
[a EUPL-1.2 license](https://github.com/cgre-aachen/gempy/blob/master/LICENSE).
If you did not write the code yourself, 
it is your responsibility to ensure that the existing license is compatible 
and included in the contributed files. 
In general we discourage contributing third party code.

### Testing

Our test suite uses [`pytest`](https://docs.pytest.org/). 
You should be familiar with `pytest` before contributing.
Please run all tests locally before creating a pull request. 
You can do this by running `pytest` via your terminal in your GemPy folder:
```bash
cd ./path/to/gempy
pytest
```
All tests are located in the `test` folder and its subfolders.  
All contributed code must include test code in the pull request.

### Pull Requests

All contributions are made via pull requests (PR's) to the master branch. 
You must complete the checklist in the [PR template](https://github.com/cgre-aachen/gempy/blob/master/.github/PULL_REQUEST_TEMPLATE.md) before we will review the PR and consider merging your contribution.
