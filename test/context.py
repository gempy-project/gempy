"""
The main function of the code you provided is to add the parent directory of the tests folder to the Python path (sys.path). 
his is done with sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))). 
his line is adding the directory one level up from the directory of the context.py file to the front of the sys.path.

sys.path is the list of directories that Python checks when it's trying to import a module. By inserting the parent
directory of the tests directory at the front of sys.path, Python will check this directory first when importing modules.
This allows test scripts in the tests directory to import modules from the parent directory (like gempy in your case) as
if they were in the same directory. This is very helpful when you're running tests because it ensures the tests are
running against the source code in your project, rather than an installed version of your package.
"""
# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
