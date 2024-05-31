Before a release.
----------------
# set version number in setup.py, also in the config file of the documentation and init of the package
- [ ] setup.py
- [ ] gempy.__init__
> Note: in the config for sphinx~ this is taken from gempy.__init__

Requirements version
--------------------
- requirements.txt, optional_requirements.txt, dev_requirements.txt should either set minimum version and reject versions

Github release
--------------
    # add new tag
    $ git tag X.X -m "Add X.X tag for PyPI"
    # push git tag
    $ git push --tags origin master

PyPi release
------------
New 
```
    #  First create the dist
    python -m build

    # Second upload the distributions
    twine upload dist/*
```

### Type of commits:

- ENH: Enhancement, new functionality
- BUG: Bug fix
- DOC: Additions/updates to documentation
- TST: Additions/updates to tests
- BLD: Updates to the build process/scripts
- PERF: Performance improvement
- CLN: Code cleanup

### To push the docs in a fancy way

- `git worktree add ../gempy_docs gh-pages` This will create a new folder called `gempy_docs` in the parent directory of the current repo. This folder will contain the `gh-pages` branch of the repo. This is where the docs will be pushed to.
- `cp -r --force ./docs/build/html/* ../gempy_docs/` This will copy the contents of the `docs/build/html` folder to the `VisualBayesicDocs` folder.
- `cd ../gempy_docs` This will change the current directory to the `gempy_docs` folder.
- `git add .` This will add all the files in the current directory to the staging area.
- `git commit -m "Update docs"` This will commit the changes to the `gh-pages` branch.
- `git push origin gh-pages` This will push the changes to the `gh-pages` branch of the repo.
- `cd ../gempy` This will change the current directory back to the `gempy` folder.
