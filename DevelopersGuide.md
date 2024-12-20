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

PyPi release:
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

## Documentation update

The documentation is created using the [Sphinx documentation framework](https://www.sphinx-doc.org/en/master/) and the [Sphinx Gallery extension](https://sphinx-gallery.github.io/stable/index.html), which builds the documentation on basis of the repository code and the added examples. The documentation includes the tutorials and example models, which are part of the main repository, as well as additional external files. 

Also, opposed to the standard approach with GitHub Actions, the documentation is created offline, to avoid an unnecessary bloating of the repository and to reduce downtimes during update.

Creating an update of the documentation can therefore be quite involved (depending on the level of changes). Please follow the steps below carefully.

### General case: using a fork

#### Create fork and perform changes

- If you don't already have one, create a fork of the main GemPy repository to your own GitHub account. Make sure to tick the box to fork _all_ branches (not only the _main_ branch).
- Clone the repository to your local computer.
- Create a new branch of the `main` repository for the subsequent pull request.
- Perform the changes, e.g.:
    - Update text in the documentation
    - Update examples (and/ or add new ones)
    - Perform changes in code and/ or function docstrings
    - etc.
- If you have performed changes before, then you can also copy these changes into the branch (Note: if you have changes in a separate branch, then they can also be copied with `git restore` or `git checkout` - please see git documentation for details).

#### Update documentation using Sphinx

- Update the documentation using sphinx in the branch where the changes were made (`main` branch or new branch based on `main`):
    - Check that you have `sphinx` and `sphinx-gallery` installed;
    - Change to folder with documentation from main repo with `cd docs`;
    - Run update with `make html`;
    - Wait... (can take some hours when run for the first time)
- The updated documentation is now in the subfolder `build/html`. Preview the generated documentation locally: open `build/html/index.html` in a web browser.

#### Add updated documentation to branch `gh-pages`


- If everything is fine, first commit all changes that were made in the main repo:
    - `git add .`
    - `git commit -m "update message"`
- Copy updated documentation to a temporary directory,for example `/tmp/docs-html` with:
    - `mkdir /tmp/docs-html`
    - `cp -R docs/build/html/* /tmp/docs-html/`
- checkout the documentation branch: `git checkout gh-pages`
- Copy the updated files into the main folder of the `gh-pages` branch: `cp -R /tmp/docs-html/* .`
- To be sure: check updated documentation, open `index.html`, now in the main folder of the repository.
- Commit all changes:
    - `git add .`
    - `git commit -m "Updated documentation in gh-pages branch`

#### Contribute updates back to main repository

- Push _both_ branches `main` and `gh-pages` back to your own remote GitHub profile
- Create a pull request for both branches (make sure to create the pull request for the `gh-pages` branch also to the `gh-pages` branch of the original gempy repository)
- Please include meaningful descriptions about your changes (both, in the main repo, as well as in the documnetation).



### Special case: documentation update for maintainers with repository write rights

- `git worktree add ../gempy_docs gh-pages` This will create a new folder called `gempy_docs` in the parent directory of the current repo. This folder will contain the `gh-pages` branch of the repo. This is where the docs will be pushed to.
- `cp -r --force ./docs/build/html/* ../gempy_docs/` This will copy the contents of the `docs/build/html` folder to the `VisualBayesicDocs` folder.
- `cd ../gempy_docs` This will change the current directory to the `gempy_docs` folder.
- `git add .` This will add all the files in the current directory to the staging area.
- `git commit -m "Update docs"` This will commit the changes to the `gh-pages` branch.
- `git push origin gh-pages` This will push the changes to the `gh-pages` branch of the repo.
- `cd ../gempy` This will change the current directory back to the `gempy` folder.
