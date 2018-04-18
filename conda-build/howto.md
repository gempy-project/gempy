# build conda package from pypi package

https://conda.io/docs/user-guide/tutorials/build-pkgs-skeleton.html

## Step 1: Build the skeleton files from pypi package

In **any** directory run `conda skeleton pypi gempy` to create a package called `gempy` with a `meta.yml` file inside. The files `bld.bat` and `build.sh` are provided.

Make sure all those files are located inside the package:

1. `meta.yaml` — Contains all the metadata in the recipe. Only the package name and package version sections are required—everything else is optional.
2. `bld.bat` — Windows commands to build the package.
3. `build.sh` — macOS and Linux commands to build the package.

## Step 2: Building the Conda package

Run `conda-build gempy`

## Step 3: Uploading the package to Anaconda.org

1. Create a free Anaconda.org account and record your new Anaconda.org username and password.
2. Run `conda install anaconda-client` and enter your Anaconda.org username and password.
3. Log into your Anaconda.org account from your Terminal or an Anaconda Prompt with the command `anaconda login`.

After this you can upload the local package using `anaconda upload absolute\path\to\<gempy-X.XX.X-pyXX_X>.tar.bz2` 