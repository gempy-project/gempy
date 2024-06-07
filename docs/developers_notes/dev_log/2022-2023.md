## Refactoring for v3 notes:

- I want to make proper subpackages of the library
- Once the subpackages are done we can start moving them out to different repos
- First proposition:
  - integration tests
  - notebooks
  - scripts
  - examples
  - GemPy
    - Plugins:  
      - Addons
      - Assets
      - Bayesian
    - API
      - API Modules
      - gempy_api
    - Plotting
    - Core
      - Data
      - Interpolation
        - Theano 
        - GemPy Engine

> I think Pluggins and addons have to be split depending on how couple are they to the interpolator?


## Refactoring notes:

Doing:
  - [-] Refactoring interpolation (Aug-Sep 2023)

- TODO:
  - [ ] Saving and loading models (Sep-Oct 2023)
  - [ ] Refactoring Geophysics (Oct-Nov 2023)
  - [ ] Refactoring topology
  - [ ] Orientations from surface points - Move to gempy_plugins