- Do I need to be able to convert legacy data classes to new ones?

Legacy "DataTypes":
 - MetaData
 - Grid
 - Colors
- Pile
 - Surfaces
 - Structure
- Config
 - Options
 - Kriging parameters
 - Additional Data
- Geometric Data
  - GeometricData (base)
  - SurfacePoints
  - Orientations
  - ScalingParam


GemPy Engine consumes:
  - InterpolationInput
  - Options
  - Structure

GemPy Engine produces:
  - Solutions

---

We have the geo_model object that has all the data.

- We need a function to from geo_model to the 3 objects that the engine consumes.
  - This substitutes `gp.set_interpolator`, `set_aesara_graph`
  - The more functional this is the better! **Compress!**
- We compute with gempy engine
- We need a function to from the engine output to the geo_model object.

---

After this I should be able to run the notebooks and tests.
