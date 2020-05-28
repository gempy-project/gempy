Design document for GemPy Engine 3
==================================

Early thoughts about how to redesign the interpolator classes. Some of the
main specifications are:

- Compatibility with **TensorFlow** and **Numpy**
- Outputs of the graph scalable
- Independence of each scalar field
    - **No global constants**, e.g. each field will have its own range
    - **Scalar field injection**, any recursive value of one field into
      the next one can have an access point.
- **Octtress** are Native. To deactive it, we just need to set the number of levels to 1
- Multiple solver (such as **sparse solver** easy to use)
    + cuda solver
    + gradient solver
    + sparse solver

- Interpolator data stored in df and always updated.