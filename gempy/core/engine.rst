Design document for GemPy Engine 3
==================================

Early thoughts about how to redesign the interpolator classes. Some of the
main specifications are:

- ?Independent interpolator from the rest so we can use it as each own **microservice** via
  some sort of **sync** function that set up all the constant values

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

- Having several sets of coordinates systems to choose from when we call compute for **anisotropies**

Graph Logic
~~~~~~~~~~~

Tensor super class
^^^^^^^^^^^^^^^^^^

aesara, numpy and tensorflow have a very similar syntax but not 100% exact.
We could create a tensor wrapper that makes sure that it returns the right value.
For example sin is the same in all 3:

.. code-block:: python

    class GemPyTensor:
        def __init__(package='numpy')
            if package=='numpy':
                _tt = np
            elif package=='aesara':
                _tt = aesara.tensor

        # Same numpy-aesara easy
        def sin(x):
            return _tt.sin(x)

        # Different
        def set_subtensor(x: slice, new_slice)
            if package=='numpy':
                x = new_slice
                return x
            if package == 'aesara':
                return _tt.set_subtensor(x , new_slice)

Advantages
----------

- Maintainability: Not having to write 3 different graphs
- If we use aesara syntax as base for `GemPyTensor` we do not need to rewrite even
  the current graph
- If a function exist in one package, we can mock it on the others using the correspondent wrappers.

Disadvantages
-------------

- How do we deal with control flow on run time that is done vastly different


Data Flow
~~~~~~~~~


Scalable Output
^^^^^^^^^^^^^^^
In GemPy 2.1 we moved from having different interpolator objects with different graphs
to having always one interpolator capable to select how much of the whole graph is going to be
constructed at compile time.

As I see it there some possible paths here:

.. note:: GemPy-server design allows to have several engines up so we could have pretty much anything

1. We construct always the full graph. This will enable to choose in runtime what do we want to compute
   e.g. some gravity, without having to recompile. The cost is memory and as the graph gains in complexity with
   features that may not be used too often it will be a waste.

2. Caching, every time

...


Independent Interpolator
^^^^^^^^^^^^^^^^^^^^^^^^

To obtain this we would need to split the interpolator in two classes.

1) One buffer class containing all the data
2) A graph class

A sync function would feed the graph class

*Which consequences has this for the* ``Model`` *methods?*

The graph class could be in a different process or could be in the same process.






