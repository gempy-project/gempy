Design document for GemPy Engine 3
==================================

Early thoughts about how to redesign the interpolator classes. Some of
the main specifications are:

-   ?Independent interpolator from the rest so we can use it as each own
    **microservice** via some sort of **sync** function that set up all
    the constant values
-   Compatibility with **TensorFlow** and **Numpy**
-   Outputs of the graph scalable
-   Independence of each scalar field
    :   -   **No global constants**, e.g. each field will have its own
            range
        -   **Scalar field injection**, any recursive value of one field
            into the next one can have an access point.
        -   Each scalar field interpolated with anything (as Tensor
            constant)

-   **Octtress** are Native. To deactive it, we just need to set the
    number of levels to 1
-   **Hierarchical modelling**/**Techtonic domains**

-   Multiple solver (such as **sparse solver** easy to use)
    :   -   cuda solver
        -   gradient solver
        -   sparse solver

-   Interpolator data stored in df and always updated.
-   Having several sets of coordinates systems to choose from when we
    call compute for **anisotropies**
-   Finite faults


## Graph Logic

``` python
class GemPyTensor:
    def __init__(package='numpy')
        if package=='numpy':
            _tt = np
        elif package=='theano':
            _tt = theano.tensor

    # Same numpy-theano easy
    def sin(x):
        return _tt.sin(x)

    # Different
    def set_subtensor(x: slice, new_slice)
        if package=='numpy':
            x = new_slice
            return x
        if package == 'theano':
            return _tt.set_subtensor(x , new_slice)
```



### Tensor super class

Theano, numpy and tensorflow have a very similar syntax but not 100%
exact. We could create a tensor wrapper that makes sure that it returns
the right value. For example sin is the same in all 3:


-   Keeping in mind tensor flow hessian

#### Advantages

-   Maintainability: Not having to write 3 different graphs
-   If we use theano syntax as base for GemPyTensor we do not need to
    rewrite even the current graph
-   If a function exist in one package, we can mock it on the others
    using the correspondent wrappers.

#### Disadvantages

-   How do we deal with control flow on run time that is done vastly
    different

### Octrees

#### Grid

-   We would add a new grid defined by:
    :   -   Initial regular gird
        -   Number of levels

-   The **size of the output** would be **undefined**. This makes the
    **result uncacheable**
-   Should be the **default grid**
-   **Topology** comes from free

Data Flow
---------

### Cache series

-   Each series is independent and therefore only changes on a given
    series needs to be recalculated. To accomplish this we need to be
    able **inject** (*alpha* in current implementation) as constant:
    -   weights
    -   scalar fields
    -   lith blocks
    -   mask arrays
-   **Changing geometric data**
    :   -   Masking logic for older events

-   **Changing grid**
    :   -   Everything change **except weights**
        -   This has heavy consequences for Octrees\_
        -   Also limits the sandbox

-   **Changing order pile**
    :   -   Masking logic for older events

-   **Adding fault**
    :   -   Everything younger changes

#### Masking logic

-   Changes on an older (lower in the pile) series would not affect
    younger (higher in the) series

### Hierarchical modelling

-   **Series/Geo. Features** would have levels
-   We can share the logic of Cache series\_

### Scalable Output

In GemPy 2.1 we moved from having different interpolator objects with
different graphs to having always one interpolator capable to select how
much of the whole graph is going to be constructed at compile time.

As I see it there some possible paths here:

> **note**
>
> GemPy-server design allows to have several engines up so we could have
> pretty much anything

1.  We construct always the full graph. This will enable to choose in
    runtime what do we want to compute e.g. some gravity, without having
    to recompile. The cost is memory and as the graph gains in
    complexity with features that may not be used too often it will be a
    waste.
2.  Caching every time, i.e. every time we compile we create a pickle of
    the interpolator. This option would require
    Independent Interpolator\_ inplace

...

### Independent Interpolator

To obtain this we would need to split the interpolator in two classes.

1)  One buffer class containing all the data
2)  A graph class

A sync function would feed the graph class

*Which consequences has this for the* `Model` *methods?*

The graph class could be in a different process or could be in the same
process.


Database
--------
## Better saving/loading models


## Storing models in Microsoft cloud

## Pulling data
- Use 


## Version Control

### Tools

https://dvc.org/features


### Options:

- Using one coordinate of xarray?
    - Can we just do an xarray per pandas.Series and use xarray interpolate?
        In this case the coordinate would be only **state**?

    - The main disadvantage is a lot of nans eventually


### Most promissing 
- In memory xarray
- on disk git csv?