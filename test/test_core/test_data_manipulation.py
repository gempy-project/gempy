

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp


# Importing auxiliary libraries
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import pytest

# ## Series
#
# Series is the object that contains the properties associated with each independent scalar field. Right now it is
# simply the order of the series (which is infered by the index order). But in the future will be add the unconformity
# relation or perhaps the type of interpolator
#
# Series and Faults classes are quite entagled since fauls are a type of series


@pytest.fixture(scope='module')
def create_faults():
    faults = gp.Faults()
    return faults


@pytest.fixture(scope='module')
def create_series(create_faults):
    faults = create_faults

    series = gp.Series(faults)
    series.set_series_index(['foo', 'foo2', 'foo5', 'foo7'])
    series.add_series('foo3')
    series.delete_series('foo2')
    series.rename_series({'foo': 'boo'})
    series.reorder_series(['foo3', 'boo', 'foo7', 'foo5'])

    faults.set_is_fault(['boo'])

    fr = np.zeros((4, 4))
    fr[2, 2] = True
    faults.set_fault_relation(fr)

    series.add_series('foo20')

    return series


@pytest.fixture(scope='module')
def create_formations(create_series):
    series = create_series
    formations = gp.Surfaces(series)
    formations.set_formation_names(['foo', 'foo2', 'foo5'])

    print(series)

    # We can add new formations:
    formations.add_formation(['feeeee'])
    print(formations)

    # The column formation is also a pandas.Categories.
    # This will be important for the Data clases (Interfaces and Orientations)

    print(formations.df['formation'])

    ### Set values

    # To set the values we do it with the following method
    formations.set_formation_values_pro([2, 2, 2, 5])

    print(formations)

    # #### Set values with a given name:

    # We can give specific names to the properties (i.e. density)
    formations.add_formation_values_pro([[2, 2, 2, 6], [2, 2, 1, 8]], ['val_foo', 'val2_foo'])
    print(formations)

    ### Delete formations values
    #
    # To delete a full propery:
    formations.delete_formation_values(['val_foo', 'value_0'])

    # #### One of the formations must be set be the basement:

    formations.set_basement()
    print(formations)

    # #### Set formation values
    #
    # We can also use set values instead adding. This will delete the previous properties and add the new one

    formations.set_formation_values_pro([[2, 2, 2, 6], [2, 2, 1, 8]], ['val_foo', 'val2_foo'])
    print(formations)

    # The last property is the correspondant series that each formation belong to. `series` and `formation`
    # are pandas categories. To get a overview of what this mean
    # check https://pandas.pydata.org/pandas-docs/stable/categorical.html.

    print(formations.df['series'])

    print(formations.df['formation'])

    # ### Map series to formation

    # To map a series to a formation we can do it by passing a dict:
    # If a series does not exist in the `Series` object, we rise a warning and we set those formations to nans

    d = {"foo7": 'foo', "booX": ('foo2', 'foo5', 'fee')}

    formations.map_series(d)
    formations.map_series({"foo7": 'foo', "boo": ('foo2', 'foo5', 'fee')})

    print(formations)

    # An advantage of categories is that they are order so no we can tidy the df by series and formation

    formations.df.sort_values(by='series', inplace=True)

    # If we change the basement:

    formations.set_basement('foo5')

    # Only one formation can be the basement:

    print(formations)

    # ### Modify formation name

    formations.rename_formations({'foo2': 'lala'})

    print(formations)

    formations.df.loc[2, 'val_foo'] = 22

    print(formations)

    formations.update_sequential_pile()

    formations.sequential_pile.figure
    # We can use `set_is_fault` to choose which of our series are faults:
    return formations


@pytest.fixture(scope='module')
def create_interfaces(create_formations, create_series):
    # # Data
    # #### Interfaces
    # These two DataFrames (df from now on) will contain the individual information of each point at an interface or
    # orientation. Some properties of this table are mapped from the *df* below.
    formations = create_formations
    interfaces = gp.Interfaces(formations)

    print(interfaces)

    interfaces.set_interfaces(pn.DataFrame(np.random.rand(6, 3)), ['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])

    print(interfaces)

    interfaces.map_data_from_formations(formations, 'series')
    print(interfaces)


    interfaces.map_data_from_formations(formations, 'id')
    print(interfaces)


    interfaces.map_data_from_series(create_series, 'order_series')
    print(interfaces)

    # In[59]:

    interfaces.sort_table()
    print(interfaces)
    return interfaces


@pytest.fixture(scope='module')
def create_orientations(create_formations, create_series):
    formations = create_formations

    # ### Orientations
    orientations = gp.Orientations(formations)

    print(orientations)

    # ### Set values passing pole vectors:

    orientations.set_orientations(np.random.rand(6, 3) * 10,
                                  np.random.rand(6, 3),
                                  surface=['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])

    print(orientations)

    # ### Set values pasing orientation data: azimuth, dip, pole (dip direction)

    orientations.set_orientations(np.random.rand(6, 3) * 10,
                                  orientation=np.random.rand(6, 3) * 20,
                                  surface=['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])

    print(orientations)

    # ### Mapping data from the other df
    orientations.map_data_from_formations(formations, 'series')
    print(orientations)

    orientations.map_data_from_formations(formations, 'id')
    print(orientations)

    orientations.map_data_from_series(create_series, 'order_series')
    print(orientations)

    orientations.set_annotations()
    return orientations


@pytest.fixture(scope='module')
def create_grid():
    # Test creating an empty list
    grid = gp.Grid()
    # Test set regular grid by hand
    grid.set_regular_grid([0, 2000, 0, 2000, -2000, 0], [50, 50, 50])
    return grid


@pytest.fixture('module')
def create_rescaling(create_interfaces, create_orientations, create_grid):
    rescaling = gp.RescaledData(create_interfaces, create_orientations, create_grid)
    return rescaling


@pytest.fixture('module')
def create_additional_data(create_interfaces, create_orientations, create_grid, create_faults,
                           create_formations, create_rescaling):

    ad = gp.AdditionalData(create_interfaces, create_orientations, create_grid, create_faults,
                           create_formations, create_rescaling)
    return ad


class TestDataManipulation:

    def test_series(self, create_series):
        return create_series

    def test_formations(self, create_formations):
        return create_formations

    def test_interfaces(self, create_interfaces):
        return create_interfaces

    def test_orientations(self, create_orientations):
        return create_orientations

    def test_rescaling(self, create_rescaling):
        return create_rescaling

    def test_additional_data(self, create_additional_data):
        return create_additional_data


class TestGrid:
    def test_set_regular_grid(self):
        # Test creating an empty list
        grid = gp.Grid()
        print(grid.create_regular_grid_3d([0,2000, 0, 2000, -2000, 0], [50, 50, 50]))

        # Test set regular grid by hand
        grid.set_regular_grid([0,2000, 0, 2000, -2000, 0], [50, 50, 50])

    def test_grid_init(self):
        # Or we can init one of the default grids since the beginning by passing
        # the correspondant attributes
        grid = gp.Grid('regular_grid', extent=[0, 2000, 0, 2000, -2000, 0],
                       resolution=[50, 50, 50])


