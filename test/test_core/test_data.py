import gempy.core.data as gd
import gempy.core.gempy_front as gp
import pandas as pn
import numpy as np
import os
import pytest


@pytest.fixture(scope="module")
def test_read_interfaces():
    interfaces = gp.Interfaces()
    interfaces.read_interfaces(os.pardir + "/input_data/FabLessPoints_Points.csv", inplace=True)

    # Test setting series
    series = gp.Series(series_distribution={"fault": 'MainFault',
                                            "Rest": ('SecondaryReservoir', 'Seal3', 'Reservoir', 'Overlying'),
                                            })
    interfaces.map_series_to_data(series)
    return interfaces


@pytest.fixture
def test_model_init(scope="module"):
    # Create empty model
    model = gp.Model()
    return model

@pytest.fixture
def test_create_series():
    series = gp.create_series(series_distribution={"fault": 'MainFault',
                                                   "Rest": ('SecondaryReservoir', 'Seal3', 'Reservoir', 'Overlying'),
                                                  })
    return series

@pytest.fixture()
def test_create_faults(test_create_series):
    faults = gp.Faults(test_create_series)
    return faults

@pytest.fixture()
def test_create_formations():
    formations = gp.Formations(values_array=np.arange(1, 8).reshape(-1, 1),
                               properties_names=np.array(['density']))
   # formations.set_formation_names(['MainFault', 'SecondaryReservoir','Seal',
   #                                 'Reservoir', 'Overlying'])

    return formations

class TestModel:
    pass


class TestInterfaces:
    def test_map_all_to_data(self, test_create_series, test_read_interfaces, test_create_formations,
                             test_create_faults):
        test_read_interfaces.map_series_to_data(test_create_series)
        test_create_formations.set_formation_names(['MainFault', 'SecondaryReservoir','Seal',
                                                    'Reservoir', 'Overlying'])

        test_read_interfaces.map_formations_to_data(test_create_formations)
        test_read_interfaces.map_formations_to_data(test_create_formations)

        test_read_interfaces.map_faults_to_data(test_create_faults)
        test_read_interfaces.set_annotations()
        print(test_read_interfaces)



class TestGrid:
    def test_set_regular_grid(self):
        # Test creating an empty list
        grid = gd.GridClass()
        print(grid.create_regular_grid_3d([0,2000, 0, 2000, -2000, 0], [50, 50, 50]))

        # Test set regular grid by hand
        grid.set_regular_grid([0,2000, 0, 2000, -2000, 0], [50, 50, 50])

    def test_grid_init(self):
        # Or we can init one of the default grids since the beginning by passing
        # the correspondant attributes
        grid = gp.GridClass('regular_grid', extent=[0, 2000, 0, 2000, -2000, 0],
                            resolution=[50, 50, 50])

    def test_grid_front(self):
        gp.create_grid('regular_grid', extent=[0, 2000, 0, 2000, -2000, 0],
                       resolution=[50, 50, 50])


class TestSeries:

    def test_set_series(self, test_read_interfaces):
        series = gp.Series()
        # We can pass a pandas df
        series.set_series_categories(pn.DataFrame({"fault": ['test2'],
                                        "Rest": 'SecondaryReservoir'}))

        # We can even pass an interface object since sometimes (GeoModeller) we
        # can import the surface in the same table
        series.set_series_categories(test_read_interfaces)
        print(series)

        # Test init series
        series = gp.Series(series_distribution={"fault": 'MainFault',
                                                "Rest": ('SecondaryReservoir', 'Seal3', 'Reservoir', 'Overlying'),
                                                })
        return series

    @pytest.fixture
    def test_series_front(self, test_model_init):
        model = test_model_init

        # Assigning series to surface as well as their order (timewise)
        gp.set_series(model, {"Fault_Series": 'Main_Fault',
                                 "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                  'Shale', 'Sandstone_1')},
                      order_series=["Fault_Series", 'Strat_Series'],
                      order_formations=['Main_Fault',
                                        'Sandstone_2', 'Siltstone',
                                        'Shale', 'Sandstone_1', 'basement'
                                        ], verbose=0)

        print(model.series)
        return model

    def test_sequential_pile(self, test_series_front):
        gp.get_sequential_pile(test_series_front)


class TestFaults:
    def test_set_faults(self, test_create_series):
        faults = gp.Faults(test_create_series)
        faults.set_is_fault(['Rest'])
        print(faults)

    def test_default_faults(self, test_create_series):
        faults = gp.Faults(test_create_series)
        print(faults)

    def test_set_fault_relations(self, test_create_faults):
        test_create_faults.set_fault_relation(np.array([[0, 1],
                                                        [0, 0]]))

        print(test_create_faults.faults_relations)


class TestFormations:
    def test_map_formations_from_series(self, test_create_formations, test_create_series):
        test_create_formations.map_formations_from_series(test_create_series)
        print(test_create_formations)

    def test_set_formation_names(self, test_create_formations):
        test_create_formations.set_formation_names(['MainFault', 'SecondaryReservoir','Seal',
                                'Reservoir', 'Overlying'])

    def test_add_formation(self, test_create_formations):
        test_create_formations.add_basement()
        print(test_create_formations)
        # Test that break
        try:
            test_create_formations.add_basement('basement2')
        except AssertionError:
            print('assertion captured')

    def test_set_formations_values(self, test_create_formations):
        test_create_formations.set_formations_values(np.random.rand(2,2))
        test_create_formations.set_formations_values(np.random.rand(5,2))
        test_create_formations.set_formations_values(np.random.rand(10,3))