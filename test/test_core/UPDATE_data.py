import gempy as gp
import pandas as pn
import numpy as np
import os
import pytest


@pytest.fixture(scope="module")
def test_read_surface_points():
    surface_points = gp.SurfacePoints()
    surface_points.read_surface_points(os.pardir + "/input_data/FabLessPoints_Points.csv", inplace=True)

    # Test setting series
    series = gp.Series(series_distribution={"fault": 'MainFault',
                                            "Rest": ('SecondaryReservoir', 'Seal3', 'Reservoir', 'Overlying'),
                                            })
    surface_points.map_data_from_series(series, 'id')
    return surface_points


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
    formations = gp.Surfaces(values_array=np.arange(1, 8).reshape(-1, 1),
                             properties_names=np.array(['density']))
   # formations.set_surfaces_names(['MainFault', 'SecondaryReservoir','Seal',
   #                                 'Reservoir', 'Overlying'])

    return formations

@pytest.fixture(scope='module')
def test_load_model():
    model = gp.load_model(os.pardir + "/input_data/test_solution.pickle")
    return model


class TestModel:
    def test_modify_inteterfaces(self, load_model):
        load_model.modify_surface_points([0, 10, 5], X=[90, 20, 30], Y=[30, 20, 40])
        load_model.modify_surface_points(0, X=[60])

        load_model.modify_surface_points([0, 1, 4], X=[60])


class Testsurface_points:
    def test_map_all_to_data(self, test_create_series, test_read_surface_points, test_create_formations,
                             test_create_faults):
        test_read_surface_points.map_data_from_series(test_create_series)
        test_create_formations.set_surfaces_names(['MainFault', 'SecondaryReservoir', 'Seal',
                                                    'Reservoir', 'Overlying'])

        test_read_surface_points.map_formations_to_data(test_create_formations)
        test_read_surface_points.map_formations_to_data(test_create_formations)

        test_read_surface_points.map_data_from_faults(test_create_faults)
        test_read_surface_points.update_annotations()
        print(test_read_surface_points)

    def test_map_formations_to_data(self, test_read_surface_points, test_create_formations):
        test_read_surface_points.map_formations_to_data(test_create_formations)


class TestOrientations:
    @pytest.fixture(scope='class')
    def test_read_orientations(self):
        orientations = gp.Orientations()
        orientations.read_orientations(os.pardir + "/input_data/FabLessPoints_Foliations.csv", inplace=True)
        return orientations

    def test_map_all_to_data(self, test_create_series, test_read_orientations, test_create_formations,
                             test_create_faults):
        test_read_orientations.map_data_from_series(test_create_series)
        test_create_formations.set_surfaces_names(['MainFault', 'SecondaryReservoir', 'Seal',
                                                    'Reservoir', 'Overlying'])

        test_read_orientations.map_formations_to_data(test_create_formations)
        test_read_orientations.map_formations_to_data(test_create_formations)

        test_read_orientations.map_data_from_faults(test_create_faults)
        test_read_orientations.update_annotations()
        print(test_read_orientations)


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

    def test_grid_front(self):
        gp.create_grid('regular_grid', extent=[0, 2000, 0, 2000, -2000, 0],
                       resolution=[50, 50, 50])


class TestSeries:

    def test_set_series(self, test_read_surface_points):
        series = gp.Series()
        # We can pass a pandas categories_df
        series.set_series_categories(pn.DataFrame({"fault": ['test2'],
                                        "Rest": 'SecondaryReservoir'}))

        # We can even pass an interface object since sometimes (GeoModeller) we
        # can import the surface in the same table
        series.set_series_categories(test_read_surface_points)
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

        print(test_create_faults.faults_relations_df)


class TestFormations:
    def test_create_formation(self, test_create_formations):
        print(test_create_formations)

    def test_map_formations_from_series(self, test_create_formations, test_create_series):
        test_create_formations.map_formations_from_series(test_create_series)
        print(test_create_formations)

    def test_map_formations_from_series2(self, test_create_series):
        formations = gp.Surfaces()
        formations.map_formations_from_series(test_create_series)
        print(formations)

    def test_set_formation_names(self, test_create_formations):
        test_create_formations.set_surfaces_names(['MainFault', 'SecondaryReservoir', 'Seal',
                                'Reservoir', 'Overlying'])

        print(test_create_formations)
        formations = gp.Surfaces(values_array=np.arange(1, 8).reshape(-1, 1),
                                 properties_names=np.array(['density']))

        formations.set_surfaces_names(['MainFault', 'SecondaryReservoir', 'Seal',
                                'Reservoir', 'Overlying'])
        print(formations)

        formations = gp.Surfaces(values_array=np.arange(1, 2).reshape(-1, 1),
                                 properties_names=np.array(['density']))

        formations.set_surfaces_names(['MainFault', 'SecondaryReservoir', 'Seal',
                                        'Reservoir', 'Overlying'])

        print(formations)
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


class TestSolution:
    def test_representation(self, load_model):
        sol = load_model.solutions
        sol.set_values(np.random.rand(4, 2, 3), compute_mesh=False)
        print(sol)

    def test_get_surfaces(self, load_model):
        model = load_model
        print(model.solutions)
        print(model.solutions.compute_all_surfaces())
        print(gp.get_surfaces(model))


def test_export_vtk(load_model):
    model = load_model
    gp.plot.export_to_vtk(model, os.path.dirname(__file__)+'/vtk/expert_test')