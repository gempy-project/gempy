import numpy as np


class Model(object):
    def __init__(self):
        self.meta = None
        self.grid = None
        self.faults = None
        self.series = None
        self.formations = None
        self.interfaces = None
        self.orientations = None
        self.structure = None
        self.model = None

    def save_data(self):
        pass

    def get_data(self):
        pass

    def get_theano_input(self):
        pass

class MetaData(object):
    def __init__(self, name_project='default_project'):
        self.name_project = name_project
        self.date = None


class GridClass(object):
    """
    Class to generate grids to pass later on to a InputData class.
    """

    def __init__(self):

        self.resolution = None
        self.extent = None
        self.values = None

    def set_custom_grid(self, custom_grid):
        """
        Give the coordinates of an external generated grid

        Args:
            custom_grid (numpy.ndarray like): XYZ (in columns) of the desired coordinates

        Returns:
              numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
        """
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] is 3, 'The shape of new grid must be (n,3)' \
                                                                              ' where n is the number of points of ' \
                                                                              'the grid'
        self.values = custom_grid
        return self.values

    @staticmethod
    def create_regular_grid_3d(extent, resolution):
        """
        Method to create a 3D regular grid where is interpolated

        Args:
            extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz].

        Returns:
            numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
        """


        dx, dy, dz = (extent[1] - extent[0]) / resolution[0], (extent[3] - extent[2]) / resolution[0],\
                                    (extent[5] - extent[4]) / resolution[0]

        g = np.meshgrid(
            np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0], dtype="float32"),
            np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1], dtype="float32"),
            np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2], dtype="float32"), indexing="ij"
        )

        values = np.vstack(map(np.ravel, g)).T.astype("float32")
        return values

    def set_regular_grid(self, extent, resolution):
        self.extent = extent
        self.resolution = resolution
        self.values = self.create_regular_grid_3d(extent, resolution)


class Series(object):
    def __init__(self):
        self.series = None

    def set_series(self):
        pass


class Faults(object):
    def __init__(self):
        self.faults_relations = None
        self.faults = None
        self.n_faults = None

    def set_faults(self):
        pass

    def check_fault_relations(self):
        pass

class Formations(object):
    def __init__(self):
        self.formations = None
        self.sequential_pile = None

    def set_formations(self):
        pass

class Data(object):
    def __init__(self):
        pass

    def import_data(self):
        pass

    def order_table(self):
        pass

    def set_annotations(self):
        pass

    def rescale_data(self):
        pass

class Interfaces(object, Data):
    def __init__(self):
        self.df = None

    def set_basement(self):
        pass

    def count_faults(self):
        pass

    def set_default_interface(self):
        pass


class Orientations(object, Data):
    def __init__(self):
        self.df = None

    def calculate_gradient(self):
        pass

    def calculate_orientations(self):
        pass

    def create_orientation_from_interface(self):
        pass

class Structure(object):
    def __init__(self):
        self.len_interfaces = None
        self.len_series_i = None
        self.len_series_o = None
        self.reference_position = None


class AdditionalData(object, Structure):
    def __init__(self):
        self.u_grade = None
        self.range_var = None
        self.c_o = None
        self.nugget_effect_gradient = None
        self.nugget_effect_scalar = None

    def default_range(self):
        pass

    def default_c_o(self):
        pass

    def get_kriging_parameters(self):
        pass

class GeoPhysiscs(object):
    def __init__(self):
        self.gravity = None
        self.magnetics = None

    def create_geophy(self):
        pass

    def set_gravity_precomputations(self):
        pass


class Interpolator(object):
    def __init__(self, input_matrices: np.ndarray):
        self.verbose = None
        self.dtype = None
        self.output = None
        self.theano_optimizer = None
        self.is_lith=None
        self.is_fault=None

        import gempy.theano_graph as tg
        self.input_matrices = input_matrices
        self.theano_graph = tg
        self.theano_function = None

    def compile_th_fn(self):
        pass

