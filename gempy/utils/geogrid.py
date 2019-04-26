'''Module with classes and methods to analyse and process exported geomodel grids
Created on 21/03/2014
@author: Florian Wellmann (some parts originally developed by Erik Schaeffer)
'''
import numpy as np
#import pynoddy
import subprocess
import os.path
import platform

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n\n\tMatplotlib not installed - plotting functions will not work!\n\n\n")

# import mpl_toolkits
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# to convert python variable types to cpp types
import ctypes
# to create array
from numpy.ctypeslib import ndpointer
# to create folder
import os
# read out and change xml file (here only used to read out model boundary information)
import geomodeller_xml_obj as GO


class GeoGrid():
    """Object definition for exported geomodel grids"""

    def __init__(self, **kwds):
        """GeoGrid contains methods to load, analyse, and process exported geomodel grids
        **Optional Keywords**:
            - *grid_filename* = string : filename of exported grid
            - *delxyz_filename* = string : file with model discretisation
            - *dimensions_filename* = string : file with model dimension (coordinates)
        """

        if 'grid_filename' in kwds:
            self.grid_filename = kwds['grid_filename']
        if 'delxyz_filename' in kwds:
            self.delxyz_filename = kwds['delxyz_filename']
        if 'dimensions_filename' in kwds:
            self.dimensions_filename = kwds['dimensions_filename']

    def __add__(self, G_other):
        """Combine grid with another GeoGrid if regions are overlapping"""
        # check overlap
        print (self.ymin, self.ymax)
        print (G_other.ymin, G_other.ymax)
        if (G_other.ymin < self.ymax and G_other.ymin > self.ymin):
            print("Grids overlapping in y-direction between %.0f and %.0f" %
                  (G_other.ymin, self.ymax))

    def load_grid(self):
        """Load exported grid, discretisation and dimensions from file"""
        if not hasattr(self, 'grid_filename'):
            raise AttributeError("Grid filename is not defined!")
        self.grid = np.loadtxt(self.grid_filename,
                               delimiter = ',',
                               dtype='int',
                               unpack=False)
        if hasattr(self, 'delxyz_filename'):
            self.load_delxyz(self.delxyz_filename)
            self.adjust_gridshape()
        if hasattr(self, 'dimensions_filename'):
            self.load_dimensions(self.dimensions_filename)

    def load_delxyz(self, delxyz_filename):
        """Load grid discretisation from file"""
        del_lines = open(delxyz_filename, 'r').readlines()
        d0 = del_lines[0].split("*")
        self.delx = np.array([float(d0[1]) for _ in range(int(d0[0]))])
        d1 = del_lines[1].split("*")
        self.dely = np.array([float(d1[1]) for _ in range(int(d1[0]))])
        d2 = del_lines[2].split(",")[:-1]
        self.delz = np.array([float(d) for d in d2])
        (self.nx, self.ny, self.nz) = (len(self.delx), len(self.dely), len(self.delz))
        (self.extent_x, self.extent_y, self.extent_z) = (sum(self.delx), sum(self.dely), sum(self.delz))

    def set_delxyz(self, delxyz):
        """Set delx, dely, delz arrays explicitly and update additional attributes
        **Arguments**:
            - *delxyz* = (delx-array, dely-array, delz-array): arrays with cell dimensions
        """
        self.delx, self.dely, self.delz = delxyz
        (self.nx, self.ny, self.nz) = (len(self.delx), len(self.dely), len(self.delz))
        (self.extent_x, self.extent_y, self.extent_z) = (sum(self.delx), sum(self.dely), sum(self.delz))

    def set_basename(self, name):
        """Set basename for grid exports, etc.
        **Arguments**:
            - *name* = string: basename
        """
        self.basename = name

    def load_dimensions(self, dimensions_filename):
        """Load project dimensions from file"""
        dim = [float(d) for d in open(dimensions_filename, 'r').readlines()[1].split(",")]
        (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax) = dim
        # calculate cell centre positions in real world coordinates

    def define_regular_grid(self, nx, ny, nz):
        """Define a regular grid from defined project boundaries and given discretisations"""
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.delx = np.ones(nx) * (self.xmax - self.xmin) / nx
        self.dely = np.ones(ny) * (self.ymax - self.ymin) / ny
        self.delz = np.ones(nz) * (self.zmax - self.zmin) / nz
        # create (empty) grid object
        self.grid = np.ndarray((nx, ny, nz))
        # update model extent
        (self.extent_x, self.extent_y, self.extent_z) = (sum(self.delx), sum(self.dely), sum(self.delz))


    def define_irregular_grid(self, delx, dely, delz):
        """Set irregular grid according to delimter arrays in each direction"""
        self.delx = delx
        self.dely = dely
        self.delz = delz
        self.nx = len(delx)
        self.ny = len(dely)
        self.nz = len(delz)
        # create (empty) grid object
        self.grid = np.ndarray((self.nx, self.ny, self.nz))
        # update model extent
        (self.extent_x, self.extent_y, self.extent_z) = (sum(self.delx), sum(self.dely), sum(self.delz))

    def get_dimensions_from_geomodeller_xml_project(self, xml_filename):
        """Get grid dimensions from Geomodeller project
        **Arguments**:
            - *xml_filename* = string: filename of Geomodeller XML file
        """
        # Note: this implementation is based on the Geomodeller API
        # The boundaries could theoretically also be extracted from the XML file
        # directly, e.g. using the geomodeller_xml_obj module - but this would
        # require an additional module being loaded, so avoid here!
        filename_ctypes = ctypes.c_char_p(xml_filename)
        # get model boundaries
            #Detection of operative system:
        if platform.system() == "Linux":
            lib = ctypes.CDLL('./libgeomod.so') #linux
        elif platform.system() == "Windows":
            lib = ctypes.windll.LoadLibrary(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "libgeomodwin_leak6.dll")     #windows
        else:
            print("Your operative system is not supported")
        lib.get_model_bounds.restype = ndpointer(dtype=ctypes.c_int, shape=(6,))
        boundaries = lib.get_model_bounds(filename_ctypes)
        (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax) = boundaries
        self.extent_x = self.xmax - self.xmin
        self.extent_y = self.ymax - self.ymin
        self.extent_z = self.zmax - self.zmin


    def set_densities(self, densities):
        """Set layer densities
        **Arguments**:
            - *densities* = dictionary of floats: densities for geology ids
        """
        self.densities = densities

    def set_sus(self, sus):
        """Set layer susceptibilities
        **Arguments**:
            - *us* = dictionary of floats: magnetic susceptibilities for geology ids
        """
        self.sus = sus

    def set_dimensions(self, **kwds):
        """Set model dimensions, if no argument provided: xmin = 0, max = sum(delx) and accordingly for y,z
        **Optional keywords**:
            - *dim* = (xmin, xmax, ymin, ymax, zmin, zmax) : set dimensions explicitly
        """
        if "dim" in kwds:
            (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax) = kwds['dim']
        else:
            self.xmin, self.ymin, self.zmin = (0., 0., 0.)
            self.xmax, self.ymax, self.zmax = (sum(self.delx), sum(self.dely), sum(self.delz))

    def determine_cell_centers(self):
        """Determine cell centers for all coordinate directions in "real-world" coordinates"""
        if not hasattr(self, 'xmin'):
            raise AttributeError("Please define grid dimensions first")
        sum_delx = np.cumsum(self.delx)
        sum_dely = np.cumsum(self.dely)
        sum_delz = np.cumsum(self.delz)
        self.cell_centers_x = np.array([sum_delx[i] - self.delx[i] / 2. for i in range(self.nx)]) + self.xmin
        self.cell_centers_y = np.array([sum_dely[i] - self.dely[i] / 2. for i in range(self.ny)]) + self.ymin
        self.cell_centers_z = np.array([sum_delz[i] - self.delz[i] / 2. for i in range(self.nz)]) + self.zmin

    def determine_cell_boundaries(self):
        """Determine cell boundaries for all coordinates in "real-world" coordinates"""
        if not hasattr(self, 'xmin'):
            raise AttributeError("Please define grid dimensions first")
        sum_delx = np.cumsum(self.delx)
        sum_dely = np.cumsum(self.dely)
        sum_delz = np.cumsum(self.delz)
        self.boundaries_x = np.ndarray((self.nx+1))
        self.boundaries_x[0] = 0
        self.boundaries_x[1:] = sum_delx
        self.boundaries_y = np.ndarray((self.ny+1))
        self.boundaries_y[0] = 0
        self.boundaries_y[1:] = sum_dely
        self.boundaries_z = np.ndarray((self.nz+1))
        self.boundaries_z[0] = 0
        self.boundaries_z[1:] = sum_delz

        # create a list with all bounds
        self.bounds = [self.boundaries_y[0], self.boundaries_y[-1],
                       self.boundaries_x[0], self.boundaries_x[-1],
                       self.boundaries_z[0], self.boundaries_z[-1]]


    def adjust_gridshape(self):
        """Reshape numpy array to reflect model dimensions"""
        self.grid = np.reshape(self.grid, (self.nz, self.ny, self.nx))
        self.grid = np.swapaxes(self.grid, 0, 2)
        # self.grid = np.swapaxes(self.grid, 0, 1)


    def export_to_vtk(self, vtk_filename="geo_grid", real_coords = True, **kwds):
        """Export grid to VTK for visualisation
        **Arguments**:
            - *vtk_filename* = string : vtk filename (obviously...)
            - *real_coords* = bool : model extent in "real world" coordinates
        **Optional Keywords**:
            - *grid* = numpy grid : grid to save to vtk (default: self.grid)
            - *var_name* = string : name of variable to plot (default: Geology)
        Note: requires pyevtk, available at: https://bitbucket.org/pauloh/pyevtk
        """
        grid = kwds.get("grid", self.grid)
        var_name = kwds.get("var_name", "Geology")
        #from evtk.hl import gridToVTK
        import pyevtk
        from pyevtk.hl import gridToVTK
        # define coordinates
        x = np.zeros(self.nx + 1)
        y = np.zeros(self.ny + 1)
        z = np.zeros(self.nz + 1)
        x[1:] = np.cumsum(self.delx)
        y[1:] = np.cumsum(self.dely)
        z[1:] = np.cumsum(self.delz)



        # plot in coordinates
        if real_coords:
            x += self.xmin
            y += self.ymin
            z += self.zmin


        gridToVTK(vtk_filename, x, y, z,
                  cellData = {var_name: grid})

    def export_to_csv(self, filename = "geo_grid.csv"):
        """Export grid to x,y,z,value pairs in a csv file
        Ordering is x-dominant (first increase in x, then y, then z)
        **Arguments**:
            - *filename* = string : filename of csv file (default: geo_grid.csv)
        """
        f = open(filename, 'w')
        for zz in self.delz:
            for yy in self.dely:
                for xx in self.delx:
                    f.write("%.1f,%.1f,%.1f,%.d" % (xx,yy,zz,self.grid[xx,yy,zz]))
        f.close()


    def determine_geology_ids(self):
        """Determine all ids assigned to cells in the grid"""
        self.unit_ids = np.unique(self.grid)

    def get_name_mapping_from_file(self, filename):
        """Get the mapping between unit_ids in the model and real geological names
        from a csv file (e.g. the SHEMAT property file)
        **Arguments**:
            - *filename* = string : filename of csv file with id, name entries
        """
        self.unit_name = {}
        filelines = open(filename, 'r').readlines()[1:]
        for line in filelines:
            l = line.split(",")
            self.unit_name[int(l[1])] = l[0]

    def get_name_mapping_from_dict(self, unit_name_dict):
        """Get the name mapping directly from a dictionary
        **Arguments**:
            - *unit_name_dict* = dict with "name" : unit_id (int) pairs
        """
        self.unit_name = unit_name_dict


    def remap_ids(self, mapping_dictionary):
        """Remap geological unit ids to new ids as defined in mapping dictionary
        **Arguments**:
            - *mapping_dictionary* = dict : {1 : 1, 2 : 3, ...} : e.g.: retain
            id 1, but map id 2 to 3 (note: if id not specified, it will be retained)
        """
        # first step: create a single mesh for each id to avoid accidential
        # overwriting below (there might be a better solution...)
        if not hasattr(self, 'unit_ids'):
            self.determine_geology_ids()
        geol_grid_ind = {}
        for k,v in mapping_dictionary.items():
            geol_grid_ind[k] = self.grid == k
            print("Remap id %d -> %d" % (k,v))
        # now reassign values in actual grid
        for k,v in mapping_dictionary.items():
            print("Reassign id %d to grid" % v)
            self.grid[geol_grid_ind[k]] = v
        # update global geology ids
        self.determine_geology_ids()

    def determine_cell_volumes(self):
        """Determine cell volumes for each cell (e.g. for total formation volume calculation)"""
        self.cell_volume = np.ndarray(np.shape(self.grid))
        for k,dz in enumerate(self.delz):
            for j,dy in enumerate(self.dely):
                for i,dx in enumerate(self.delx):
                    self.cell_volume[i,j,k] = dx * dy * dz


    def determine_indicator_grids(self):
        """Determine indicator grids for all geological units"""
        self.indicator_grids = {}
        if not hasattr(self, 'unit_ids'):
            self.determine_geology_ids()
        grid_ones = np.ones(np.shape(self.grid))
        for unit_id in self.unit_ids:
            self.indicator_grids[unit_id] = grid_ones * (self.grid == unit_id)

    def determine_id_volumes(self):
        """Determine the total volume of each unit id in the grid
        (for example for cell discretisation studies, etc."""
        if not hasattr(self, 'cell_volume'):
            self.determine_cell_volumes()
        if not hasattr(self, 'indicator_grids'):
            self.determine_indicator_grids()
        self.id_volumes = {}
        for unit_id in self.unit_ids:
            self.id_volumes[unit_id] = np.sum(self.indicator_grids[unit_id] * self.cell_volume)

    def print_unit_names_volumes(self):
        """Formatted output to STDOUT of unit names (or ids, if names are note
        defined) and calculated volumes
        """
        if not hasattr(self, 'id_vikumes'):
            self.determine_id_volumes()

        if hasattr(self, "unit_name"):
            # print with real geological names
            print("Total volumes of modelled geological units:\n")
            for unit_id in self.unit_ids:
                print("%26s : %.2f km^3" % (self.unit_name[unit_id],
                                            self.id_volumes[unit_id]/1E9))
        else:
            # print with unit ids only
            print("Total volumes of modelled geological units:\n")
            for unit_id in self.unit_ids:
                print("%3d : %.2f km^3" % (unit_id,
                                            self.id_volumes[unit_id]/1E9))


    def extract_subgrid(self, subrange, **kwds):
        """Extract a subgrid model from existing grid
        **Arguments**:
            - *subrange* = (x_from, x_to, y_from, y_to, z_from, z_to) : range for submodel in either cell or world coords
        **Optional keywords**:
            - *range_type* = 'cell', 'world' : define if subrange in cell ids (default) or real-world coordinates
        """
        range_type = kwds.get('range_type', 'cell')

        if not hasattr(self, 'boundaries_x'):
            self.determine_cell_boundaries()

        if range_type == 'world':
            # determine cells
            subrange[0] = np.argwhere(self.boundaries_x > subrange[0])[0][0]
            subrange[1] = np.argwhere(self.boundaries_x < subrange[1])[-1][0]
            subrange[2] = np.argwhere(self.boundaries_y > subrange[2])[0][0]
            subrange[3] = np.argwhere(self.boundaries_y < subrange[3])[-1][0]
            subrange[4] = np.argwhere(self.boundaries_z > subrange[4])[0][0]
            subrange[5] = np.argwhere(self.boundaries_z < subrange[5])[-1][0]

        # create a copy of the original grid
        import copy
        subgrid = copy.deepcopy(self)

        # extract grid
        subgrid.grid = self.grid[subrange[0]:subrange[1],
                                 subrange[2]:subrange[3],
                                 subrange[4]:subrange[5]]

        subgrid.nx = subrange[1] - subrange[0]
        subgrid.ny = subrange[3] - subrange[2]
        subgrid.nz = subrange[5] - subrange[4]

        # update extent
        subgrid.xmin = self.boundaries_x[subrange[0]]
        subgrid.xmax = self.boundaries_x[subrange[1]]
        subgrid.ymin = self.boundaries_y[subrange[2]]
        subgrid.ymax = self.boundaries_y[subrange[3]]
        subgrid.zmin = self.boundaries_z[subrange[4]]
        subgrid.zmax = self.boundaries_z[subrange[5]]

        subgrid.extent_x = subgrid.xmax - subgrid.xmin
        subgrid.extent_y = subgrid.ymax - subgrid.ymin
        subgrid.extent_z = subgrid.zmax - subgrid.zmin

        # update cell spacings
        subgrid.delx = self.delx[subrange[0]:subrange[1]]
        subgrid.dely = self.dely[subrange[2]:subrange[3]]
        subgrid.delz = self.delz[subrange[4]:subrange[5]]

        # now: update other attributes:
        subgrid.determine_cell_centers()
        subgrid.determine_cell_boundaries()
        subgrid.determine_cell_volumes()
        subgrid.determine_geology_ids()

        # finally: return subgrid
        return subgrid



# ******************************************************************************
#  Some additional helper functions
# ******************************************************************************

def combine_grids(G1, G2, direction, merge_type = 'keep_first', **kwds):
    """Combine two grids along one axis
    ..Note: this implementation assumes (for now) that the overlap is perfectly matching,
    i.e. grid cell sizes identical and at equal positions, or that they are perfectly adjacent!
    **Arguments**:
        - G1, G2 = GeoGrid : grids to be combined
        - direction = 'x', 'y', 'z': direction in which grids are combined
        - merge_type = method to combine grid:
            'keep_first' : keep elements of first grid (default)
            'keep_second' : keep elements of second grid
            'random' : randomly choose an element to retain
        ..Note: all other dimensions must be matching perfectly!!
    **Optional keywords**:
        - *overlap_analysis* = bool : perform a detailed analysis of the overlapping area, including
        mismatch. Also returns a second item, a GeoGrid with information on mismatch!
    **Returns**:
        - *G_comb* = GeoGrid with combined grid
        - *G_overlap* = Geogrid with analysis of overlap (of overlap_analysis=True)
    """
    overlap_analysis = kwds.get("overlap_analysis", False)
    # first step: determine overlap
    if direction == 'x':
        if G2.xmax > G1.xmax:
            overlap_min = G2.xmin
            overlap_max = G1.xmax
            # identifier alias for grids with higher/ lower values
            G_high = G2
            G_low = G1
        else:
            overlap_min = G1.xmin
            overlap_max = G2.xmax
            # identifier alias for grids with higher/ lower values
            G_high = G1
            G_low = G2

        # check if all other dimensions are perfectly matching
        if (G1.ymin != G2.ymin) or (G1.zmin != G2.zmin) or \
            (G1.ymax != G2.ymax) or (G1.zmax != G2.zmax):
            raise ValueError("Other dimensions (apart from %s) not perfectly matching! Check and try again!" % direction)

    elif direction == 'y':
        if G2.ymax > G1.ymax:
            overlap_min = G2.ymin
            overlap_max = G1.ymax
            # identifier alias for grids with higher/ lower values
            G_high = G2
            G_low = G1
        else:
            overlap_min = G1.ymin
            overlap_max = G2.ymax
            # identifier alias for grids with higher/ lower values
            G_high = G1
            G_low = G2

        # check if all other dimensions are perfectly matching
        if (G1.xmin != G2.xmin) or (G1.zmin != G2.zmin) or \
            (G1.xmax != G2.xmax) or (G1.zmax != G2.zmax):
            raise ValueError("Other dimensions (apart from %s) not perfectly matching! Check and try again!" % direction)

    elif direction == 'z':
        if G2.zmax > G1.zmax:
            overlap_min = G2.zmin
            overlap_max = G1.zmax
            # identifier alias for grids with higher/ lower values
            G_high = G2
            G_low = G1
        else:
            overlap_min = G1.zmin
            overlap_max = G2.zmax
            # identifier alias for grids with higher/ lower values
            G_high = G1
            G_low = G2

        # check if all other dimensions are perfectly matching
        if (G1.ymin != G2.ymin) or (G1.xmin != G2.xmin) or \
            (G1.ymax != G2.ymax) or (G1.xmax != G2.xmax):
            raise ValueError("Other dimensions (apart from %s) not perfectly matching! Check and try again!" % direction)

    overlap = overlap_max - overlap_min

    if overlap == 0:
        print("Grids perfectly adjacent")
    elif overlap < 0:
        raise ValueError("No overlap between grids! Check and try again!")
    else:
        print("Positive overlap in %s direction of %f meters" % (direction, overlap))

    # determine cell centers
    G1.determine_cell_centers()
    G2.determine_cell_centers()

    # intialise new grid
    G_comb = GeoGrid()
    # initialise overlap grid, if analyis performed
    if overlap_analysis:
        G_overlap = GeoGrid()

    if direction == 'x':
        pass
    elif direction == 'y':
        #=======================================================================
        # Perform overlap analysis
        #=======================================================================

        # initialise overlap grid with dimensions of overlap
        G_overlap.set_dimensions(dim = (G1.xmin, G1.xmax, overlap_min, overlap_max, G1.zmin, G1.zmax))
        G_low_ids = np.where(G_low.cell_centers_y > overlap_min)[0]
        G_high_ids = np.where(G_high.cell_centers_y < overlap_max)[0]
        delx = G1.delx
        dely = G_low.dely[G_low_ids]
        delz = G1.delz
        G_overlap.set_delxyz((delx, dely, delz))
        # check if overlap region is identical
        if not (len(G_low_ids) == len(G_high_ids)):
            raise ValueError("Overlap length not identical, please check and try again!")
        # now: determine overlap mismatch
        G_overlap.grid = G_low.grid[:,G_low_ids,:] - G_high.grid[:,G_high_ids,:]
        # for some very strange reason, this next step is necessary to enable the VTK
        # export with pyevtk - looks like a bug in pyevtk...
        G_overlap.grid = G_overlap.grid + np.zeros(G_overlap.grid.shape)
        #

        #=======================================================================
        # Set up combined grid
        #=======================================================================

        G_comb.set_dimensions(dim = (G1.xmin, G1.xmax, G_low.ymin, G_high.ymax, G1.zmin, G1.zmax))
        # combine dely arrays
        dely = np.hstack((G_low.dely[:G_low_ids[0]], G_high.dely))
        G_comb.set_delxyz((delx, dely, delz))

        #=======================================================================
        # Now merge grids
        #=======================================================================
        if merge_type == 'keep_first':
            if G1.ymax > G2.ymax:
                G_comb.grid = np.concatenate((G2.grid[:,:G_low_ids[0],:], G1.grid), axis=1)
            else:
                G_comb.grid = np.concatenate((G1.grid, G2.grid[:,:G_low_ids[0],:]), axis=1)

        elif merge_type == 'keep_second':
            pass
        elif merge_type == 'random':
            pass
        else:
            raise ValueError("Merge type %s not recognised! Please check and try again!" % merge_type)

    elif direction == 'z':
        pass




    # Return combined grid and results of overlap analysis, if determined
    if overlap_analysis:
        return G_comb, G_overlap
    else:
        return G_comb


def optimial_cell_increase(starting_cell_width, n_cells, width):
    """Determine an array with optimal cell width for a defined starting cell width,
    total number of cells, and total width
    Basically, this function optimised a factor between two cells to obtain a total
    width
    **Arguments**:
    	- *starting_cell_width* = float : width of starting/ inner cell
	- *n_cells* = int : total number of cells
	- *total_width* = float : total width (sum over all elements in array)
    **Returns**:
        del_array : numpy.ndarray with cell discretisations
    Note: optmisation with scipy.optimize - better (analytical?) methods might exist but
    I can't think of them at the moment
    """
    import scipy.optimize
    # define some helper functions

    def width_sum(inc_factor, inner_cell, n_cells, total_width):
        return sum(del_array(inc_factor, inner_cell, n_cells)) - total_width

    def del_array(inc_factor, inner_cell, n_cells):
        return np.array([inner_cell * inc_factor**i for i in range(n_cells)])

    # now the actual optimisation step:
    opti_factor = scipy.optimize.fsolve(width_sum, 1.1, (starting_cell_width, n_cells, width))

    # return the discretisation array
    return del_array(opti_factor, starting_cell_width, n_cells).flatten()
