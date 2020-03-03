extent = '[x_min, x_max, y_min, y_max, z_min, z_max] '
resolution = '(np.ndarray): [nx, ny, nz] '

coord = '2D numpy array where axis 1 is the XYZ coordinates while axis 0 is n number of input '
coord_ori = coord + 'Notice that orientations may be place anywhere in the 3D space.'

pole_vector = '2D numpy array where axis 1 is the gradient values G_x, G_y, G_z of the pole while axis 0 is n number of' \
              ' orientations.'

orientations = '2D numpy array where axis 1 is are orientation values [dip, azimuth, polarity] of the pole while axis 0' \
               ' is n number of orientations. --- ' \
               '*Dip* is the inclination angle of 0 to 90 degrees measured from the horizontal plane downwards. ' \
               '*Azimuth* is the dip direction defined by a 360 degrees clockwise rotation, i.e. 0 = North,' \
               ' 90 = East, 180 = South, and 270 = West.' \
               '*Polarity* defines where the upper (geologically younger) side of the orientation plane ' \
               'is and can be declared to be either normal (1) or reversed (-1).' \
               'The orientation plane is perpendicular to the gradient.'

surface_sp = 'list with the surface names for each input point. They must exist in the surfaces ' \
             'object linked to SurfacePoints.'

x = 'values or list of values for the x coordinates'
y = 'values or list of values for the y coordinates'
z = 'values or list of values for the z coordinates'
idx_sp = 'If passed, list of indices where the function will be applied.'

file_path = 'Any valid string path is acceptable. The string could be a URL. Valid URL schemes include http, ftp, s3,' \
            ' and file. For file URLs, a host is expected. A local file could be: file://localhost/path/to/table.csv. '\
            'If you want to pass in a path object, pandas accepts either pathlib.Path or py._path.local.LocalPath.' \
            ' By file-like object, we refer to objects with a read() method, such as a file handler (e.g. via builtin '\
            'open function) or StringIO.'

debug = 'of debug is True the method will return the result without modify any related object'
inplace = 'if True, perform operation in-place'

centers = 'XYZ array with the center of the data. This controls how much we shift the input coordinates'
rescaling_factor = 'Scaling factor by which all the parameters will be rescaled.'

theano_graph_pro = 'GemPy object that contains all graph structure of theano'

ctrl = 'List that controls what parts of the graph for each series have to be recomputed.'

weights_vector = 'Numpy array that containing the kriging weights for each input data sorted by series.'
sfai = 'Value of the scalar field at each interface. Axis 0 is each series and axis 1 contain each surface in order'
bai = '3D array with all interpolated values for a given series and at the interfaces'
mai = 'Boolean array containing the logic to combine multiple series to obtain the final model at each interface.'
vai = '2D array with the final values once the superposition of series has been carried out at each interface.'

lith_block = ' Array with the id of each layer evaluated in each point of the regular grid. '
sfm = 'Value of the scalar field at each value of the regular grid. '
bm = '3D array with all interpolated values for a given series and at each value of the regular grid. '
mm = 'Boolean array containing the logic to combine multiple series to obtain the final model at each value of the' \
     ' regular grid. '
vm = '2D array with the final values once the superposition of series has been carried out at each value of the ' \
     'regular grid.'

vertices = 'List of numpy arrays containing the XYZ coordinates of each triangle vertex.'
edges = 'List of numpy arrays containing the indices of the vertices numpy arrays that compose each individual' \
        ' triangle. '
geological_map = '2D array containing the lithologies at the surfaces. '

recompute_rf = 'if True recompute the rescaling factor'

compile_theano = 'Default true. Either if the theano graph must be compiled or not'
theano_optimizer = 'Type of theano compilation. This rules the number ' \
                   'optimizations theano performs at compilation time: fast_run will take longer \
             to compile but at run time will be faster and will consume significantly less memory. fast_compile will \
             compile faster.'