coord = '2D numpy array where axis 1 is the XYZ coordinates while axis 0 is n number of input '
coord_ori = coord + 'Notice that orientations may be place anywhere in the 3D space.'

pole_vector = '2D numpy array where axis 1 is the gradient values G_x, G_y, G_z of the pole while axis 0 is n number of' \
              'orientations.'

orientations = '2D numpy array where axis 1 is are orientation values [dip, azimuth, polarity] of the pole while axis 0' \
               ' is n number of orientations.'

surface_sp = 'list with the surface names for each input point. They must exist in the surfaces ' \
             'object linked to SurfacePoints '

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
