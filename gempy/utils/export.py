
from matplotlib.cm import ScalarMappable as SM
from gempy.plot._visualization_2d import PlotData2D
import numpy as np
import os
import itertools as it


def export_geomap2geotiff(path, geo_model, geo_map=None, geotiff_filepath=None):
    """

    Args:
        path (str): Filepath for the exported geotiff, must end in .tif
        geo_map (np.ndarray): 2-D array containing the geological map
        cmap (matplotlib colormap): The colormap to be used for the export
        geotiff_filepath (str): Filepath of the template geotiff

    Returns:
        Saves the geological map as a geotiff to the given path.
    """
    from osgeo import gdal

    plot = PlotData2D(geo_model)
    cmap = plot._cmap
    norm = plot._norm

    if geo_map is None:
        geo_map = geo_model.solutions.geological_map[0].reshape(geo_model._grid.topography.resolution)

    if geotiff_filepath is None:
        # call the other function
        print('stupid')

    # **********************************************************************
    geo_map_rgb = SM(norm=norm, cmap=cmap).to_rgba(geo_map.T) # r,g,b,alpha
    # **********************************************************************
    # gdal.UseExceptions()
    ds = gdal.Open(geotiff_filepath)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    outFileName = path
    driver = gdal.GetDriverByName("GTiff")
    options = ['PROFILE=GeoTiff', 'PHOTOMETRIC=RGB', 'COMPRESS=JPEG']
    outdata = driver.Create(outFileName, rows, cols, 3, gdal.GDT_Byte, options=options)

    outdata.SetGeoTransform(ds.GetGeoTransform())  # sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(geo_map_rgb[:, ::-1, 0].T * 256)
    outdata.GetRasterBand(2).WriteArray(geo_map_rgb[:, ::-1, 1].T * 256)
    outdata.GetRasterBand(3).WriteArray(geo_map_rgb[:, ::-1, 2].T * 256)
    outdata.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
    outdata.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
    outdata.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
    # outdata.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)  # alpha band

    # outdata.GetRasterBand(1).SetNoDataValue(999)##if you want these values transparent
    outdata.FlushCache()  # saves to disk
    outdata = None  # closes file (important)
    band = None
    ds = None

    print("Successfully exported geological map to  " +path)

def export_moose_input(geo_model, path=None, filename='geo_model_units_moose_input.i'):
    """
    Method to export a 3D geological model as MOOSE compatible input. 

    Args:
        path (str): Filepath for the exported input file
        filename (str): name of exported input file

    Returns:
        
    """
    # get model dimensions
    nx, ny, nz = geo_model.grid.regular_grid.resolution
    xmin, xmax, ymin, ymax, zmin, zmax = geo_model.solutions.grid.regular_grid.extent
    
    # get unit IDs and restructure them
    ids = np.round(geo_model.solutions.lith_block)
    ids = ids.astype(int)
    
    liths = ids.reshape((nx, ny, nz))
    liths = liths.flatten('F')

    # create unit ID string for the fstring
    idstring = '\n  '.join(map(str, liths))

    # create a dictionary with unit names and corresponding unit IDs
    sids = dict(zip(geo_model._surfaces.df['surface'], geo_model._surfaces.df['id']))
    surfs = list(sids.keys())
    uids = list(sids.values())
    # create strings for fstring, so in MOOSE, units have a name instead of an ID
    surfs_string = ' '.join(surfs)
    ids_string = ' '.join(map(str, uids))
    
    fstring = f"""[MeshGenerators]
  [./gmg]
  type = GeneratedMeshGenerator
  dim = 3
  nx = {nx}
  ny = {ny}
  nz = {nz}
  xmin = {xmin}
  xmax = {xmax}
  yim = {ymin}
  ymax = {ymax}
  zmin = {zmin}
  zmax = {zmax}
  block_id = '{ids_string}'
  block_name = '{surfs_string}'
  [../]
  
  [./subdomains]
    type = ElementSubdomainIDGenerator
    input = gmg
    subdomain_ids = '{idstring}'
  [../]
[]

[Mesh]
  type = MeshGeneratorMesh
[]
"""
    if not path:
        path = './'
    if not os.path.exists(path):
      os.makedirs(path)

    f = open(path+filename, 'w+')
    
    f.write(fstring)
    f.close()
    
    print("Successfully exported geological model as moose input to "+path)

def export_shemat_suite_input_file(geo_model, path: str=None, filename: str='geo_model_SHEMAT_input'):
    """
    Method to export a 3D geological model as SHEMAT-Suite input-file for a conductive HT-simulation. 

    Args:
        path (str): Filepath for the exported input file (default './')
        filename (str): name of exported input file (default 'geo_model_SHEMAT_input')
    """
    # get model dimensions
    nx, ny, nz = geo_model.grid.regular_grid.resolution
    xmin, xmax, ymin, ymax, zmin, zmax = geo_model.solutions.grid.regular_grid.extent
    
    delx = (xmax - xmin)/nx
    dely = (ymax - ymin)/ny
    delz = (zmax - zmin)/nz
    
    # get unit IDs and restructure them
    ids = np.round(geo_model.solutions.lith_block)
    ids = ids.astype(int)
    
    liths = ids.reshape((nx, ny, nz))
    liths = liths.flatten('F')

    # group litho in space-saving way
    sequence = [len(list(group)) for key, group in it.groupby(liths)]
    unit_id = [key for key, group in it.groupby(liths)]
    combined = ["%s*%s" % (pair) for pair in zip(sequence,unit_id)]

    combined_string = " ".join(combined)

    # get number of units and set units string
    units = geo_model.surfaces.df[['surface', 'id']]
    
    unitstring = ""
    for index, rows in units.iterrows():
        unitstring += f"0.01d-10    1.d0  1.d0  1.e-14	 1.e-10  1.d0  1.d0  3.74	0.  2077074.  10  2e-3	!{rows['surface']} \n" 	

    # input file as f-string
    fstring = f"""!==========>>>>> INFO
# Title
{filename}

# linfo
1 2 1 1

# runmode
1

# timestep control
0
1           1           0           0

# tunit
1
 
# time periods, records=1
0      60000000    200      lin
           
# output times, records=10
1
6000000
12000000
18000000
24000000
30000000
36000000
42000000
48000000
54000000
    
# file output: hdf vtk

# active temp

# PROPS=bas

# USER=none


# grid
{nx} {ny} {nz}

# delx
{nx}*{delx}

# dely
{ny}*{dely}

# delz
{nz}*{delz}

!==========>>>>> NONLINEAR SOLVER
# nlsolve
50 0

!==========>>>>> FLOW
# lsolvef (linear solver control)
1.d-12 64 500
# nliterf (nonlinear iteration control)
1.0d-10 1.0

!==========>>>>> TEMPERATURE
# lsolvet (linear solver control)
1.d-12 64 500
# nlitert (nonlinear iteration control)
1.0d-10 1.0

!==========>>>>> INITIAL VALUES
# temp init
{nx*ny*nz}*15.0d0  

# head init
{nx*ny*nz}*7500

!==========>>>>> UNIT DESCRIPTION
!!
# units
{unitstring}

!==========>>>>>   define boundary properties
# temp bcd, simple=top, value=init

# temp bcn, simple=base, error=ignore
{nx*ny}*0.06

# uindex
{combined_string}"""

    if not path:
        path = './'
    if not os.path.exists(path):
        os.makedirs(path)

    f = open(path+filename, 'w+')
    
    f.write(fstring)
    f.close()
    
    print("Successfully exported geological model as SHEMAT-Suite input to "+path) 


def export_pflotran_input(geo_model, path=None, filename='pflotran.ugi'):
    """
    Method to export a 3D geological model as PFLOTRAN implicit unstructured grid
    (see pflotran.org)

    Args:
        path (str): Filepath for the exported input file
        filename (str): name of exported input file

    Returns:
        
    """
    #
    # Added by Moise Rousseau, December 8th, 2020
    # see https://www.pflotran.org/documentation/user_guide/cards/subsurface/grid_card.html
    #
    # create vertices, elements and groups
    vertices, elements, groups = __build_vertices_elements_groups__(geo_model)
    
    #create mesh
    if not path:
        path = './'
    if not os.path.exists(path):
        os.makedirs(path)
    ext = filename.split('.')[-1]
    if ext == 'ugi': 
        mesh_format = 'ascii'
    elif ext == 'h5': 
        mesh_format = 'hdf5'
        try:
            import h5py
        except:
            print("h5py library not installed. Please install it to enable hdf5 output")
            print("Export as ascii instead")
            mesh_format = 'ascii'
            filename += '.ugi'
    else: #assume ugi mesh
        mesh_format = 'ascii'
        filename += '.ugi'
    
    #export mesh
    if mesh_format == 'ascii': #export as ugi
        out = open(path+filename, 'w')
        out.write(f"{len(elements)} {len(vertices)}")
        for element in elements:
            out.write('\nH')
            for x in element[1:]:
              out.write(f" {int(x)}")
        for vertice in vertices:
            out.write('\n')
            out.write(f"{vertice[0]} {vertice[1]} {vertice[2]}")
        out.close()
    elif mesh_format == 'hdf5':
        out = h5py.File(path+filename, 'w')
        out.create_dataset("Domain/Cells", data=elements)
        out.create_dataset("Domain/Vertices", data=vertices)
        out.close()
    
    #topography z faces
    nx, ny, nz = geo_model.grid.regular_grid.resolution
    if len(elements) != nx*ny*nz:
        inter = __generate_topographic_surface__(vertices,elements)
        if mesh_format == 'ascii':
            out = open(path+"topography_surface.ss",'w')
            out.write(str(len(inter))+'\n')
            for elem in inter:
                out.write("Q ")
                for node in elem[1:]:
                    out.write(f"{node} ")
                out.write('\n')
            out.close()
        if mesh_format == "hdf5":
            out = h5py.File(path+filename,'r+')
            out.create_dataset(f"Regions/Topography_surface/Vertex Ids", data=inter)
            out.close()
    
    #export groups
    for grp_name,cell_ids in groups.items():
        if mesh_format == 'ascii':
            out = open(path+grp_name+'.vs','w')
            for x in cell_ids:
                out.write(f"{x}\n")
            out.close()
        if mesh_format == "hdf5":
            out = h5py.File(path+filename,'r+')
            out.create_dataset(f"Regions/{grp_name}/Cell Ids", data=cell_ids)
            out.close()
            
    print("Successfully exported geological model as PFLOTRAN input to "+path+filename)
    return

def export_flac3D_input(geo_model, path=None, filename='geomodel.f3grid'):
    """
    Method to export a 3D geological model FLAC3D readable format

    Args:
        path (str): Filepath for the exported input file
        filename (str): name of exported input file

    Returns:
        
    """
    print("Warning! This is experimental and this have not been validated")
    #
    # Added by Moise Rousseau, December 9th, 2020
    #
    # create vertices and elements
    vertices, elements, groups = __build_vertices_elements_groups__(geo_model)
    
    #open output file
    #if not path:
    #    path = './'
    #if not os.path.exists(path):
    #    os.makedirs(path)

    out = open(path+filename, 'w')
    
    #write gridpoints
    out.write("*GRIDPOINTS")
    for i,vertice in enumerate(vertices):
        out.write(f"\nG {i+1} {vertice[0]} {vertice[1]} {vertice[2]}")
    
    #write elements
    out.write('\n*ZONES')
    for i,elem in enumerate(elements):
        out.write('\nB8')
        for x in elem[1:]:
            out.write(f" {x}")
    
    #make groups
    out.write('\n*GROUPS\n')
    for grp_name,grp in groups.items():
        out.write(f"*ZGROUP {grp_name}\n")
        count = 0
        for x in grp:
            out.write(f"{x} ")
            count += 1
            if count == 8:
                out.write("\n")
                count = 0
        if count != 0: out.write("\n")
    
    out.close()
    print("Successfully exported geological model as FLAC3D input to "+path)
    return



def __build_vertices_elements_groups__(geo_model):
    """
    In: GemPy model class instance
        Generate topo interface (optional)
    Out: Vertices coordinates numpy array
         Elements defined by the number of vertices (always 8)
           and their vertices id (bottom face and top face)
    """
    # get model information
    nx, ny, nz = geo_model.grid.regular_grid.resolution
    xmin, xmax, ymin, ymax, zmin, zmax = geo_model.solutions.grid.regular_grid.extent
    
    # create vertices array
    dx, dy, dz = (xmax-xmin)/nx, (ymax-ymin)/ny, (zmax-zmin)/nz
    n_vertices = (nx+1) * (ny+1) * (nz+1)
    vertices = np.zeros((n_vertices, 3), dtype='f8')
    vertices_ids = np.arange(n_vertices) #used to generate coordinate
    vertices[:,0] = vertices_ids % (nx+1) * dx + xmin
    vertices[:,1] = ( vertices_ids % ( (nx+1)*(ny+1) ) ) // (nx+1) * dy + ymin
    vertices[:,2] = vertices_ids // ( (nx+1)*(ny+1) ) * dz + zmin
    
    #build elements
    n_elements = nx*ny*nz
    element_ids = np.arange(n_elements) #used to generate elems
    elements = np.zeros((n_elements,9), dtype='i8')
    i = element_ids % nz
    j = element_ids // nz % ny
    k = element_ids // (nz*ny)
    elements[:,0] = 8 #all hex
    elements[:,1] = 1 + i*(nx+1)*(ny+1) + j*(nx+1) + k
    elements[:,2] = elements[:,1] + 1
    elements[:,3] = elements[:,2] + (nx+1)
    elements[:,4] = elements[:,3] - 1
    elements[:,5] = elements[:,1] + ( (nx+1)*(ny+1) )
    elements[:,6] = elements[:,5] + 1
    elements[:,7] = elements[:,6] + (nx+1)
    elements[:,8] = elements[:,7] - 1
    
    #build groups
    lith_ids = np.round(geo_model.solutions.lith_block)
    lith_ids = lith_ids.astype(int)
    sids = dict(zip(geo_model._surfaces.df['surface'], geo_model._surfaces.df['id']))
    groups = {}
    for region_name,region_id in sids.items():
        cell_ids = np.where(lith_ids == region_id)[0] + 1
        if not len(cell_ids): continue
        groups[region_name] = cell_ids
            
    #remove element above topography
    lith_ids = np.round(geo_model.solutions.lith_block)
    lith_ids = lith_ids.astype(int)
    inactive_cells = ~np.isin(lith_ids, geo_model._surfaces.df['id']) #get cell not on a lith_block
    if np.any(inactive_cells):
        #update correspondance
        new_id_vertices = np.zeros(len(vertices),dtype='i8')
        new_id_elements = np.zeros(len(elements),dtype='i8')
        #remove inactive cell
        elements = elements[~inactive_cells]
        new_id_elements[~inactive_cells] = np.arange(len(elements))+1
        #remove deleted vertices
        cond = np.isin(np.arange(len(vertices))+1, elements[:,1:].flatten())
        vertices = vertices[cond]
        new_id_vertices[cond] = np.arange(len(vertices))+1
        #renumber vertices in element
        elements[1:] = new_id_vertices[elements[1:]-1]
        #renumber groups
        for grp in groups.values():
          grp[:] = new_id_elements[grp-1]
    
    return vertices, elements, groups


def __generate_topographic_surface__(vertices, elements):
    """
    In: Vertices array generated by __build_vertices_and_elements__
        Elements array generated by __build_vertices_and_elements__
    Out: Topography surface defined by the top faces
    """
    inter = np.zeros((len(elements),5), dtype='i8')
    count = 1
    inter[:,0] = 4 #all quad face
    inter[0, 1:] = elements[-1,5:] 
    last_node_z = -1e20
    for ie,elem in enumerate(elements):
        node = elem[-1]
        node_z = vertices[node-1][2]
        if node_z < last_node_z:
            inter[count,1:] = elements[ie-1,5:]
            count += 1
        last_node_z = node_z
    inter = inter[:count]
    return inter
        
