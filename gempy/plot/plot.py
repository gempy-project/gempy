"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.

    Module with classes and methods to perform implicit regional modelling based on
    the potential field method.
    Tested on Ubuntu 16

    Created on 10/04/2018

    @author: Elisa Heim, Miguel de la Varga
"""

# This is for sphenix to find the packages
# sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from typing import Set, Tuple, Dict, Union
from nptyping import Array
import gempy as _gempy
from .visualization_2d import PlotData2D, PlotSolution
from .visualization_3d import GemPyvtkInteract


def plot_data_3D(geo_data, ve=1, **kwargs):
    """
    Plot in vtk all the input data of a model
    Args:
        geo_data (gempy.DataManagement.InputData): Input data of the model

    Returns:
        None
    """
    vv = GemPyvtkInteract(geo_data, ve=ve, **kwargs)
    # vv.restart()
    vv.set_surface_points()
    vv.set_orientations()
    vv.render_model(**kwargs)

    return vv


def plot_3D(geo_model, render_surfaces=True, render_data=True,
            render_topography=True,
            real_time=False, **kwargs):
    """
        Plot in vtk all the input data of a model
        Args:
            geo_model (gempy.DataManagement.InputData): Input data of the model

        Returns:
            None
        """
    vv = GemPyvtkInteract(geo_model, real_time=real_time, **kwargs)
    # vv.restart()
    if render_data is True:
        vv.set_surface_points()
        vv.set_orientations()
    if render_surfaces is True:
        vv.set_surfaces(geo_model.surfaces)
    if render_topography is True and geo_model.grid.topography is not None:
        vv.set_topography()

    vv.render_model(**kwargs)

    return vv


def export_to_vtk(geo_data, path=None, name=None, voxels=True, block=None, surfaces=True):
    """
      Export data to a vtk file for posterior visualizations

      Args:
          geo_data(:class:`Model`)
          path(str): path to the location of the vtk
          name(str): Name of the files. Default name: Default
          voxels(bool): if True export lith_block
          block(Optional[np.array]): One of the solutions of the regular grid. This can be used if for
           example you want to export an scalar field or an specific series block. If None is passed, lith_block
           will be exported.
          surfaces(bool): If True, export the polydata surfaces.

      Returns:
          None
      """

    if voxels is True:
        GemPyvtkInteract.export_vtk_lith_block(geo_data, lith_block=block,
                                               path=path)
    if surfaces is True:
        geo_data.solutions.compute_all_surfaces()
        ver, sim = _gempy.get_surfaces(geo_data)
        GemPyvtkInteract.export_vtk_surfaces(geo_data, ver, sim, path=path,
                                             name=name)
    return True


def plot_data(geo_data, direction="y", data_type='all', series="all",
              show_legend=True, **kwargs):
    """
    Plot the projection of the raw data (surface_points and orientations) in 2D following a
    specific directions

    Args:
        direction(str): xyz. Caartesian direction to be plotted
        series(str): series to plot
        ve(float): Vertical exageration
        **kwargs: seaborn lmplot key arguments. (TODO: adding the link to them)

    Returns:
        None
    """
    plot = PlotData2D(geo_data)
    p = plot.plot_data(direction=direction, data_type=data_type, series=series,
                       show_legend=show_legend, **kwargs)
    # TODO saving options
    return plot


def plot_stereonet(geo_data, litho=None, planes=True, poles=True,
                   single_plots=False,
                   show_density=False):
    '''
    Plot an equal-area projection of the orientations dataframe using mplstereonet.

    Args:
        geo_model (gempy.DataManagement.InputData): Input data of the model
        series_only: To select whether a stereonet is plotted perries or per formation
        litho: selection of formation or series names, as list. If None, all are plotted
        planes: If True, azimuth and dip are plotted as great circles
        poles: If True, pole points (plane normal vectors) of azimuth and dip are plotted
        single_plots: If True, each formation is plotted in a single stereonet
        show_density: If True, density contour plot around the pole points is shown

    Returns:
        None
    '''

    plot = PlotData2D(geo_data)
    plot.plot_stereonet(litho=litho, planes=planes, poles=poles,
                        single_plots=single_plots,
                        show_density=show_density)


def plot_map(model, contour_lines=True, show_data=True, figsize=(12, 12)):
    """

    Args:
        model:
        contour_lines:
        show_faults:
        show_data:

    Returns:

    """
    plot = PlotSolution(model)
    plot.plot_map(contour_lines=contour_lines, show_data=show_data,
                  figsize=figsize)


def plot_section_traces(model, section_names=None, contour_lines=False,
                        show_data=True, show_all_data=False):
    """

    Args:
        model:
        show_data:
        section_names:
        contour_lines:

    Returns:

    """
    plot = PlotSolution(model)
    if plot.model.solutions.geological_map is not None:
        plot.plot_map(contour_lines=contour_lines, show_data=show_data,
                      show_all_data=show_all_data)
    # else:
    # fig = plt.figure()
    # plt.title('Section traces, z direction')

    plot.plot_section_traces(show_data=show_data, section_names=section_names,
                             contour_lines=contour_lines,
                             show_all_data=show_all_data)


"""
def plot_predef_sections(model, show_traces=True, show_data=False, section_names=None, show_faults=True,
                         show_topo=True, figsize=(12, 12)):


    Args:
        model:
        show_traces:
        show_data:
        section_names:
        show_faults:
        show_topo:
        figsize:

    Returns:


    plot = PlotSolution(model)
    plot.plot_sections(show_traces=show_traces, show_data=show_data, section_names=section_names,
                       show_faults=show_faults, show_topo=show_topo, figsize=figsize)

"""


def plot_section_by_name(model, section_name, show_faults=True, show_topo=True,
                         show_data=True,
                         show_all_data=False, radius='default',
                         contourplot=True):
    # Todo needs more keywords:
    ### if show_data: radius, data_type
    plot = PlotSolution(model)
    plot.plot_section_by_name(section_name=section_name, show_topo=show_topo,
                              show_faults=show_faults,
                              show_data=show_data, show_all_data=show_all_data,
                              radius=radius, contourplot=contourplot)


def plot_all_sections(model, show_data=False, section_names=None,
                      show_topo=True, figsize=(12, 12)):
    plot = PlotSolution(model)
    plot.plot_all_sections(show_data=show_data, section_names=section_names,
                           show_topo=show_topo,
                           figsize=figsize)


def plot_section(model, cell_number=13, block=None, direction="y",
                 interpolation='none',
                 show_data=True, show_faults=True, show_topo=False,
                 block_type=None, ve=1,
                 show_all_data=False, show_legend=True, **kwargs):
    """
    Plot a section of the block model

    Args:
        cell_number(int): position of the array to plot
        direction(str): xyz. Caartesian direction to be plotted
        interpolation(str): Type of interpolation of plt.imshow. Default 'none'.  Acceptable values are 'none'
        ,'nearest', 'bilinear', 'bicubic',
        'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
        'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
        'lanczos'
        ve(float): Vertical exageration
        **kwargs: imshow keywargs

    Returns:
        None
    """
    plot = PlotSolution(model)
    plot.fig = plot.plot_block_section(model.solutions, cell_number, block,
                                       direction, interpolation,
                                       show_data, show_faults, show_topo,
                                       block_type, ve,
                                       show_all_data=show_all_data,
                                       show_legend=show_legend, **kwargs)
    return plot


def plot_scalar_field(model, cell_number, N=20,
                      direction="y", block=None, alpha=0.6, show_data=True,
                      show_all_data=False, series=0, *args, **kwargs):
    """
    Plot a potential field in a given direction.

    Args:
        cell_number(int): position of the array to plot
        potential_field(str): name of the potential field (or series) to plot
        n_pf(int): number of the  potential field (or series) to plot
        direction(str): xyz. Caartesian direction to be plotted
        serie: *Deprecated*
        **kwargs: plt.contour kwargs

    Returns:
        None
    """
    plot = PlotSolution(model)
    if block is not None:
        block = block
    else:
        block = model.solutions

    plot.plot_scalar_field(block, cell_number, N=N,
                           direction=direction, show_data=show_data,
                           series=series, alpha=alpha,
                           show_all_data=show_all_data,
                           *args, **kwargs)


def plot_section_scalarfield(model, section_name, sn, levels=50,
                             show_faults=True, show_topo=True, lithback=True):
    """
    Plot the potential field in the predefined sections.
    Args:
        model:
        section_name: name of the section
        sn: scalar field number, order like in model.series
        levels: number of isolines you want to plot
        show_faults: whether or not faults should be plotted
        show_topo: whether or not the topography should be plotted
        lithback: lithology background

    Returns:
        None
    """
    plot = PlotSolution(model)
    plot.plot_section_scalarfield(section_name=section_name, sn=sn,
                                  levels=levels, show_faults=show_faults,
                                  show_topo=show_topo, lithback=lithback)


def plot_gradient(geo_data, scalar_field, gx, gy, gz, cell_number, q_stepsize=5,
                  direction="y", plot_scalar=True, **kwargs):
    """
        Plot the gradient of the scalar field in a given direction.

        Args:
            geo_data (gempy.DataManagement.InputData): Input data of the model
            scalar_field(numpy.array): scalar field to plot with the gradient
            gx(numpy.array): gradient in x-direction
            gy(numpy.array): gradient in y-direction
            gz(numpy.array): gradient in z-direction
            cell_number(int): position of the array to plot
            q_stepsize(int): step size between arrows to indicate gradient
            direction(str): xyz. Caartesian direction to be plotted
            plot_scalar(bool): boolean to plot scalar field
            **kwargs: plt.contour kwargs

        Returns:
            None
    """
    plot = PlotSolution(geo_data)
    plot.plot_gradient(scalar_field, gx, gy, gz, cell_number,
                       q_stepsize=q_stepsize,
                       direction=direction, plot_scalar=plot_scalar,
                       **kwargs)


def plot_topology(
        geo_model,
        edges: Set[Tuple[int, int]],
        centroids: Dict[int, Array[int, 3]],
        direction: Union["x", "y", "z"] = "y",
        scale: bool = True,
        label_kwargs: dict = None,
        edge_kwargs: dict = None
):
    """Plot the topology adjacency graph in 2-D.

    Args:
        geo_model ([type]): GemPy geomodel instance.
        edges (Set[Tuple[int, int]]): Set of topology edges.
        centroids (Dict[int, Array[int, 3]]): Dictionary of topology id's and
            their centroids.
        direction (Union["x", "y", "z", optional): Section direction.
            Defaults to "y".
        label_kwargs (dict, optional): Keyword arguments for topology labels.
            Defaults to None.
        edge_kwargs (dict, optional): Keyword arguments for topology edges.
            Defaults to None.
    """
    PlotSolution.plot_topo_g(
        geo_model,
        edges,
        centroids,
        direction=direction,
        scale=scale,
        label_kwargs=label_kwargs,
        edge_kwargs=edge_kwargs
    )

def plot_ar(geo_model, path=None, project_name=None, api_token=None, secret=None):
    """ Create, upload and retrieve tag to visualize the model in AR in rexview

    https://www.rexos.org/getting-started/

    Args:
        geo_model (gempy.Model):
        path: Location for rex files. Default cwd
        project_name: Name of the project in rexos
        api_token: rexos api token
        secret: rexos secret

    Returns:
        gempy.addons.rex_api.Rextag
    """
    from gempy.addons.rex_api import upload_to_rexcloud
    from gempy.addons.gempy_to_rexfile import geo_model_to_rex
    if project_name is None:
        project_name = geo_model.meta.project_name

    if path is None:
        path='./'

    files_path = geo_model_to_rex(geo_model, path)
    project_name_ = project_name
    for i in range(40):
        try:
            tag = upload_to_rexcloud(files_path, project_name=project_name_, api_token=api_token, secret=secret)
            break
        except ConnectionError:
            project_name_ = project_name+str(i)
            pass

    return tag