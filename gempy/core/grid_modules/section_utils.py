from gempy.plot import visualization_2d as vv
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def _extract_boundaries(self, axes, section_name='topography'):
    """
    Should be part of viz 2d.
    Args:
        self:
        axes:
        section_name:

    Returns:

    """
    cs = []

    faults = list(self.model._faults.df[self.model._faults.df['isFault'] == True].index)

    if section_name == 'topography':
        shape = self.model._grid.topography.resolution
        a = self.model.solutions.geological_map[1]
        extent = [self.model._grid.topography.extent[0],
                  self.model._grid.topography.extent[1],
                  self.model._grid.topography.extent[2],
                  self.model._grid.topography.extent[3]]
    else:
        l0, l1 = self.model._grid.sections.get_section_args(section_name)
        j = np.where(self.model._grid.sections.names == section_name)[0][0]
        shape = [self.model._grid.sections.resolution[j][0], self.model._grid.sections.resolution[j][1]]
        a = self.model.solutions.sections[1][:, l0:l1]
        # b = self.model.solutions.sections[0][:, l0:l1].reshape(shape).T
        extent = [0, self.model._grid.sections.dist[j][0], self.model._grid.regular_grid.extent[4],
                  self.model._grid.regular_grid.extent[5]]

    zorder = 2
    counter = a.shape[0]

    counters = np.arange(0, counter, 1)
    c_id = 0  # color id startpoint
    colors = []
    for f_id in counters:
        block = a[f_id]
        level = self.model.solutions.scalar_field_at_surface_points[f_id][np.where(
            self.model.solutions.scalar_field_at_surface_points[f_id] != 0)]

        levels = np.insert(level, 0, block.max())
        c_id2 = c_id + len(level)
        if f_id == counters.max():
            levels = np.insert(levels, level.shape[0], block.min())
            c_id2 = c_id + len(levels)  # color id endpoint
        if section_name == 'topography':
            block = block.reshape(shape)
        else:
            block = block.reshape(shape).T
        zorder = zorder - (f_id + len(level))

        if f_id >= len(faults):
            color = self.cmap.colors[c_id:c_id2][::-1]
            plot = axes.contourf(block, 0, levels=np.sort(levels), colors=color,
                                 linestyles='solid', origin='lower',
                                 extent=extent, zorder=zorder)
        else:
            color = self.cmap.colors[c_id:c_id2][0]
            plot = axes.contour(block, 0, levels=np.sort(levels), colors=color,
                                linestyles='solid', origin='lower',
                                extent=extent, zorder=zorder)
        c_id += len(level)
        cs.append(plot)
        if type(color) == str:
            colors.append(color)
        else:
            for c in color:
                colors.append(c)
    return cs, colors, extent


def get_polygon_dictionary(geo_model, section_name):
    """

    Args:
        geo_model: the geological model
        section_name: the section from which the polygons should be retrieved. Must be 'topography' or a predefined
        section of model.grid.sections

    Returns: [0]: pathdict. A dictionary of every surface with its corresponding polygon xy values.
             [1]: color dictionary
             [2]: extent of the section

    """
    p = vv.Plot2D(geo_model)
    p.create_figure((13, 13))
    t = p.add_section(section_name, ax_pos=111)

    cs, colors, extent = _extract_boundaries(p, p.axes[0], section_name)
    all_paths = []
    for contour in cs:
        for col in contour.collections:
            all_paths.append(col.get_paths())

    for p in all_paths:
        if p == []:
            all_paths.remove(p)

    surflist = []
    for color in colors:
        surflist.append(geo_model._surfaces.df[geo_model._surfaces.df['color'] == color]['surface'].values[0])

    pathdict = dict(zip(surflist, all_paths))
    cdict = dict(zip(surflist, colors))

    return pathdict, cdict, extent

