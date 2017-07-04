import vtk
import random
import os
from os import path
import sys
# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
from gempy.colors import *

def color_lot_create(geo_data, cd_rgb=color_dict_rgb, c_names=color_names, c_subname="400"):
    c_lot = {}
    for i, fmt in enumerate(geo_data.formations):
        c_lot[fmt] = cd_rgb[c_names[i]][c_subname]
    return c_lot


class InterfaceSphere(vtk.vtkSphereSource):
    def __init__(self, index, color=(0.5, 0.5, 0.5), fmt=None):
        self.index = index  # df index
        if fmt is None:
            self.color = color
        else:
            self.color = C_LOT[fmt]


class FoliationArrow(vtk.vtkArrowSource):
    def __init__(self, index, color=(0.5, 0.5, 0.5), fmt=None):
        self.index = index  # df index
        if fmt is None:
            self.color = color
        else:
            self.color = C_LOT[fmt]


class ColoredActor(vtk.vtkActor):
    def __init__(self, index, color=(0.5, 0.5, 0.5)):
        self.index = index
        self.color = color


class CustomTransformPolyDataFilter(vtk.vtkTransformPolyDataFilter):
    def __init__(self, index, color=(0.5, 0.5, 0.5)):
        self.index = index
        self.color = color


def visualize(geo_data,
              pot_field=None,
              surface_vals=None,
              interf_bool=True, fol_bool=True,
              verbose=0,
              win_size=(1000, 800),
              sphere_r=None,
              surface_alpha=1,
              surface_spacing_modifier=1
              ):
    """
    Args:
        geo_data: geo_data object
        pot_field: np.array
        surface_vals: list of potential field values
        interf_bool: bool
        fol_bool: bool
        surf_bool: bool
        verbose: int

    Returns:

    """
    # TODO: Fix move lock if window gets resized
    global C_LOT  # TODO: Make this more elegant, less shitty
    C_LOT = color_lot_create(geo_data)

    n_ren = 4

    # get model extent and calculate parameters for camera and sphere size
    _e = geo_data.extent  # array([ x, X,  y, Y,  z, Z])
    _e_dx = _e[1] - _e[0]
    _e_dy = _e[3] - _e[2]
    _e_dz = _e[5] - _e[4]
    _e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3
    _e_max = np.argmax(geo_data.extent)

    res = geo_data.resolution

    # create render window, settings
    renwin = vtk.vtkRenderWindow()
    renwin.SetSize(win_size[0], win_size[1])
    renwin.SetWindowName('GeMpy 3D-Editor')

    if interf_bool:
        # create interface SphereSource
        if sphere_r is None:
            sphere_r = _e_d_avrg / 50
        spheres = _create_interface_spheres(geo_data, r=sphere_r)
        # create sphere mappers and actors
        interf_mappers, interf_actors = _create_mappers_actors(spheres)

    if fol_bool:
        # create foliation ArrowSource
        arrows = _create_foliation_arrows(geo_data)
        # create arrow transformer
        arrows_transformers = _create_arrow_transformers(arrows, geo_data, _e_d_avrg / 35)
        # create arrow mappers and actors
        arrow_mappers, arrow_actors = _create_mappers_actors(arrows_transformers)

    if pot_field is not None:
        # create PolyData object for each surface
        surfaces = []
        for c, val in enumerate(surface_vals):
            vertices, simplices, normals, values = _extract_surface(pot_field,
                                                                    val,
                                                                    res,
                                                                    (res[0]/surface_spacing_modifier,
                                                                     res[1]/surface_spacing_modifier,
                                                                     res[2]/surface_spacing_modifier))
            _pf_p = vtk.vtkPoints()
            _pf_tris = vtk.vtkCellArray()
            _pf_tri = vtk.vtkTriangle()

            if verbose:
                print(vertices)
                print(np.shape(vertices))
                print(simplices)
                print(np.shape(simplices))

            for p in vertices:
                if verbose:
                    print(p)
                    print(np.shape(p))
                # scale points with resolution and extent
                # TODO: Check correctness with other models
                _p_temp = [p[0] * -(_e[0] - _e[1]) / res[0] + _e[0],
                           p[1] * -(_e[2] - _e[3]) / res[1] + _e[2],
                           p[2] * -(_e[4] - _e[5]) / res[2] + _e[4]]
                _pf_p.InsertNextPoint(_p_temp)
            for i in simplices:
                if verbose:
                    print(i)
                    print(np.shape(i))
                _pf_tri.GetPointIds().SetId(0, i[0])
                _pf_tri.GetPointIds().SetId(1, i[1])
                _pf_tri.GetPointIds().SetId(2, i[2])

                _pf_tris.InsertNextCell(_pf_tri)

            # TODO: Hand down correct iso-surface colors for each layer interface
            surfaces.append(CustomPolyData(color=C_LOT[geo_data.formations[c]],
                                           surface_alpha=surface_alpha))  # vtk.vtkPolyData
            surfaces[-1].SetPoints(_pf_p)
            surfaces[-1].SetPolys(_pf_tris)

        # create surface mappers and actors
        surface_mappers, surface_actors = _create_mappers_actors(surfaces)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # viewport dimensions setup
    xmins = [0, 0.6, 0.6, 0.6]
    xmaxs = [0.6, 1, 1, 1]
    ymins = [0, 0, 0.33, 0.66]
    ymaxs = [1, 0.33, 0.66, 1]

    # create list of renderers, set vieport values
    ren_list = []
    for i in range(n_ren):
        # append each renderer to list of renderers
        ren_list.append(vtk.vtkRenderer())
        # add each renderer to window
        renwin.AddRenderer(ren_list[-1])
        # set viewport for each renderer
        ren_list[-1].SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # create interactor and set interactor style, assign render window
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(CustomInteractorCamera(ren_list, geo_data, interactor, pot_field))
    interactor.SetRenderWindow(renwin)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # 3d model camera
    camera_list = _create_cameras(_e, verbose=verbose)
    # define background colors of the renderers
    # TODO: Tune renderer colors
    # TODO: Try to make renderer titles (floating text?!)
    ren_color = [(66 / 250, 66 / 250, 66 / 250), (0.5, 0., 0.1), (0.1, 0.5, 0.1), (0.1, 0.1, 0.5)]

    for i in range(n_ren):
        # set active camera for each renderer
        ren_list[i].SetActiveCamera(camera_list[i])
        # set background color for each renderer
        ren_list[i].SetBackground(ren_color[i][0], ren_color[i][1], ren_color[i][2])

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # create AxesActor and customize
    cube_axes_actor = _create_axes(geo_data, camera_list)

    # add actors to all renderers
    for r in ren_list:
        # add axes actor to all renderers
        r.AddActor(cube_axes_actor)
        # r.AddActor(axes_actor)
        if interf_bool:
            for a in interf_actors:
                r.AddActor(a)
        if fol_bool:
            for a in arrow_actors:
                r.AddActor(a)
        if pot_field is not None:
            for a in surface_actors:
                r.AddActor(a)

        # reset cameras for all renderers
        r.ResetCamera()

    # initialize and start the app
    interactor.Initialize()
    interactor.Start()

    # close_window(interactor)
    del renwin, interactor


def _extract_surface(pot_field, val, res, spacing):
    from skimage import measure

    vertices, simplices, normals, values = measure.marching_cubes(pot_field.reshape(res[0], res[1], res[2]),
                                                                  val  # , #0.2424792,  # -0.559606
                                                                  # spacing=spacing, # (10.0, 10.0, 10.0)
                                                                  )
    return vertices, simplices, normals, values


def _create_cameras(_e, verbose=0):
    _e_dx = _e[1] - _e[0]
    _e_dy = _e[3] - _e[2]
    _e_dz = _e[5] - _e[4]
    _e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3
    _e_max = np.argmax(_e)

    model_cam = vtk.vtkCamera()
    model_cam.SetPosition(_e[_e_max] * 5, _e[_e_max] * 5, _e[_e_max] * 5)
    model_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                            np.min(_e[2:4]) + _e_dy / 2,
                            np.min(_e[4:]) + _e_dz / 2)

    model_cam.SetViewUp(-0.239,0.155,0.958)
    #model_cam.Roll(-80.)

    # XY camera RED
    xy_cam = vtk.vtkCamera()
    # if np.argmin(_e[4:]) == 0:
    xy_cam.SetPosition(np.min(_e[0:2]) + _e_dx / 2,
                       np.min(_e[2:4]) + _e_dy / 2,
                       _e[_e_max] * 4)

    xy_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                         np.min(_e[2:4]) + _e_dy / 2,
                         np.min(_e[4:]) + _e_dz / 2)

    # YZ camera GREEN
    yz_cam = vtk.vtkCamera()
    yz_cam.SetPosition(_e[_e_max] * 4,
                       np.min(_e[2:4]) + _e_dy / 2,
                       np.min(_e[4:]) + _e_dz / 2)

    yz_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                         np.min(_e[2:4]) + _e_dy / 2,
                         np.min(_e[4:]) + _e_dz / 2)
    yz_cam.Roll(-90)

    # XZ camera BLUE
    xz_cam = vtk.vtkCamera()
    xz_cam.SetPosition(np.min(_e[0:2]) + _e_dx / 2,
                       _e[_e_max] * 4,
                       np.min(_e[4:]) + _e_dz / 2)

    xz_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                         np.min(_e[2:4]) + _e_dy / 2,
                         np.min(_e[4:]) + _e_dz / 2)
    xz_cam.SetViewUp(1, 0, 0)
    xz_cam.Roll(90)

    # camera position debugging
    if verbose == 1:
        print("RED XY:", xy_cam.GetPosition())
        print("RED FP:", xy_cam.GetFocalPoint())
        print("GREEN YZ:", yz_cam.GetPosition())
        print("GREEN FP:", yz_cam.GetFocalPoint())
        print("BLUE XZ:", xz_cam.GetPosition())
        print("BLUE FP:", xz_cam.GetFocalPoint())

    return [model_cam, xy_cam, yz_cam, xz_cam]


def _create_interface_spheres(geo_data, r=0.33):
    "Creates InterfaceSphere (vtkSphereSource) for all interface positions in dataframe."
    spheres = []
    for index, row in geo_data.interfaces.iterrows():
        spheres.append(InterfaceSphere(index, fmt=geo_data.interfaces.ix[index]["formation"]))
        spheres[-1].SetCenter(geo_data.interfaces.ix[index]["X"],
                              geo_data.interfaces.ix[index]["Y"],
                              geo_data.interfaces.ix[index]["Z"])
        spheres[-1].SetRadius(r)
    return spheres


def _create_foliation_arrows(geo_data):
    "Creates FoliationArrow (vtkArrowSource) for all foliation positions in dataframe."
    arrows = []
    for index, row in geo_data.foliations.iterrows():
        arrows.append(FoliationArrow(index, fmt=geo_data.interfaces.iloc[index]["formation"]))
    return arrows


def _create_mappers_actors(sources):
    "Creates mappers and connected actors for all given sources."
    mappers = []
    actors = []
    for s in sources:
        mappers.append(vtk.vtkPolyDataMapper())
        if type(s) == CustomPolyData:
            mappers[-1].SetInputData(s)
            actors.append(ColoredSurfaceActor(color=s.color))

            actors[-1].GetProperty().SetColor(actors[-1].color[0], actors[-1].color[1], actors[-1].color[2])
            actors[-1].GetProperty().SetOpacity(s.surface_alpha)
        else:
            mappers[-1].SetInputConnection(s.GetOutputPort())
            actors.append(ColoredActor(s.index, color=s.color))

            actors[-1].GetProperty().SetColor(actors[-1].color[0], actors[-1].color[1], actors[-1].color[2])
        actors[-1].SetMapper(mappers[-1])
    return mappers, actors


class CustomPolyData(vtk.vtkPolyData):
    def __init__(self, color=(0.5, 0.5, 0.5), surface_alpha=1):
        self.color = color
        self.surface_alpha = surface_alpha


class ColoredSurfaceActor(vtk.vtkActor):
    def __init__(self, color=(0.5, 0.5, 0.5), surface_alpha=1):
        self.color = color
        self.surface_alpha = surface_alpha


def _get_transform(startPoint, endPoint, f):
    # Compute a basis
    normalized_x = [0 for i in range(3)]
    normalized_y = [0 for i in range(3)]
    normalized_z = [0 for i in range(3)]

    # The X axis is a vector from start to end
    math = vtk.vtkMath()
    math.Subtract(endPoint, startPoint, normalized_x)
    length = math.Norm(normalized_x)
    math.Normalize(normalized_x)

    # The Z axis is an arbitrary vector cross X
    arbitrary = [0 for i in range(3)]
    arbitrary[0] = random.uniform(-10, 10)
    arbitrary[1] = random.uniform(-10, 10)
    arbitrary[2] = random.uniform(-10, 10)
    math.Cross(normalized_x, arbitrary, normalized_z)
    math.Normalize(normalized_z)

    # The Y axis is Z cross X
    math.Cross(normalized_z, normalized_x, normalized_y)
    matrix = vtk.vtkMatrix4x4()

    # Create the direction cosine matrix
    matrix.Identity()
    for i in range(3):
        matrix.SetElement(i, 0, normalized_x[i])
        matrix.SetElement(i, 1, normalized_y[i])
        matrix.SetElement(i, 2, normalized_z[i])

    # Apply the transforms
    transform = vtk.vtkTransform()
    transform.Translate(startPoint)
    transform.Concatenate(matrix)
    transform.Scale(length * f, length * f, length * f)

    return transform


def _create_arrow_transformers(arrows, geo_data, f2):
    "Creates list of arrow transformation objects."
    # grab start and end points for foliation arrows
    arrows_sp = []
    arrows_ep = []
    f = 0.75
    for arrow in arrows:
        _sp = (geo_data.foliations.ix[arrow.index]["X"] - geo_data.foliations.ix[arrow.index]["G_x"] / f,
               geo_data.foliations.ix[arrow.index]["Y"] - geo_data.foliations.ix[arrow.index]["G_x"] / f,
               geo_data.foliations.ix[arrow.index]["Z"] - geo_data.foliations.ix[arrow.index]["G_x"] / f)
        _ep = (geo_data.foliations.ix[arrow.index]["X"] + geo_data.foliations.ix[arrow.index]["G_x"] / f,
               geo_data.foliations.ix[arrow.index]["Y"] + geo_data.foliations.ix[arrow.index]["G_y"] / f,
               geo_data.foliations.ix[arrow.index]["Z"] + geo_data.foliations.ix[arrow.index]["G_z"] / f)
        arrows_sp.append(_sp)
        arrows_ep.append(_ep)

    # ///////////////////////////////////////////////////////////////
    # create transformers for ArrowSource and transform

    arrows_transformers = []
    for i, arrow in enumerate(arrows):
        arrows_transformers.append(CustomTransformPolyDataFilter(arrow.index, arrow.color))
        arrows_transformers[-1].SetTransform(_get_transform(arrows_sp[i], arrows_ep[i], f2))
        arrows_transformers[-1].SetInputConnection(arrow.GetOutputPort())

    return arrows_transformers


def _create_axes(geo_data, camera_list, verbose=0):
    "Create and returnr cubeAxesActor, settings."
    cube_axes_actor = vtk.vtkCubeAxesActor()
    cube_axes_actor.SetBounds(geo_data.extent)
    cube_axes_actor.SetCamera(camera_list[1])
    if verbose == 1:
        print(cube_axes_actor.GetAxisOrigin())

    # set axes and label colors
    cube_axes_actor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
    cube_axes_actor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)
    # font size doesn't work seem to work - maybe some override in place?
    # cubeAxesActor.GetLabelTextProperty(0).SetFontSize(10)
    cube_axes_actor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
    cube_axes_actor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)
    cube_axes_actor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
    cube_axes_actor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)

    cube_axes_actor.DrawXGridlinesOn()
    cube_axes_actor.DrawYGridlinesOn()
    cube_axes_actor.DrawZGridlinesOn()

    cube_axes_actor.XAxisMinorTickVisibilityOff()
    cube_axes_actor.YAxisMinorTickVisibilityOff()
    cube_axes_actor.ZAxisMinorTickVisibilityOff()

    cube_axes_actor.SetXTitle("X")
    cube_axes_actor.SetYTitle("Y")
    cube_axes_actor.SetZTitle("Z")

    cube_axes_actor.SetXAxisLabelVisibility(1)
    cube_axes_actor.SetYAxisLabelVisibility(1)
    cube_axes_actor.SetZAxisLabelVisibility(1)

    # only plot grid lines furthest from viewpoint
    # ensure platform compatibility for the grid line options
    if sys.platform == "win32":
        cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
    else:  # rather use elif == "linux" ? but what about other platforms
        cube_axes_actor.SetGridLineLocation(vtk.VTK_GRID_LINES_FURTHEST)

    return cube_axes_actor


def export_vtk_rectilinear(geo_data, block_lith, path=None):
    """
        export vtk
        :return:
        """

    from evtk.hl import gridToVTK

    import numpy as np

    import random as rnd

    # Dimensions

    nx, ny, nz = geo_data.resolution

    lx = geo_data.extent[0] - geo_data.extent[1]
    ly = geo_data.extent[2] - geo_data.extent[3]
    lz = geo_data.extent[4] - geo_data.extent[5]

    dx, dy, dz = lx / nx, ly / ny, lz / nz

    ncells = nx * ny * nz

    npoints = (nx + 1) * (ny + 1) * (nz + 1)

    # Coordinates
    x = np.arange(0, lx + 0.1 * dx, dx, dtype='float64')

    y = np.arange(0, ly + 0.1 * dy, dy, dtype='float64')

    z = np.arange(0, lz + 0.1 * dz, dz, dtype='float64')

    # Variables

    lith = block_lith.reshape((nx, ny, nz))
    if not path:
        path = "./Lithology_block"

    gridToVTK(path, x, y, z, cellData={"Lithology": lith})


class CustomInteractorActor(vtk.vtkInteractorStyleTrackballActor):
    """
    Modified vtkInteractorStyleTrackballActor class to accomodate for interface df modifications.
    """

    def __init__(self, ren_list, geo_data, parent):

        self.On()
        self.DebugOn()
        self.parent = parent
        self.ren_list = ren_list
        self.geo_data = geo_data
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release_event)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("KeyPressEvent", self.key_down_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)

        self.PickedActor = None
        self.PickedProducer = None

    def key_down_event(self, obj, event):
        iren = self.GetInteractor()
        if iren is None:
            return

        key = iren.GetKeyCode()
        if key == "5":  # switch to other renderer
            self.parent.SetInteractorStyle(CustomInteractorCamera(self.ren_list, self.geo_data, self.parent))

        # elif key == "d":
        #     mouse_pos = self.GetInteractor().GetEventPosition()
        #     pickers = []
        #     picked_actors = []
        #     for r in self.ren_list:
        #         pickers.append(vtk.vtkPicker())
        #         pickers[-1].Pick(mouse_pos[0], mouse_pos[1], 0, r)
        #         picked_actors.append(pickers[-1].GetActor())
        #     for pa in picked_actors:
        #         if pa is not None:
        #             _m = pa.GetMapper()
        #             _i = _m.GetInputConnection(0, 0)
        #             _p = _i.GetProducer()
        #
        #             if type(_p) is InterfaceSphere:
        #                 self.geo_data.interface_drop(_p.index)
        #
        #             elif type(_p) is FoliationArrow:
        #                 alg = _p.GetInputConnection(0, 0)
        #                 self.geo_data.foliation_drop(_p)
        #                 _p = alg.GetProducer()
        #         for r in self.ren_list:
        #             r.RemoveActor(pa)
        #             r.Render()
        # # TODO: Make point deletion work.



    def left_button_press_event(self, obj, event):
        #self.middle_button_press_event(obj, event)
        pass
        # # print("Pressed left mouse button")
        #
        # # m = vtk.vtkMatrix4x4()
        #
        # clickPos = self.GetInteractor().GetEventPosition()
        # pickers = []
        # picked_actors = []
        # for r in self.ren_list:
        #     pickers.append(vtk.vtkPicker())
        #     pickers[-1].Pick(clickPos[0], clickPos[1], 0, r)
        #     picked_actors.append(pickers[-1].GetActor())
        # for pa in picked_actors:
        #     if pa is not None:
        #         self.PickedActor = pa
        #
        # # TODO: Arrow Rotation -> modify foliation dataframe
        # # vtk.vtkOpenGLActor.GetOrientation?
        # # matrix = self.PickedActor.GetMatrix(m)
        # # if self.PickedActor is
        # # self.PickedActor.SetScale(2)
        # # renwin.Render()
        # #try:
        # #    orientation = self.PickedActor.GetOrientation()
        # #    print(str(orientation))
        # #except AttributeError:
        # #    pass
        #
        # self.OnLeftButtonDown()

    def left_button_release_event(self, obj, event):
        #self.middle_button_release_event(obj, event)
        pass
        # # matrix = self.PickedActor.GetMatrix(vtk.vtkMatrix4x4())
        # try:
        #     matrix = self.PickedActor.GetOrientation()
        #     # print(str(matrix))
        # except AttributeError:
        #     pass
        # self.OnLeftButtonUp()

    def middle_button_press_event(self, obj, event):
        # get event position of click event

        self.PickedProducer = None
        self.PickedActor = None


        clickPos = self.GetInteractor().GetEventPosition()

        pickers = []
        picked_actors = []
        # go through every renderer and pick
        for r in self.ren_list:
            pickers.append(vtk.vtkPicker())
            pickers[-1].Pick(clickPos[0], clickPos[1], 0, r)
            picked_actors.append(pickers[-1].GetActor())

        # select the actual actor picked (if exists)
        for pa in picked_actors:
            if pa is not None:
                self.PickedActor = pa
                #self.PickedActor.Update()

        if self.PickedActor is not None:
            _m = self.PickedActor.GetMapper()
            _i = _m.GetInputConnection(0, 0)
            _p = _i.GetProducer()

            #print(type(_p))

        if type(_p) is not InterfaceSphere and type(_p) is not CustomTransformPolyDataFilter and _p is not None:
            # then go deeper
            alg = _p.GetInputConnection(0, 0)
            self.PickedProducer = alg.GetProducer()

        else:
            self.PickedProducer = _p

            if self.PickedProducer is not None:
                # print("Moving actor: ",type(self.PickedActor))
                # print("Producer: ",type(self.PickedProducer))
                if type(self.PickedProducer) is InterfaceSphere:
                    _c = self.PickedActor.GetCenter()
                    self.geo_data.interface_modify(self.PickedProducer.index, X=_c[0], Y=_c[1], Z=_c[2])
                elif type(self.PickedProducer) is CustomTransformPolyDataFilter:
                    _c = self.PickedActor.GetCenter()
                    self.geo_data.foliation_modify(self.PickedProducer.index, X=_c[0], Y=_c[1], Z=_c[2])


            # print(str(type(self.PickedProducer)))
        self.OnMiddleButtonDown()
        return

    def mouse_move_event(self, obj, event):
        self.OnMouseMove()

    def middle_button_release_event(self, obj, event):
        # TODO: disable moving surfaces

        if self.PickedProducer is not None:
            #print("Moving actor: ",type(self.PickedActor))
            #print("Producer: ",type(self.PickedProducer))
            if type(self.PickedProducer) is InterfaceSphere:
                # if self.PickedActor.GetPosition() == (0.0, 0.0, 0.0):
                #     pr = 'The point ' + str(self.PickedProducer.index) + ' did not move. Click on it again'
                # else:
                #     pr = 'The point ' + str(self.PickedProducer.index) + ' moved.'
                # print("MiddleMouseButton released. ", pr)
                _c = self.PickedActor.GetCenter()
                self.geo_data.interface_modify(self.PickedProducer.index, X=_c[0], Y=_c[1], Z=_c[2])
            elif type(self.PickedProducer) is CustomTransformPolyDataFilter:
                _c = self.PickedActor.GetCenter()
                self.geo_data.foliation_modify(self.PickedProducer.index, X=_c[0], Y=_c[1], Z=_c[2])

        self.PickedProducer = None
        self.PickedActor = None

        self.OnMiddleButtonUp()
        return

    def na(self):
        print("MiddleMouseButton released")

        if self.PickedProducer is not None:
            # print("Moving actor: ",type(self.PickedActor))
            # print("Producer: ",type(self.PickedProducer))

            if type(self.PickedProducer) is InterfaceSphere:
                _c = self.PickedActor.GetCenter()
                self.geo_data.interface_modify(self.PickedProducer.index, X=_c[0], Y=_c[1], Z=_c[2])
            elif type(self.PickedProducer) is CustomTransformPolyDataFilter:
                _c = self.PickedActor.GetCenter()
                self.geo_data.foliation_modify(self.PickedProducer.index, X=_c[0], Y=_c[1], Z=_c[2])

        self.PickedProducer = None
        self.PickedActor = None


class CustomInteractorCamera(vtk.vtkInteractorStyleTrackballCamera):
    """
    Custom camera interactor class.
    """

    def __init__(self, ren_list, geo_data, parent, pot_field):
        self.parent = parent
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.AddObserver("KeyPressEvent", self.key_down_event)

        self.renwin = self.parent.GetRenderWindow()

        self.ren_list = ren_list
        self.geo_data = geo_data
        self.prev_mouse_pos = None

        self.left_button_hold = False
        self.pot_field = pot_field

    def key_down_event(self, obj, ev):
        iren = self.GetInteractor()
        if iren is None:
            return

        key = iren.GetKeyCode()
        if key == "5":  # switch to other renderer
            if self.pot_field is None:
                self.parent.SetInteractorStyle(CustomInteractorActor(self.ren_list, self.geo_data,
                                                                     self.parent))
        # elif key =="6":
        #     print("viewup:",self.ren_list[0].GetActiveCamera().GetViewUp())
        #     print("roll:",self.ren_list[0].GetActiveCamera().GetRoll())
        #elif key == "7":
        #    self.renwin.Finalize()
        #    self.parent.TerminateApp()
        #    del self.renwin, self.parent
        #    # vtk.vtkRenderWindowInteractor

    def left_button_press_event(self, obj, ev):
        if self.renwin is not None:
            self.renwin_size = self.renwin.GetSize()
        else:
            self.renwin_size = (1000, 800)

        self.left_button_hold = True
        click_pos = self.GetInteractor().GetEventPosition()
        # self.parent.SetCurrentRenderer(self.ren_list[0])
        try:
            if click_pos[0] < self.renwin_size[0] * 0.66:  # self.renwin.GetSize()[0]*0.66:
                self.OnLeftButtonDown()
            else:
                pass
        except AttributeError:
            pass

    def left_button_release_event(self, obj, ev):
        self.left_button_hold = False
        self.OnLeftButtonUp()

    def mouse_move_event(self, obj, ev):
        mouse_pos = self.GetInteractor().GetEventPosition()

        if self.renwin is not None:
            self.renwin_size = self.renwin.GetSize()
        else:
            self.renwin_size = (1000, 800)

        if self.prev_mouse_pos is not None:
            dx = mouse_pos[0] - self.prev_mouse_pos[0]
            if mouse_pos[0] + dx >= self.renwin_size[0] * 0.66:  # self.renwin.GetSize()[0]*0.66:
                self.left_button_release_event(obj, ev)
            else:
                self.OnMouseMove()
        self.prev_mouse_pos = mouse_pos
