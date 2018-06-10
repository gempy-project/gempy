"""
This is the hackathon python file!
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import scipy as sp
import scipy.ndimage
from matplotlib import cm
try:
    from examples.seismic import Model, plot_velocity
    from devito import TimeFunction
    from devito import Eq
    from sympy import solve
    from examples.seismic import RickerSource
    from devito import Operator
except ImportError:
    print('Devito is not working')

### LEGO/SHAPE RECOGNITION
def where_shapes(image, thresh_value=80, min_area=30):
    """Get the coordinates for all detected shapes.

            Args:
                image (image file): Image input.
                min_area (int, float): Minimal area for a shape to be detected.
            Returns:
                x- and y- coordinates for all detected shapes as a 2D array.

        """
    bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
    gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)[1]
    edge_detected_image = cv2.Canny(thresh, 75, 200)

    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    contour_coords = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = 0
            cY = 0

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (len(approx) < 23) & (area > min_area)):
            contour_list.append(contour)
            contour_coords.append([cX, cY])

    return np.array(contour_coords)

def read_image(image):
    img=cv2.imread(image)
    return img

def where_circles(image, thresh_value=80):
    """Get the coordinates for all detected circles.

                Args:
                    image (image file): Image input.
                    thresh_value (int, optional, default = 80): Define the lower threshold value for shape recognition.
                Returns:
                    x- and y- coordinates for all detected circles as a 2D array.

            """
    #output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)[1]
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2 100)
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 2, np.array([]), 200, 8, 4, 8)


    if len(circles) >0:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # print(circles)
        circle_coords = np.array(circles)[:, :2]
        dist = distance.cdist(circle_coords, circle_coords, 'euclidean')
        #minima = np.min(dist, axis=1)
        dist_bool = (dist > 0) & (dist < 5)
        pos = np.where(dist_bool == True)[0]
        grouped = circle_coords[pos]
        mean_grouped = (np.sum(grouped, axis=0) / 2).astype(int)
        circle_coords = np.delete(circle_coords, list(pos), axis=0)
        circle_coords = np.vstack((circle_coords, mean_grouped))

        return circle_coords.tolist()
    else:
        return []



def filter_circles(shape_coords, circle_coords):
    dist = distance.cdist(shape_coords, circle_coords, 'euclidean')
    minima = np.min(dist, axis=1)
    non_circle_pos = np.where(minima > 10)
    return non_circle_pos

def where_non_circles(image, thresh_value=80, min_area=30):
    shape_coords = where_shapes(image, thresh_value, min_area)
    circle_coords = where_circles(image, thresh_value)
    if len(circle_coords)>0:
        non_circles = filter_circles(shape_coords, circle_coords)
        return shape_coords[non_circles].tolist()
    else:
        return shape_coords.tolist()

def get_shape_coords(image, thresh_value=80, min_area=30):
    """Get the coordinates for all shapes, classified as circles and non-circles.

                    Args:
                        image (image file): Image input.
                        thresh_value (int, optional, default = 80): Define the lower threshold value for shape recognition.
                        min_area (int, float): Minimal area for a non-circle shape to be detected.
                    Returns:
                        x- and y- coordinates for all detected shapes as 2D arrays.
                        [0]: non-circle shapes
                        [1]: circle shapes

                """
    non_circles = where_non_circles(image, thresh_value, min_area)
    circles = where_circles(image, thresh_value)

    return non_circles, circles


def plot_all_shapes(image, thresh_value=80, min_area=30):
    """Plot detected shapes onto image.

                        Args:
                            image (image file): Image input.
                            thresh_value (int, optional, default = 80): Define the lower threshold value for shape recognition.
                            min_area (int, float): Minimal area for a non-circle shape to be detected.

                    """
    output = image.copy()
    non_circles, circles = get_shape_coords(image, thresh_value, min_area)
    for (x, y) in circles:
        cv2.circle(output, (x, y), 5, (0, 255, 0), 3)
        # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    for (x, y) in non_circles:
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    out_image = np.hstack([image, output])
    plt.imshow(out_image)


def draw_markers(image,coords):
    for point in coords:
        cv2.circle(image,tuple(point), 6, (255,255,255), -1)
    return image

def draw_line(image, coords): #takes list of exactly 2 coordinate pairs
    lineThickness = 2
    cv2.line(image, tuple(coords[0]), tuple(coords[1]),(255,255,255), lineThickness)
    return image




def scale_linear(data, high, low):
    mins = np.amin(data)
    maxs = np.amax(data)
    rng = maxs - mins
    return high - (((high - low) * (maxs - data)) / rng)

def smooth_topo(data, sigma_x=2, sigma_y=2):
    sigma = [sigma_y, sigma_x]
    dataSmooth = sp.ndimage.filters.gaussian_filter(data, sigma, mode='nearest')
    return dataSmooth

def simulate_seismic_topo (topo, circles_list, not_circles, f0 = 0.02500, dx=10, dy=10, t0=0, tn=1000,pmlthickness=40 ,slice_to_display = 200):

    circles = np.array(circles_list)

    topo = topo.astype(np.float32)
    topoRescale = scale_linear(topo, 5, 1)
    veltopo=smooth_topo( topoRescale )

    # Define the model
    model = Model(vp=veltopo,        # A velocity model.
                  origin=(0, 0),     # Top left corner.
                  shape=veltopo.shape,    # Number of grid points.
                  spacing=(dx, dy),  # Grid spacing in m.
                  nbpml=pmlthickness)          # boundary layer.

    dt = model.critical_dt  # Time step from model grid spacing
    nt = int(1 + (tn-t0) / dt)  # Discrete time axis length
    time = np.linspace(t0, tn, nt)  # Discrete modelling time

    u = TimeFunction(name="u", grid=model.grid,
                 time_order=2, space_order=2,
                 save=True, time_dim=nt)
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward)[0])


    src_coords = np.multiply(circles[0],[dx,dy])
    src = RickerSource(name='src0', grid=model.grid, f0=f0, time=time, coordinates=src_coords)
    src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m, offset=model.nbpml)

    if circles.shape[0]>1:
        for idx, row in enumerate(circles[1:,:]):
            namesrc = 'src' + str(idx+1)
            src_coords = np.multiply(row,[dx,dy])
            src_temp = RickerSource(name=namesrc, grid=model.grid, f0=f0, time=time, coordinates=src_coords)
            src_term_temp = src_temp.inject(field=u.forward, expr=src * dt**2 / model.m, offset=model.nbpml)
            src_term += src_term_temp

    op_fwd = Operator( [stencil] + src_term )
    op_fwd(time=nt, dt=model.critical_dt)

    wf_data = u.data[:,pmlthickness:-pmlthickness,pmlthickness:-pmlthickness]
    wf_data_normalize = wf_data/np.amax(wf_data)
    return wf_data_normalize


def get_arbitrary_2d_grid(geo_data, px, py, s):
    """Creates arbitrary 2d grid given two input points.

    Args:
        geo_data: gempy geo_Data object
        px (list): x coordinates of the two input points (e.g. [0, 2000])
        py (list: y coordinates of the two input points (e.g. [0, 2000])
        s (int): pixel/voxel edge length

    Returns:
        np.ndarray: grid (n, 3) for use with gempy.compute_model_at function
        tuple: shape information to reshape into 2d array

    """
    px = np.array(px)
    py = np.array(py)

    from scipy.stats import linregress
    gradient, *_ = linregress(px, py)

    theta = np.arctan(gradient)
    dy = np.sin(theta) * s
    dx = np.cos(theta) * s

    if px[1] - px[0] == 0:
        ys = np.arange(py[0], py[1], s)
        xs = np.repeat(px[0], len(ys))
    elif py[1] - py[0] == 0:
        xs = np.arange(px[0], px[1], s)
        ys = np.repeat(py[0], len(xs))
    else:
        xs = np.arange(px[0], px[1], dx)
        ys = np.arange(py[0], py[1], dy)

    zs = np.arange(geo_data.extent[4], geo_data.extent[5] + 1, s)

    a = np.tile([xs, ys], len(zs)).T
    b = np.repeat(zs, len(xs))
    grid = np.concatenate((a, b[:, np.newaxis]), axis=1)

    return grid, (len(zs), len(ys))


def DRILLLL(geo_data, x, y, s=1):
    """Creates 1d vertical grid at given x,y location.

    Args:
        geo_data: gempy geo_data object
        x (int): x coordinate of the drill location
        y (int): y coordinate of the drill location
        s (int, optional): pixel/voxel edge length (default: 1)

    Returns:
        np.ndarray: grid (n, 3) for use with gempy.compute_model_at

    """
    zs = np.arange(geo_data.extent[4], geo_data.extent[5], s)
    grid = np.array([np.repeat(x, len(zs)), np.repeat(y, len(zs)), zs]).T
    return grid
