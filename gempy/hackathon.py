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

    if circles is not None:
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


def filter_circles(shape_coords, circle_coords):
    dist = distance.cdist(shape_coords, circle_coords, 'euclidean')
    minima = np.min(dist, axis=1)
    non_circle_pos = np.where(minima > 10)
    return non_circle_pos

def where_non_circles(image, thresh_value=80, min_area=30):
    shape_coords = where_shapes(image, thresh_value, min_area)
    circle_coords = where_circles(image, thresh_value)
    non_circles = filter_circles(shape_coords, circle_coords)
    return shape_coords[non_circles].tolist()

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


def scale_linear(data, high, low):
    mins = np.amin(data)
    maxs = np.amax(data)
    rng = maxs - mins
    return high - (((high - low) * (maxs - data)) / rng)

def smooth_topo(data, sigma_x=2, sigma_y=2):
    sigma = [sigma_y, sigma_x]
    dataSmooth = sp.ndimage.filters.gaussian_filter(data, sigma, mode='nearest')
    return dataSmooth

def simulate_seismic_topo (topo, circles, not_circles, f0 = 0.02500, dx=10, dy=10, t0=0, tn=1000,pmlthickness=40 ,slice_to_display = 200):

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

    return u.data[:,pmlthickness:-pmlthickness,pmlthickness:-pmlthickness]
