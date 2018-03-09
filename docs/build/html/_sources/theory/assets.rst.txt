ASSETS – Model analysis and further use
=======================================

In this second half of the paper we will explore different features that
complement and expand the construction of the geological model itself.
These extensions are just some examples of how *GemPy* can be used as
geological modeling engine for diverse research projects. The numerous
libraries in the open-source ecosystem allow to choose the best narrow
purpose tool for very specific tasks. Considering the visualization of
*GemPy*, for instance: *matplotlib*
:cite:`c-hunter2007matplotlib` for 2-D visualization, *vtk*
for fast and interactive 3-D visualization, *steno3D* for sharing block
models visualizations online—or even the open-source 3-D modeling
software Blender :cite:`c-blender` for creating high quality
renderings and Virtual Reality, are only some examples of the
flexibility that the combination of *GemPy* with other open-source
packages offers. In the same fashion we can use the geological model as
basis for the subsequent geophysical simulations and process
simulations. Due to Python’s modularity, combining distinct modules to
extend the scope of a project to include the geological modeling process
into a specific environment is effortless. In the next sections we will
dive into some of the built-in functionality implemented to date on top
of the geological modeling core. Current assets are: (i) 2-D and 3-D
visualizations, (ii) forward calculation of gravity, (iii) topology
analysis, (iv) uncertainty quantification (UQ) as well as (v) full
Bayesian inference.

Visualization
-------------

The segmentation of meaningful units is the central task of geological
modelling. It is often a prerequisite for engineering projects or
process simulations. An intuitive 3-D visualization of a geological
model is therefore a fundamntal requirement.

.. figure:: figs/vtkFault.png
   :alt:

   In-built *vtk* 3-D visualization of *GemPy* provides an interactive
   visualization of the geological model (left) and three additional
   orthogonal viewpoints (right) from different directions.

For its data and model visualization, *GemPy* makes use of freely
available tools in the Python module ecosystem to allow the user to
inspect data and modeling results from all possible angles. The
fundamental plotting library *matplotlib*
:cite:`c-hunter2007matplotlib`, enhanced by the statistical
data visualization library *seaborn*
:cite:`c-michael_waskom_2017_883859`, provides the 2-D
graphical interface to visualize input data and 2-D sections of scalar
fields and geological models. In addition, making use of the capacities
of *pyqt* implemented with *matplotlib*, we can generate interactive
sequence piles, where the user can not only visualize the temporal
relation of the different unconformities and faulting events, but also
modify it using intuitive drag and drop functionality (see figure
[fig:vtk]).

On top of these features, *GemPy* offers in-built 3-D visualization
based on the the open-source Visualization Toolkit
:cite:`c-schroeder2004visualization`. It provides
users with an interactive 3-D view of the geological model, as well as
three additional orthogonal viewpoints (see figure [fig:vtk]). The user
can decide to plot just the data, the geological surfaces, or both. In
addition to just visualizing the data in 3-D, *GemPy* makes use of the
interaction capabilities provided by *vtk* to allow the user to move
input data points on the fly via drag-and-drop. Combined with
*GemPy*\ ’s optimized modeling process (and the ability to use GPUs for
efficient model calculation), this feature allows for data modification
with real-time updating of the geological model (in the order of
milliseconds per scalar field). This functionality can not only improve
the understanding of the model but can also help the user to obtain the
desired outcome by working directly in 3-D space while getting direct
visual feedback on the modeling results. Yet, due to the exponential
increase of computational time with respect to the number of input data
and the model resolution), very large and complex models may have
difficulties to render fast enough to perceive continuity on
conventional computer systems.

For additional high quality visualization, we can generate vtk files
using *pyevtk*. These files can later be loaded into external VTK viewer
as Paraview :cite:`c-ayachit2015paraview` in order to take
advantage of its intuitive interface and powerful visualization options.
Another natural compatibility exists with Blender
:cite:`c-blender` due to its use of Python as front-end.
Using the Python distribution shipped within a Blender installation, it
is possible to import, run and automatically represent *GemPy*\ ’s data
and results (figure [fig:examples], see appendix [blender] for code
extension). This not only allow to render high quality images and videos
but also to visualize the models in Virtual Reality, making use of the
Blender Game engine and some of the plug-ins that enable this
functionality.

For sharing models, *GemPy* also includes functionality to upload
discretized models to the Steno 3D platform (a freemium business model).
Here, it is possible to visualize manipulate and shared the model with
any number of people effortless by simple invitations or the
distribution of a link.

In short, *Gempy* is not limited to a unique visualization library.
Currently *Gempy* gives support to many of the available visualization
options to fulfill the different needs of the developers accordingly.
However, these are not by all means the only possible alternatives and
in the future we expect that *GemPy* to be employed as backend of other
further projects.

Gravity forward modeling
------------------------

In recent years gravity measurements has increased in quality
:cite:`c-nabighian200575` and is by now a valuable additional
geophysical data source to support geological modeling. There are
different ways to include the new information into the modeling
workflow, and one of the most common is via inversions
:cite:`c-Tarantola:2005wd`. Geophysics can validate the
quality of the model in a probabilistic or optimization framework but
also by back-propagating information, geophysics can improve
automatically the modeling process itself. As a drawback, simulating
forward geophysics adds a significant computational cost and increases
the uncertainty to the parametrization of the model. However, due to the
amount of uncorrelated information—often continuous in space—the
inclusion of geophysical data in the modeling process usually becomes
significant to evaluate the quality of a given model.

.. figure:: figs/gravity.png
   :alt:

   Forward gravity response overlayed on top of a XY cross section of
   the lithology block.

*GemPy* includes built-in functionality to compute forward gravity
conserving the automatic differentiation of the package. It is
calculated from the discretized block model applying the method of
:cite:`c-nagy1966gravitational` for rectangular prisms in the
z direction,

.. math:: F_z = G_\rho|||x \ln(y+r) + y\ln(x+r)-z \arctan \left( \frac{x y}{z  r} \right) |^{x_2}_{x_1}|^{y_2}_{y_1}|^{z_2}_{z_1}


where :math:`x`, :math:`y`, and :math:`z` are the Cartesian components
from the measuring point of the prism, :math:`r` the euclidean distance
and :math:`G_\rho` the average gravity pull of the prism. This
integration provides the gravitational pull of every voxel for a given
density and distance in the component :math:`z`. Taking advantage of the
immutability of the involved parameters with the exception of density
allow us to precompute the decomposition of :math:`t_z`, leaving just
its product with the weight :math:`G_\rho`

.. math::

   F_z = G_\rho \cdot t_z
   \label{eq:grav}


as a recurrent operation.

As an example, we show here the forward gravity response of the
geological model in figure [fig:model\_comp]c. The first important
detail is the increased extent of the interpolated model to avoid
boundary errors. In general, a padding equal to the maximum distance
used to compute the forward gravity computation would be the ideal
value. In this example (figure [fig:gravity]) we l add
10\ :math:`\; \text{km}` to the X and Y coordinates. The next step is to
define the measurement 2-D grid—i.e. where to simulate the gravity
response and the densities of each layers. The densities chosen are:
2.92, 3.1, 2.61 and 2.92\ :math:`\; \text{kg/m^3}` for the basement,
“Unconformity” layer (i.e. the layer on top of the unconformity), Layer
1 and Layer 2 respectively.

::

    import matplotlib.pyplot as plt
    import gempy as gp

    # Main data management object containing. The extent must be large enough respect the forward gravity plane to account the effect of all cells at a given distance, $d$ to any spatial direction $x, y, z$.
    geo_data = gp.create_data(extent=[-10,30,-10,20,-10,0],
                  resolution=[50,50,50],
                  path_o = "paper_Foliations.csv",
                  path_i = "paper_Points.csv")

    # Defining the series of the sequential pile
    gp.set_series(geo_data, series_distribution={'fault_serie1': 'fault1',
                                 'younger_serie' : 'Unconformity',
                             'older_serie': ('Layer1', 'Layer2')},
              order_formations= ['fault1', 'Unconformity', 'Layer2', 'Layer1'])

    # Creating object with data prepared for interpolation and compiling.
    interp_data = gp.InterpolatorData(geo_data, output='gravity')

    # Setting the 2D grid of the airborn where we want to compute the forward gravity
    gp.set_geophysics_obj(interp_data_g,  ai_extent = [0, 20, 0, 10, -10, 0],
                  ai_resolution = [30,10])

    # Making all possible precomputations: Decomposing the value tz for every point of the 2D grid to each voxel
    gp.precomputations_gravity(interp_data_g, 25, densities=[2.92, 3.1, 2.61, 2.92])

    # Computing gravity (Eq. 10)
    lith, fault, grav = gp.compute_model(interp_data_g, 'gravity')

    # Plotting lithology section
    gp.plot_section(geo_data, lith[0], 0, direction='z',plot_data=True)

    # Plotting forward gravity
    plt.imshow(grav.reshape(10,30), cmap='viridis', origin='lower', alpha=0.8, extent=[0,20,0,10])

The computation of forward gravity is a required step towards a fully
coupled gravity inversion. Embedding this step into a Bayesian inference
allows to condition the initial data used to create the model to the
final gravity response. This idea will be further developed in Section
[sec:geol-invers-prob].

Topology
--------

The concept of topology provides a useful tool to describe adjacency
relations in geomodels, such as stratigraphic contacts or across-fault
connectivity
:cite:`c-Thiele:2016wx, Thiele:2016vg`.
*GemPy* has in-built functionality to analyze the adjacency topology of
its generated models as Region Adjacency Graphs (RAGs), using the
``topology_compute`` method (see Listing 6). It can be directly
visualized on top of model sections (see figure [fig:topology]), where
each unique topological region in the geomodel is represented by a graph
node, and each connection as a graph edge. The function outputs the
graph object G, the region centroid coordinates, a list of all the
unique node labels, and two look-up tables to conveniently reference
node labels and lithologies

.. figure:: figs/topology.png
   :alt:

   Section of the example geomodel with overlaid topology graph. The
   geomodel contains eight unique regions (graph nodes) and 13 unique
   connections (graph edges). White edges represent stratigraphic and
   unconformity connections, while black edges correspond to
   across-fault connections.

To analyze the model topology, *GemPy* makes use of a general connected
component labeling (CCL) algorithm to uniquely label all separated
geological entities in 3-D geomodels. The algorithm is provided via the
widely used, open-source, Python-based image processing library
*scikit-image* :cite:`c-van2014scikit` by the function
``skimage.measure.label``, which is based on the optimized algorithms of
:cite:`c-fiorio1996two`. But just using CCL
on a 3-D geomodel fails to discriminate a layer cut by a fault into two
unique regions because in practice both sides of a fault are represented
by the same label. To achieve the detection of edges across the fault,
we need to precondition the 3-D geomodel matrix, which contains just the
lithology information (layer id), with a 3-D matrix containing the
information about the faults (fault block id). This yields a 3-D matrix
which combines the lithology information and the fault block
information. This matrix can then be successfully labeled using CCL with
a 2-connectivity stamp, resulting in a new matrix of uniquely labeled
regions for the geomodel. From these, an adjacency graph is generated
using ``skimage.future.graph.RAG``, which created a Region Adjacency
Graph (RAG) of all unique regions contained in a 2-D or 3-D matrix,
representing each region with a node and their adjacency relations as
edges, successfully capturing the topology information of our geomodel.
The connections (edges) are then further classified into either
stratigraphic or across-fault edges, to provide further information. If
the argument ``compute_areas=True`` was given, the contact area for the
two regions of an edge is automatically calculated (number of voxels)
and stored inside the adjacency graph.

::

    ...
    Add Listing 3
    ...

    # Computing result
    lith, fault = gp.compute_model(interp_data)

    # Compute topology
    G, centroids, labels_unique, labels_lot, lith_lot = gp.topology_compute(geo_data, lith[0], fault[0], compute_areas=True)

    # Plotting topology network
    gp.plot_section(geo_data, lith[0], 5)
    gp.topology_plot(geo_data, G, centroids)

Stochastic Geomodeling and probabilistic programming
----------------------------------------------------

Raw geological data is noisy and measurements are usually sparse. As a
result, geological models contain significant uncertainties
:cite:`c-Wellmann:2010bz,Bardossy.2004,Lark:2013cj,Caers:2011jr,McLane:2008wz,Chatfield.1995`
that must be addressed thoughtfully to reach a plausible level of
confidence in the model. However, treating geological modeling
stochastically implies many considerations: (i) from tens or hundreds of
variables involved in the mathematical equations which ones should be
latent?; (ii) can we filter all the possible outcomes which represent
unreasonable geological settings? and (iii) how can we use other sources
of data—especially geophysics—to improve the accuracy of the inference
itself?

The answers to these questions are still actively debated in research
and are highly dependent on the type of mathematical and computational
framework chosen. In the interpolation method explained in this paper,
the parameters suitable to behave as a latent variables (see figure
[fig:overall] for an overview of possible stochastic parameters) could
be the interface points :math:`{\bf{x}}_{\alpha}` (i.e. the 3 Cartesian
coordinates :math:`x, \; y, \; z`), orientations
:math:`{\bf{x}}_{\beta}` (i.e. the 3 Cartesian coordinates
:math:`x, \; y, \; z` and the plane orientation normal
:math:`Gx, \; Gy, \; Gz`) or densities for the computation of the
forward gravity. But not only parameters with physical meaning are
suitable to be considered stochastic. Many mathematical parameters used
in the kriging interpolation—such as: covariance at distance zero
:math:`C_0` (i.e. nugget effect) or the range of the covariance
:math:`r` (see Appendix
[covariance-function-cubic.-discuss-it-with-france] for an example of a
covariance function)—play a crucial role during the computation of the
final models and, at best, are inferred by an educated guess to a
greater or lesser extent
:cite:`c-Chiles.2004,Calcagno.2008`. To tackle this problem
in a strict manner, it would be necessary to combine Bayesian
statistics, information theory and sensitivity analysis among other
expertises, but in essence all these methodologies begin with a
probabilistic programming framework.

*GemPy* is fully designed to be coupled with probabilistic frameworks,
in particular with *pymc3* :cite:`c-Salvatier:2016ki` as both
libraries are based on *Theano*.

*pymc* is a series of Python libraries that provide intuitive tools to
build and subsequently infer complex probabilistic graphical models
:cite:`c-Koller:2009wk`.
These libraries offer expressive and clean syntax to write and use
statistical distributions and different samplers. At the moment two main
libraries coexist due to their different strengths and weaknesses. On
the one hand, we have *pymc2* :cite:`c-Patil10pymc:bayesian`
written in FORTRAN and Python. *pymc2* does not allow gradient based
sampling methods, since it does not have automatic differentiation
capabilities. However, for that same reason, the model construction and
debugging is more accessible. Furthermore, not computing gradients
enables an easy integration with 3rd party libraries and easy
extensibility to other scientific libraries and languages. Therefore,
for prototyping and lower dimensionality problems—where the posterior
can be tracked by Metropolis-Hasting methods
:cite:`c-Haario:2001tg`–*pymc2* is still the go-to choice.

On the other hand the latest version, *pymc3*
:cite:`c-Salvatier:2016ki`, allows the use of next generation
gradient-based samplers such as No-U-Turn Sampler
 or Automatic Variational Inference
:cite:`c-kucukelbir2015automatic`. These sampling methods are
proving to be a powerful tool to deal with multidimensional
problems—i.e. models with high number of uncertain parameters
:cite:`c-betancourt2017geometric`. The weakness of these
methods are that they rely on the computation of gradients, which in
many cases cannot be manually derived. To circumvent this limitation
*pymc3* makes use of the AD capabilities of *Theano*. Being built on top
of *Theano* confer to the Bayesian inference process all the
capabilities discussed in section [theano] in exchange for the clarity
and flexibility that pure Python provides.

In this context, the purpose of *GemPy* is to fill the gap of complex
algebra between the prior data and observations, such as geophysical
responses (e.g. gravity or seismic inversions) or geological
interpretations (e.g. tectonics, model topologies). Since *GemPy* is
built on top of *Theano* as well, the compatibility with both libraries
is relatively straightforward. However, being able to encode most of the
conceivable probabilistic graphical models derived from, often, diverse
and heterogeneous data would be an herculean task. For this reason most
of the construction of the PGM has to be coded by the user using the
building blocks that the *pymc* packages offer (see listing 6). By doing
so, we can guarantee full flexibility and adaptability to the
necessities of every individual geological setting.

For this paper we will use *pymc2* for its higher readability and
simplicity. *pymc3* architecture is analogous with the major difference
that the PGM is constructed in *Theano*—and therefore symbolically (for
examples using *pymc3* and *GemPy* check the online documention detailed
in Appendix [sec:documentation]).

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

An essential aspect of probabilistic programming is the inherent
capability to quantify uncertainty. Monte Carlo error propagation
:cite:`c-ogilvie1984monte` has been introduced in the field
of geological modeling a few years ago
:cite:`c-Wellmann:2010bz,Jessell.2010,Lindsay:2012gx`,
exploiting the automation of the model construction that implicit
algorithms offer.

In this paper example (figure [fig:uncertainty]-Priors), we fit a normal
distribution of standard deviation :math:`300 \,`\ [m] around the Z axis
of the interface points in initial model (figure [fig:model\_comp] c).
In other words, we allows to the interface points that define the model
to oscillate independently along the axis Z accordingly randomly—using
normal distributions—and subsequently we compute the geomodels that
these new data describe.

The first step to the creation of a PGM is to define the parameters that
are supposed to be stochastic and the probability functions that
describe them. To do so, *pymc2* provides a large selection of
distributions as well as a clear framework to create custom ones. Once
we created the stochastic parameters we need to substitute the initial
value in the *GemPy* database (``interp_data`` in the snippets) for the
corresponding *pymc2* objects. Next, we just need to follow the usual
*GemPy* construction process—i.e. calling the ``compute_model``
function—wrapping it using a deterministic *pymc2* decorator to describe
that these function is part of the probabilistic model (figure
[fig:PGM-prior]). After creating the graphical model we can sample from
the stochastic parameters using Monte Carlo sampling using *pymc2*
methods.

::

    ...
    Add Listing 3
    ...

    # Coping the initial data
    geo_data_stoch_init = deepcopy(interp_data.geo_data_res)
    # MODEL CONSTRUCTION
    # ==================
    # Positions (rows) of the data we want to make stochastic
    ids = range(2,12)

    # List with the stochastic parameters. pymc.Normal attributes: Name, mean, std
    interface_Z_modifier = [pymc.Normal("interface_Z_mod_"+str(i), 0., 1./0.01**2) for i in ids]

    # Modifing the input data at each iteration
    @pymc.deterministic(trace=True)
    def input_data(value = 0, interface_Z_modifier = interface_Z_modifier,
               geo_data_stoch_init = geo_data_stoch_init,
               ids = ids, verbose=0):

    # First we extract from our original intep_data object the numerical data that
    # is necessary for the interpolation. geo_data_stoch is a pandas Dataframe
        geo_data_stoch = gp.get_data(geo_data_stoch_init, numeric=True)

    # Now we loop each id which share the same uncertainty variable. In this case, each layer.  We add the stochastic part to the initial value
        for num, i in enumerate(ids):
            interp_data.geo_data_res.interfaces.set_value(i, "Z",
              geo_data_stoch_init.interfaces.iloc[i]["Z"] + interface_Z_modifier[num])

    # Return the input data to be input into the modeling function. Due to the way pymc2
    # stores the traces we need to save the data as numpy arrays
        return interp_data.geo_data_res.interfaces[["X", "Y", "Z"]].values,
               interp_data.geo_data_res.orientations[["X", "Y", "Z", "dip", "azimuth",          "polarity"]].values

    # Computing the geological model
    @pymc.deterministic(trace=True)
    def gempy_model(value=0, input_data=input_data, verbose=False):

    # modify input data values accordingly
        interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = input_data[0]

    # Gx, Gy, Gz are just used for visualization. The Theano function gets azimuth dip and polarity!!!
        interp_data.geo_data_res.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z", 'dip',         'azimuth', 'polarity']] = input_data[1]

    # Some iterations will give a singular matrix, that's why we need to
    # create a try to not break the code.
        try:
            lb, fb, grav = gp.compute_model(interp_data, outup='gravity')
            return lb, fb, grav

        except np.linalg.linalg.LinAlgError as err:
    # If it fails (e.g. some input data combinations could lead to
    # a singular matrix and thus break the chain) return an empty model
    # with same dimensions (just zeros)
            if verbose:
                print("Exception occured.")
            return np.zeros_like(lith_block), np.zeros_like(fault_block), np.zeros_like(grav_i)

    # Extract the vertices in every iteration by applying the marching cube algorithm
    @pymc.deterministic(trace=True)
    def gempy_surfaces(value=0, gempy_model=gempy_model):
        vert, simp = gp.get_surfaces(interp_data, gempy_model[0][1], gempy_model[1][1],
                         original_scale=True)
        return vert

    # We add all the pymc objects to a list
    params = [input_data, gempy_model, gempy_surfaces, *interface_Z_modifier]

    # We create the pymc model i.e. the probabilistic graph
    model = pymc.Model(params)
    runner = pymc.MCMC(model)

    # BAYESIAN INFERENCE
    # ==================
    # Number of iterations
    iterations = 10000

    # Inference. By default without likelihoods: Sampling from priors
    runner.sample(iter=iterations, verbose=1)

The suite of possible realization of the geological model are stored, as
traces, in a database of choice (HDF5, SQL or Python pickles) for
further analysis and visualization.

In 2-D we can display all possible locations of the interfaces on a
cross-section at the center of the model (see figure
[fig:uncertainty]-Priors-2-D representation), however the extension of
uncertainty visualization to 3D is not as trivial. *GemPy* makes use of
the latest developments in uncertainty visualization for 3-D structural
geological modeling
:cite:`c-Lindsay:2012gx,Lindsay:2013cv,Lindsay:2013dr,Wellmann:2012wf`.
The first method consists on representing the probability of finding a
given geological unit :math:`F` at each discrete location in the model
domain. This can be done by defining a probability function

.. math:: p_F(x) = \sum_{k\in n} \frac{I_{F_k}(x)}{n}


where n is the number of realizations and :math:`I_{F_k}(x)` is a
indicator function of the mentioned geological unit (figure
[fig:uncertainty]-Probability shows the probability of finding Layer 1).
However this approach can only display each unit individually. A way to
encapsulate geomodel uncertainty with a single parameter to quantify and
visualize it, is by applying the concept of information entropy
:cite:`c-Wellmann:2012wf`, based on the general concept
developed by :cite:`c-Shannon.1948`. For a discretized
geomodel the information entropy :math:`H` (normalized by the total
number of voxels :math:`n`) can be defined as

.. math:: H = - \sum_{i=1}^{n}p_i \log_2p_i


where :math:`p_F` represents the probability of a layer at cell
:math:`x`. Therefore, we can use information entropy to compress our
uncertainty into a single value at each voxel as an indication of
uncertainty, reflecting the possible number of outcomes and their
relative probability (see figure [fig:uncertainty]-Entropy).

.. figure:: figs/uncertainty1.png
   :alt:

   Probabilistic Programming results on a cross-section at the middle of
   the model (:math:`Y = 10000 \, [m]`). (i) Priors-UQ shows the
   uncertainty of geological models given stochastic values to the Z
   position of the input data (:math:`\sigma = 300`): (top) 2-D
   interface representation ; (middle) probability of occurrence for
   Layer 1; (bottom) information entropy. (ii) Representation of data
   used as likelihood functions: (top) ideal topology graph; (middle)
   Synthetic model taken as reference for the gravity inversion;
   (bottom) Reference forward gravity overlain on top of an XY
   cross-section of the synthetic reference model. Posterior analysis
   after combining priors and likelihood in a Bayesian inference: (top)
   2-D interface representation; (middle) probability of occurrence for
   Layer 1; (bottom) information entropy.

Geological inversion: Gravity and Topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although computing the forward gravity has its own value for many
applications, the main aim of *GemPy* is to integrate all possible
sources of information into a single probabilistic framework. The use of
likelihood functions in a Bayesian inference in opposition to simply
rejection sampling has been explored by the authors during the recent
years
:cite:`c-delaVarga:dj,wellmann2017uncertainty,schaaf17master`.
This approach enables to tune the conditioning of possible stochastic
realizations by varying the probabilistic density function used as
likelihoods. In addition, Bayesian networks allow to combine several
likelihood functions, generating a competition among the prior
distribution of the input data and likelihood functions resulting in
posterior distributions that best honor all the given information. To
give a flavor of what is possible, we apply custom likelihoods to the
previous example based on, topology and gravity constrains in an
inversion.

As, we have shown above, topological graphs can represent the
connectivity among the segmented areas of a geological model. As is
expected, stochastic perturbations of the input data can rapidly alter
the configuration of mentioned graphs. In order to preserve a given
topological configuration partially or totally, we can construct
specific likelihood functions. To exemplify the use of a topological
likelihood function, we will use the topology computed in the section
[sec:topology] derived from the initial model realization (figure
[fig:topology] or [fig:uncertainty]-Likelihoods) as “ideal topology”.
This can be based on an expert interpretation of kinematic data or
deduced from auxiliary data.

The first challenge is to find a metric that captures the similarity of
two graphs. As a graph is nothing but a set of nodes and their edges we
can compare the intersection and union of two different sets using the
the Jaccard index
:cite:`c-jaccard1912distribution,Thiele:2016wx`. It
calculates the ratio of intersection and union of two given graphs A and
B:

.. math:: J(A, B) = \frac{A \cap B}{A \cup B}


The resulting ratio is zero for entirely different graphs, while the
metric rises as the sets of edges and nodes become more similar between
two graphs and reaches exactly one for an identical match. Therefore,
the Jaccard index can be used to express the similarity of topology
graphs as a single number we can evaluate using a probability density
function. The type of probability density function used will determine
the “strength” or likelihood that the mean graph represent. Here, we use
a half Cauchy distribution (:math:`\alpha = 0` and
:math:`\beta = 10^{-3}`) due to its tolerance to outliers.

::

    ...
    Add Listing 6
    ...

    # Computation of toplogy
    @pymc.deterministic(trace=True)
    def gempy_topo(value=0, gm=gempy_model, verbose=False):
        G, c, lu, lot1, lot2 = gp.topology_compute(geo_data, gm[0][0], gm[1], cell_number=0, direction="y")

        if verbose:
            gp.plot_section(geo_data, gm[0][0], 0)
            gp.topology_plot(geo_data, G, c)

        return G, c, lu, lot1, lot2

    # Computation of L2-Norm for the forward gravity
    @pymc.deterministic
    def e_sq(value = original_grav, model_grav = gempy_model[2], verbose = 0):
        square_error =  np.sqrt(np.sum((value*10**-7 - (model_grav*10**-7))**2))
        return square_error

    # Likelihoods
    # ===========
    @pymc.stochastic
    def like_topo_jaccard_cauchy(value=0, gempy_topo=gempy_topo, G=topo_G):
    """Compares the model output topology with a given topology graph G using an inverse Jaccard-index embedded in a half-cauchy likelihood."""
    # jaccard-index comparison
        j = gp.Topology.compare_graphs(G, gempy_topo[0])
    # the last parameter adjusts the "strength" of the likelihood
        return pymc.half_cauchy_like(1 - j, 0, 0.001)

    @pymc.observed
    def inversion(value = 1, e_sq = e_sq):
        return pymc.half_cauchy_like(e_sq,0,0.1)

    # We add all the pymc objects to a list
    params = [input_data, gempy_model, gempy_surfaces, gempy_topo, *interface_Z_modifier,
    like_topo_jaccard_cauchy, e_sq, inversion]

    # We create the pymc model i.e. the probabilistic graph
    model = pymc.Model(params)
    runner = pymc.MCMC(model)

    # BAYESIAN INFERENCE
    # ==================
    # Number of iterations
    iterations = 15000

    # Inference. Adaptive Metropolis
    runner.use_step_method(pymc.AdaptiveMetropolis, params, delay=1000)
    runner.sample(iter = 20000, burn=1000, thin=20, tune_interval=1000, tune_throughout=True)

Gravity likelihoods exploit the spatial distribution of density which
can be related to different lithotypes
:cite:`c-Dentith:2014uk`. To test the likelihood function
based on gravity data, we first generate the synthetic “measured” data.
This was done simply by computing the forward gravity for one of the
extreme models (to highlight the effect that a gravity likelihood can
have) generated during the Monte Carlo error propagation in the previous
section. This model is particularly characteristic by its high dip
values (figure [fig:uncertainty]-Syntetic model to produce forward
gravity). Once we have an “observed” gravity, we can compare it to a
simulated gravity response. To do so, we compare their values applying
an L2-norm encapsulating the difference into a single error value. This
error value acts as the input of the likelihood function, in this case,
a half Cauchy (:math:`\alpha = 0` and :math:`\beta = 10^{-1}`). This
probabilistic density function increases as we approach to 0 and at both
extremes (very low or high values of error) the function flatters to
accommodate to possible measurement errors.

As sampler we use an adaptive Metropolis method
(:cite:`c-Haario:2001tg`, for a more in depth explanation
of samplers and their importance see
:cite:`c-delaVarga:dj`). This method varies the metropolis
sampling size according to the covariance function that gets updated
every :math:`n` iterations. For the results here exposed, we performed
20000 iterations, tuning the adaptive covariance every 1000 steps (a
convergence analysis can be found in the Jupyter notebooks attached to
the on-line supplement of this paper).

As a result of applying likelihood functions we can appreciate a clear
change in the posterior (i.e. the possible outcomes) of the inference. A
closer look shows two main zones of influence, each of them related to
one of the likelihood functions. On one hand, we observe a reduction of
uncertainty along the fault plane due to the restrictions that the
topology function imposes by conditioning the models to high Jaccard
values. On the other hand, what in the first example—i.e. Monte Carlo
error propagation—was just an outlier, due to the influence of the
gravity inversion, now it becomes the norm bending the layers
pronouncedly. In both cases, it is important to keep in mind that the
grade of impact into the final model is inversely proportional to the
amount of uncertainty that each stochastic parameter carries. Finally,
we would like to remind the reader that the goal of this example is not
to obtain realistic geological models but to serve as an example how the
in-built functionality of *GemPy* can be used to handle similar cases.


.. bibliography:: small.bib
   :cited:
   :labelprefix: C
   :keyprefix: c-
   :style: unsrt