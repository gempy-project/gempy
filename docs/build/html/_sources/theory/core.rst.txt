
CORE - Geological modeling with GemPy
=====================================


In this section, we describe the core functionality of *GemPy*: the
construction of 3-D geological models from geological input data
(surface contact points and orientation measurements) and defined
topological relationships (stratigraphic sequences and fault networks).
We begin with a brief review of the theory underlying the implemented
interpolation algorithm. We then describe the translation of this
algorithm and the subsequent model generation and visualisation using
the Python front-end of *GemPy* and how an entire model can be
constructed by calling only a few functions. Across the text, we include
code snippets with minimal working examples to demonstrate the use of
the library.

After describing the simple functionality required to construct models,
we go deeper into the underlying architecture of *GemPy*. This part is
not only relevant for advanced users and potential developers, but also
highlights a key aspect: the link to *Theano*
:cite:`b-2016theano`, a highly evolved Python library for
efficient vector algebra and machine learning, which is an essential
aspect required for making use of the more advanced aspects of
stochastic geomodeling and Bayesian inversion, which will also be
explained in the subsequent sections.

Geological modeling and the potential-field approach
----------------------------------------------------

Concept of the potential-field method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The potential-field method developed by
:cite:`b-Lajaunie.1997` is the central method to generate the
3D geological models in *GemPy*, which has already been successfully
deployed in the modeling software GeoModeller 3-D see
:cite:`b-Calcagno.2008`. The general idea is to
construct an interpolation function :math:`\textbf{Z}({\bf{x}}_{0})`
where :math:`\text{x}` is any point in the continuous three-dimensional
space (:math:`x, y, z`) :math:`\in \mathbb{R}^3` which describes the
domain :math:`\mathcal{D}` as a scalar field. The gradient of the scalar
field will follow the direction of the anisotropy of the stratigraphic
structure or, in other words, every possible isosurface of the scalar
field will represent every synchronal deposition of the layer (see
figure [fig:potfield]).

Let’s break down what we actually mean by this: Imagine that a
geological setting is formed by a perfect sequence of horizontal layers
piled one above the other. If we know the exact timing of when one of
these surfaces was deposited, we would know that any layer above had to
occur afterwards while any layer below had to be deposited earlier in
time. Obviously, we cannot have data for each of these infinitesimal
synchronal layers, but we can interpolate the “date” between them. In
reality, the exact year of the synchronal deposition is
meaningless—since the related uncertainty would be out of proportion.
What has value to generate a 3D geomodel is the location of those
synchronal layers and especially the lithological interfaces where the
change of physical properties are notable. Due to this, instead
interpolating *time*, we use a simple dimensionless parameter—that we
simply refer to as *scalar field value*.

The advantages of using a global interpolator instead of interpolating
each layer of interest independently are twofold: (i) the location of
one layer affects the location of others in the same depositional
environment, making impossible for two layers in the same potential
field to cross; and (ii) it enables the use of data in-between the
interfaces of interest, opening the range of possible measurements that
can be used in the interpolation.

.. figure:: figs/potential_field_simple.png
   :alt:

   Example of scalar field. The input data is formed by six points
   distributed in two layers (:math:`{\bf{x}}_{\alpha \, i}^1` and
   :math:`{\bf{x}}_{\alpha \, i}^2`) and and two orientations
   (:math:`{\bf{x}}_{\beta \, j}`). A isosurface connect the interface
   points and the scalar field is perpendicular to the foliation
   gradient.

The interpolation function is obtained as a weighted interpolation based
on Universal CoKriging :cite:`b-chiles2009geostatistics`.
Kriging or Gaussian process regression
:cite:`b-matheron1981splines` is a spatial interpolation that
treats each input as a random variable, aiming to minimize the
covariance function to obtain the best linear unbiased predictor
:cite:`b-wackernagel2013multivariate`.
Furthermore, it is possible to combine more than one type of data—i.e. a
multivariate case or CoKriging—to increase the amount of information in
the interpolator, as long as we capture their relation using a
cross-covariance. The main advantage in our case is to be able to
utilize data sampled from different locations in space for the
estimation. Simple Kriging, as a regression, only minimizes the second
moment of the data (or variances). However in most geological settings,
we can expect linear trends in our data—i.e. the mean thickness of a
layer varies across the region linearly. This trend is captured using
polynomial drift functions to the system of equations in what is called
Universal Kriging.

Adjustments to structural geological modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far we have shown what we want to obtain and how Universal CoKriging
is a suitable interpolation method to get there. In the following, we
will describe the concrete steps from taking our input data to the final
interpolation function :math:`\textbf{Z}({\bf{x}}_{0})`, which describes
the domain. Much of the complexity of the method comes from the
difficulty of keeping highly nested nomenclature consistent across
literature. For this reason, we will try to be especially verbose
regarding the mathematical terminology. The terms of *potential field*
(original coined by :cite:`b-Lajaunie.1997`) and *scalar
field* (preferred by the authors) are used interchangeably across the
paper. The result of a Kriging interpolation is a random function and
hence both *interpolation function* and *random function* are used to
refer the function of interest :math:`\textbf{Z}({\bf{x}}_{0})`. The
CoKriging nomenclature quickly grows complicated, since it has to
consider *p* random functions :math:`\bf{Z}_{\it{i}}`, with :math:`p`
being the number of distinct parameters involved in the interpolation,
sampled at different points :math:`\bf{x}` of the three-dimensional
domain :math:`\mathbb{R}^3`. Two types of parameters are used to
characterize the *scalar field* in the interpolation: (i) layer
interface points :math:`{\bf{x}}_{\alpha}` describing the respective
isosurfaces of interest—usually the interface between two layers; and
(ii) the gradients of the scalar field, :math:`{\bf{x}}_{\beta}`—or in
geological terms: poles of the layer, i.e. normal vectors to the dip
plane. Therefore gradients will be oriented perpendicular to the
isosurfaces and can be located anywhere in space. We will refer to the
main random function—the scalar field itself—\ :math:`{\bf{Z}}_{\alpha}`
simply as :math:`\bf{Z}`, and its set of samples as
:math:`{\bf{x}}_{\alpha}` while the second random function
:math:`{\bf{Z}}_{\beta}`—the gradient of the scalar field—will be
referred to as :math:`\partial {\bf{Z}}/ \partial u` and its samples as
:math:`{\bf{x}}_{\beta}`, so that we can capture the relationship
between the potential field :math:`\bf{Z}` and its gradient as

.. math::

   \frac{\partial \bf{Z}}{\partial u}(x) = \lim_{\it{p}\to 0} \frac{ {\bf{Z}} (x+pu)-{\bf{Z}}(x)}{p}
   \label{eq_der}


It is also important to keep the values of every individual synchronal
layer identified since they have the same scalar field value. Therefore,
samples that belong to a single layer :math:`k` will be expressed as a
subset denoted using superscript as :math:`{\bf{x}}_\alpha ^k` and every
individual point by a subscript, :math:`{\bf{x}}_{\alpha \, i}^k` (see
figure [fig:potfield]).

Note that in this context data does not have any meaningful physical
parameter associated with it that we want to interpolate as long as
stratigraphic deposition follows gradient direction. Therefore the two
constraints we want to conserve in the interpolated scalar field are:
(i) all points belonging to a determined interface
:math:`{\bf{x}}_{\alpha \, i}^k` must have the same scalar field value
(i.e. there is an isosurface connecting all data points)

.. math::

   {{\bf{Z}}( \bf{x}}_{\alpha_\, i}^k ) - {\bf{Z}}({\bf{x}}_{\alpha_\, 0}^k) = 0
   \label{eq_rel}


where :math:`{\bf{x}}_{\alpha_\, 0}^k` is a reference point of the
interface and (ii) the scalar field will be perpendicular to the poles
:math:`{\bf{x}}_{\beta}` anywhere in 3-D space.

Considering equation [eq\_rel], we do not care about the exact value at
:math:`{{\bf{Z}}(\bf{x}}_{\alpha_\, i}^k)` as long as it is constant at
all points :math:`{\bf{x}}_{\alpha_\, i}^k`. Therefore, the random
function **Z** in the CoKriging system (equation [krig\_sys]) can be
substituted by equation [eq\_rel]. This formulation entails that the
specific *scalar field values* will depend only on the gradients and
hence at least one gradient is necessary to keep the system of equations
defined. The reason for this formulation rest on that by not fixing the
values of each interface :math:`{{\bf{Z}}( \bf{x}}_{\alpha}^k )`, the
compression of layers—which is derived by the gradients—can propagate
smoother beyond the given interfaces. Otherwise, the gradients will only
have effect in the area within the boundaries of the two interfaces that
contains the variable.

The algebraic dependency between **Z** and
:math:`\partial {\bf{Z}}/ \partial u` (equation [eq\_der]) gives a
mathematical definition of the relation between the two variables
avoiding the need of an empirical cross-variogram, enabling instead the
use of the derivation of the covariance function. This dependency must
be taken into consideration in the computation of the drift of the first
moment as well having a different function for each of the variables

.. math:: \lambda F_1 + \lambda F_2 = f_{10}


where :math:`F_1` is a the polynomial of degree :math:`n` and
:math:`F_2` its derivative. Having taken this into consideration, the
resulting CoKriging system takes the form of:

.. math::

   \left[ \begin{array}{ccc}
   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}} &
   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, Z}}} &
   \bf{U_{\partial {\bf{Z}}/ \partial u}} \\
   {\bf{C_{Z, \, \partial {\bf{Z}}/ \partial u }}} &
   {\bf{C_{\bf{Z}, \, \bf{Z}}}} &
   {\bf{U_{Z}}} \\
   \bf{U'_{\partial {\bf{Z}}/ \partial u}} &
   {\bf{U'_{Z}}} &
   {\bf{0}} \end{array} \right]
   \left[ \begin{array}{cc}
   \lambda_{{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}} &
   \lambda_{\partial {\bf{Z}}/ \partial u, \, Z}\\
   \lambda_{Z, \,\partial {\bf{Z}}/ \partial u} &
   \lambda_{\bf{Z}, \,\bf{Z}}\\
   {\mu_{\partial {\text{u}}}} & {\mu_{\text{u}}} \end{array} \right] =
   \left[ \begin{array}{cc}
   {\bf{c_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}} & {\bf{c_{\partial {\bf{Z}}/ \partial u, \, Z}}} \\
   {\bf{c_{Z, \,\partial {\bf{Z}}/ \partial u}}} &  {\bf{c_{\bf{Z}, \,\bf{Z}}}} \\
   {\bf{f_{10}}} & {\bf{f_{20}}} \end{array} \right]
   \label{krig_sys}


where, :math:`{\bf{C_{\partial {\bf{Z}}/ \partial u}}}` is the gradient
covariance-matrix; :math:`{\bf{C_{\bf{Z}, \, \bf{Z}}}}` the
covariance-matrix of the differences between each interface points to
reference points in each layer

.. math::

   {{C}}_{{\bf{x}}_{\alpha \, i}^r, \, {\bf{x}}_{\alpha \,j}^s} =
   C_{x^r_{\alpha, \,i} \, x^s_{\alpha, \,j}} - C_{x^r_{\alpha, \,0} \, x^s_{\alpha, \,j}} -
   C_{x^r_{\alpha, \,i} \, x^s_{\alpha, \,0}} + C_{x^r_{\alpha, \,0} \, x^s_{\alpha, \,0}}
   \label{one_val}


(see Appendix [interface-covariance-matrix] for further analysis);
:math:`{\bf{C_{Z, \, \partial {\bf{Z}}/ \partial u }}}` encapsulates the
cross-covariance function; and :math:`{\bf{U_{Z}}}` and
:math:`\bf{U'_{\partial {\bf{Z}}/ \partial u}}` are the drift functions
and their gradient, respectively. On the right hand side we find the
vector of the matrix system of equations, being
:math:`{\bf{c_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial
      v}}}` the gradient of the covariance function to the point **x**
of interest; :math:`{\bf{c_{Z, \,\partial {\bf{Z}}/ \partial u}}}` the
cross-covariance; :math:`{\bf{c_{\bf{Z}, \,\bf{Z}}}}` the actual
covariance function; and :math:`{\bf{f_{10}}}` and :math:`{\bf{f_{20}}}`
the gradient of the drift functions and the drift functions themselves.
Lastly, the unknown vectors are formed by the corresponding weights,
:math:`\lambda`, and constants of the drift functions, :math:`\mu`. A
more detail inspection of this system of equations is carried out in
Appendix [kse].

As we can see in equation [krig\_sys], it is possible to solve the
Kriging system for the scalar field **Z** (second column in the weights
vector), as well as its derivative :math:`\partial {\bf{Z}}/ \partial u`
(first column in the weights vector). Even though the main goal is the
segmentation of the layers, which is done using the value of **Z** (see
Section [from-potential-field-to-block]), the gradient of the scalar
field can be used for further mathematical applications, such as
meshing, geophysical forward calculations or locating geological
structures of interest (e.g. spill points of a hydrocarbon trap).

Furthermore, since the choice of covariance parameters is ad hoc
(Appendix [covariance-function-cubic.-discuss-it-with-france] show the
used covariance function in GemPy), the uncertainty derived by the
Kriging interpolation does not bear any physical meaning. This fact
promotes the idea of only using the mean value of the Kriging solution.
For this reason it is recommended to solve the Kriging system (equation
[krig\_sys]) in its dual form
:cite:`b-matheron1981splines`.

Geological model interpolation using *GemPy*
--------------------------------------------

From scalar field to geological block model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most scenarios the goal of structural modeling is to define the
spatial distribution of geological structures, such as layers interfaces
and faults. In practice, this segmentation usually is done either by
using a volumetric discretization or by depicting the interfaces as
surfaces.

The result of the Kriging interpolation is the random function
:math:`\textbf{Z}(x)` (and its gradient
:math:`\partial {\bf{Z}} / \partial u (x)`, which we will omit in the
following), which allows the evaluation of the value of the scalar field
at any given point :math:`x` in space. From this point on, the easiest
way to segment the domains is to discretize the 3-D space (e.g. we use a
regular grid in figure [fig:model\_comp]). First, we need to calculate
the scalar value at every interface by computing
:math:`{{\bf{Z}}( \bf{x}}_{\alpha, i}^k)` for every interface
:math:`k_i`. Once we know the value of the scalar field at the
interfaces, we evaluate every point of the mesh and compare their value
to those at the interfaces, identifying every point of the mesh with a
topological volume. Each of these compartmentalizations will represent
each individual domain, this is, each lithology of interest (see figure
[fig:model\_comp]a).

At the time of this manuscript preparation, *GemPy* only provides
rectilinear grids but it is important to notice that the computation of
the scalar field happens in continuous space, and therefore allows the
use of any type of mesh. The result of this type of segmentation is
referred in *GemPy* as a *lithology block*.

The second segmentation alternative consist on locating the layer
isosurfaces. *GemPy* makes use of the marching cube algorithm
:cite:`b-lorensen1987marching` provided by the *scikit-image*
library :cite:`b-scikit-image`. The basics of the marching
cube algorithm are quite intuitive: (i) First, we discretize the volume
in 3-D voxels and by comparison we look if the value of the isosurface
we want to extract falls within the boundary of every single voxel; (ii)
if so, for each edge of the voxel, we interpolate the values at the
corners of the cube to obtain the coordinates of the intersection
between the edges of the voxels and the isosurface of interest, commonly
referred to as vertices; (iii) those intersections are analyzed and
compared against all possible configurations to define the simplices
(i.e. the vertices which form an individual polygon) of the triangles.
Once we obtain the coordinates of vertices and their correspondent
simplices, we can use them for visualization (see Section [vis]) or any
sub-sequential computation that may make use of them (e.g. weighted
voxels). For more information on meshing algorithms refer to
:cite:`b-geuzaine2009gmsh`.

::

    import gempy as gp

    # Main data management object containing
    geo_data = gp.create_data(extent=[0, 20, 0, 10, -10, 0],
                  resolution=[100, 10, 100],
                  path_o="paper_Foliations.csv",
                  path_i="paper_Points.csv")

    # Creating object with data prepared for interpolation and compiling
    interp_data = gp.InterpolatorData(geo_data)

    # Computing result
    lith, fault = gp.compute_model(interp_data)

    # Plotting result: scalar field
    gp.plot_scalar_field(geo_data, lith[1], 5, plot_data=True)

    # Plotting result: lithology block
    gp.plot_section(geo_data, lith[0], 5, plot_data=True)

    # Getting vertices and faces
    vertices, simpleces = gp.get_surfaces(interp_data, lith[1], [fault[1]], original_scale=True)

.. figure:: figs/model_comp.png
   :alt:

   Example of different lithological units and their relation to scalar
   fields. a) Simple stratigraphic sequence generated from a scalar
   field as product of the interpolation of interface points and
   orientation gradients. b) Addition of an unconformity horizon from
   which the unconformity layer behaves independently from the older
   layers by overlying a second scalar field. c) Combination of
   unconformity and faulting using three scalar fields.

Combining scalar fields: Depositional series and faults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In reality, most geological settings are formed by a concatenation of
depositional phases partitioned by unconformity boundaries and subjected
to tectonic stresses which displace and deform the layers. While the
interpolation is able to represent realistic folding—given enough
data—the method fails to describe discontinuities. To overcome this
limitation, it is possible to combine several scalar fields to recreate
the desired result.

So far the implemented discontinuities in *GemPy* are unconformities and
infinite faults. Both types are computed by specific combinations of
independent scalar fields. We call these independent scalar fields
*series*
from stratigraphic series in accordance to the use in GeoModeller 3-D]
and in essence, they represent a subset of grouped interfaces—either
layers or fault planes—that are interpolated together and therefore
their spatial location affect each other. To handle and visualize these
relationships, we use a so called sequential pile; representing the
order—from the first scalar field to the last—and the grouping of the
layers (see figure [fig:model\_comp]).

Modeling unconformities is rather straightforward. Once we have grouped
the layers into their respective series, younger series will overlay
older ones beyond the unconformity. The scalar fields themselves,
computed for each of these series, could be seen as a continuous
depositional sequence in the absence of an unconformity.

::

    import gempy as gp

    # Main data management object containing
    geo_data = gp.create_data(extent=[0, 20, 0, 10, -10, 0],
                  resolution=[100, 10, 100],
                  path_o="paper_Foliations.csv",
                  path_i="paper_Points.csv")

    # Defining the series of the sequential pile
    gp.set_series(geo_data,
             {'younger_serie' : 'Unconformity', 'older_serie': ('Layer1', 'Layer2')},
             order_formations= ['Unconformity', 'Layer2', 'Layer1'])

    # Creating object with data prepared for interpolation and compiling
    interp_data = gp.InterpolatorData(geo_data)

    # Computing result
    lith, fault = gp.compute_model(interp_data)

    # Plotting result
    gp.plot_section(geo_data, lith[0], 5, plot_data=True)

Faults are modeled by the inclusion of an extra drift term into the
kriging system :cite:`b-marechal1984kriging`:

.. math::

   \left[ \begin{array}{cccc}
   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}} &
   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, Z}}} &
   \bf{U_{\partial {\bf{Z}}/ \partial u}}         &
   \bf{F_{\partial {\bf{Z}}/ \partial u}} \\
   {\bf{C_{Z, \, \partial {\bf{Z}}/ \partial u }}} &
   {\bf{C_{\bf{Z}, \, \bf{Z}}}} &
   {\bf{U_{Z}}}                 &
   {\bf{F_{Z}}} \\
   \bf{U'_{\partial {\bf{Z}}/ \partial u}} &
   {\bf{U'_{Z}}} &
   {\bf{0}}      &
   {\bf{0}}    \\
   \bf{F'_{\partial {\bf{Z}}/ \partial u}} &
   {\bf{F'_{Z}}} &
    {\bf{0}}    &
    {\bf{0}}
   \end{array} \right]
   \left[ \begin{array}{cc}
   \lambda_{{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}} &
   \lambda_{\partial {\bf{Z}}/ \partial u, \, Z}\\
   \lambda_{Z, \,\partial {\bf{Z}}/ \partial u} &
   \lambda_{\bf{Z}, \,\bf{Z}}\\
   {\mu_{\partial {\text{u}}}} & {\mu_{\text{u}}} \\
   {\mu_{\partial {\text{f}}}} & {\mu_{\text{f}}}
    \end{array} \right] =
   \left[ \begin{array}{cc}
   {\bf{c_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}} & {\bf{c_{\partial {\bf{Z}}/ \partial u, \, Z}}} \\
   {\bf{c_{Z, \,\partial {\bf{Z}}/ \partial u}}} &  {\bf{c_{\bf{Z}, \,\bf{Z}}}} \\
   {\bf{f_{10}}} & {\bf{f_{20}}} \\
   {\bf{f_{10}}} & {\bf{f_{20}}}
    \end{array} \right]
   \label{krig_sys_f}


which is a function of the faulting structure. This means that for
every location :math:`{\bf{x}}_{0}` the drift function will take a value
depending on the fault compartment—i.e. a segmented domain of the fault
network—and other geometrical constrains such as spatial influence of a
fault or variability of the offset. To obtain the offset effect of a
fault, the value of the drift function has to be different at each of
its sides. The level of complexity of the drift functions will determine
the quality of the characterization as well as its robustness.
Furthermore, finite or localize faults can be recreated by selecting an
adequate function that describe those specific trends.

::

    import gempy as gp

    # Main data management object containing
    geo_data = gp.create_data(extent=[0,20,0,10,-10,0],
                  resolution=[100,10,100],
                  path_o = "paper_Foliations.csv",
                  path_i = "paper_Points.csv")

    # Defining the series of the sequential pile
    gp.set_series(geo_data, series_distribution={'fault_serie1': 'fault1',
                'younger_serie' : 'Unconformity',
                'older_serie': ('Layer1', 'Layer2')},
    order_formations= ['fault1', 'Unconformity', 'Layer2', 'Layer1'])

    # Creating object with data prepared for interpolation and compiling
    interp_data = gp.InterpolatorData(geo_data)

    # Computing result
    lith, fault = gp.compute_model(interp_data)

    # Plotting result
    gp.plot_section(geo_data, lith[0], 5, plot_data=True)

    # Getting vertices and faces and pltting
    vertices, simpleces = gp.get_surfaces(interp_data,lith[1], [fault[1]], original_scale=True)
    gp.plot_surfaces_3D(geo_data, ver_s, sim_s)

The computation of the segmentation of fault compartments (called *fault
block* in *GemPy*)—prior to the inclusion of the fault drift functions
which depends on this segmentation—can be performed with the
potential-field method itself. In the case of multiple faults,
individual drift functions have to be included in the kriging system for
each fault, representing the subdivision of space that they produce.
Naturally, younger faults may offset older tectonic events. This
behavoir is replicated by recursively adding drift functions of younger
faults to the computation of the older *fault blocks*. To date, the
fault relations—i.e. which faults offset others—is described by the user
in a boolean matrix. An easy to use implementation to generate fault
networks is being worked on at the time of manuscript preparation. An
important detail to consider is that drift functions will bend the
isosurfaces according to the given rules but they will conserve their
continuity. This differs from the intuitive idea of offset, where the
interface presents a sharp jump. This fact has direct impact in the
geometry of the final model, and can, for example, affect certain
meshing algorithms. Furthermore, in the ideal case of choosing the
perfect drift function, the isosurface would bend exactly along the
faulting plane. At the current state *GemPy* only includes the addition
of an arbitrary integer to each segmented volume. This limits the
quality to a constant offset, decreasing the sharpness of the offset as
data deviates from that constrain. Any deviation from this theoretical
concept, results in a bending of the layers as they approximate the
fault plane to accommodate to the data, potentially leading to overly
smooth transitions around the discontinuity.

“Under the hood”: The *GemPy* architecture
------------------------------------------

The graph structure
~~~~~~~~~~~~~~~~~~~

The architecture of *GemPy* follows the Python Software Foundation
recommendations of modularity and reusability
:cite:`b-van2001pep`. The aim is to divide all functionality
into independent small logical units in order to avoid duplication,
facilitate readability and make changes to the code base easier.

The design of *GemPy* revolves around an automatic differentiation (AD)
scheme. The main constraint is that the mathematical functions need to
be continuous from the input parameters (in probabilistic jargon priors)
to the cost function (or likelihoods), and therefore the code must be
written in the same language (or at the very least compatible) to
automatically compute the derivatives. In practice, this entails that
any operation involved in the AD must be coded symbolically using the
library *Theano* (see Section [theano] for further details). One of the
constrains of writing symbolically is the a priori declaration of the
possible input parameters of the graph which will behave as latent
variables—i.e. the parameters we try to tune for optimization or
uncertainty quantification—while leaving others involved parameters
constant either due to their nature or because of the relative slight
impact of their variability. This rigidity dictates the whole design of
input data management that needs to revolved around the preexistent
symbolic graph.

*GemPy* encapsulates this creation of the symbolic graph in its the
module ``theanograph``. Due to the significant complexity to program
symbolically, features shipped in *GemPy* that rely heavily in external
libraries are not written in *Theano*. The current functionality written
in *Theano* can be seen in the figure [fig:overall] and essentially it
encompasses all the interpolation of the geological modeling (section
[kriging]) as well as forward calculation of the gravity (section
[gravity]).

.. figure:: figs/GemPy.png
   :alt:

   Graph of the logical structure of *GemPy* logic. There are several
   levels of abstraction represented. (i) The first division is between
   the implicit interpolation of the geological modeling (dark gray) and
   other subsequent operations for different objectives (light gray).
   (ii) All the logic required to perform automatic differentiation is
   presented under the “Theano” label (in purple) (iii) The parts under
   labels “Looping pile” (green) and “Single potential field” (gray),
   divide the logic to control the input data of each necessary scalar
   field and the operations within one of them. (iv) Finally, each
   superset of parameters is color coded according to their
   probabilistic nature and behavior in the graph: in blue, stochastic
   variables (priors or likelihoods); in yellow, deterministic
   functions; and in red the inputs of the graph, which are either
   stochastic or constant depending on the problem.

Regarding data structure, we make use of the Python package *pandas*
:cite:`b-mckinney2011pandas` to store and prepare the input
data for the symbolic graph (red nodes in figure [fig:overall]) or other
processes, such as visualization. All the methodology to create, export
and manipulate the original data is encapsulated in the class
``DataManagement``. This class has several child classes to facilitate
specific precomputation manipulations of data structures (e.g. for
meshing). The aim is to have all constant data prepared before any
inference or optimization is carried out to minimize the computational
overhead of each iteration as much as possible.

It is important to keep in mind that, in this structure, once data
enters the part of the symbolic graph, only algebraic operations are
allowed. This limits the use of many high-level coding structures (e.g.
dictionaries or undefined loops) and external dependencies. As a result
of that, the preparation of data must be exhaustive before starting the
computation. This includes ordering the data within the arrays, passing
the exact lengths of the subsets we will need later on during the
interpolation or the calculation of many necessary constant parameters.
The preprocessing of data is done within the sub-classes of
``DataManagement``, the ``InterpolatorData`` class–of which an instance
is used to call the *Theano* graph—and ``InterpolatorClass``—which
creates the the *Theano* variables and compiles the symbolic graph.

The rest of the package is formed by—an always growing—series of modules
that perform different tasks using the geological model as input (see
Section [sec:model-analys-furth] and the assets-area in figure
[fig:overall]).

Theano
~~~~~~

Efficiently solving a large number of algebraic equations, and
especially their derivatives, can easily get unmanageable in terms of
both time and memory. Up to this point we have referenced many times
*Theano* and its related terms such as automatic differentiation or
symbolic programming. In this section we will motivate its use and why
its capabilities make all the difference in making implicit geological
modeling available for uncertainty analysis.

*Theano* is a Python package that takes over many of the optimization
tasks in order to create a computationally feasible code implementation.
*Theano* relies on the creation of symbolical graphs that represent the
mathematical expressions to compute. Most of the extended programming
paradigms e.g. procedural languages and
object-oriented programming; see :cite:`b-normark2013overview` are executed
sequentially without any interaction with the subsequent instructions.
In other words, a later instruction has access to the memory states but
is clueless about the previous instructions that have modified mentioned
states. In contrast, symbolic programming define from the beginning to
the end not only the primary data structure but also the complete logic
of a function , which in turn enables the optimization (e.g. redundancy)
and manipulation (e.g. derivatives) of its logic.

Within the Python implementation, *Theano* create an acyclic network
graph where the parameters are represented by nodes, while the
connections determine mathematical operators that relate them. The
creation of the graph is done in the class ``theanograph``. Each
individual method corresponds to a piece of the graph starting from the
input data all the way to the geological model or the forward gravity
(see figure [fig:overall], purple Theano area).

The symbolic graph is later analyzed to perform the optimization, the
symbolic differentiation and the compilation to a faster language than
Python (C or CUDA). This process is computational demanding and
therefore it must be avoided as much as possible.

Among the most outstanding optimizers shipped with *Theano*
:cite:`b-2016theano`, we
can find : (i) the canonicalization of the operations to reduce the
number of duplicated computations, (ii) specialization of operations to
improve consecutive element-wise operations, (iii) in-place operations
to avoid duplications of memory or (iv) Open MP parallelization for CPU
computations. These optimizations and more can speed up the code an
order of magnitude.

However, although *Theano* code optimization is useful, the real
game-changer is its capability to perform automatic differentiation.
There is extensive literature explaining all the ins and outs and
intuitions of the method since it is a core algorithm to train neural
networks e.g. a detailed explanation is given by
:cite:`b-baydin2015automatic`.
Here, we will highlight the main differences with numerical approaches
and how they can be used to improve the modeling process.

Many of the most advanced algorithms in computer science rely on an
inverse framework i.e. the result of a forward computation
:math:`f(\textbf{x})` influences the value of one or many of the
:math:`\textbf{x}` latent variables (e.g. neuronal networks,
optimizations, inferences). The most emblematic example of this is the
optimization of a cost function. All these problems can be described as
an exploration of a multidimensional manifold
:math:`f: \mathbb{R}^N \rightarrow \mathbb{R}`. Hence the gradient of
the function
:math:`\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)`
becomes key for an efficient analysis. In case that the output is also
multidimensional—i.e.
:math:`f: \mathbb{R}^N \rightarrow \mathbb{R}^M`—the entire manifold
gradient can be expressed by the Jacobian matrix

.. math::

   Jf =
   \begin{bmatrix}
   \frac{\partial f_1}{\partial x_1} & ... & \frac{\partial f_1}{\partial x_n}\\
   \vdots                          & \ddots & \vdots \\
   \frac{\partial f_n}{\partial x_1} & ... & \frac{\partial f_m}{\partial x_n}
   \end{bmatrix}


of dimension :math:`N \cdot M`, where :math:`N` is the number of
variables and :math:`M` the number of functions that depend on those
variables. Now the question is how we compute the Jacobian matrix in a
consistent and efficient manner. The most straightforward methodology
consists in approximating the derivate by numerical differentiation
applying finite differences approximations, for example a forward FD
scheme:

.. math:: \frac{\partial f_i}{\partial x_i} = \lim_{\it{h}\to 0} \frac{f(x_i+h)-f(x_i)}{h}


where :math:`h` is a discrete increment. The main advantage of
numerical differentiation is that it only computes :math:`f`—evaluated
for different values of :math:`x`—which makes it very easy to implement
it in any available code. By contrast, a drawback is that for every
element of the Jacobian we are introducing an approximation error that
eventually can lead to mathematical instabilities. But above all, the
main limitation is the need of :math:`2 \cdot M \cdot N` evaluations of
the function :math:`f`, which quickly becomes prohibitively expensive to
compute in high-dimensional problems.

The alternative is to create the symbolic differentiation of :math:`f`.
This encompasses decomposing :math:`f` into its primal operators and
applying the chain rule to the correspondent transformation by following
the rules of differentiation to obtain :math:`f'`. However, symbolic
differentiation is not enough since the application of the chain rule
leads to exponentially large expressions of :math:`f'` in what is known
as “expression swell” :cite:`b-cohen2003computer`. Luckily,
these large symbolic expressions have a high level of redundancy in
their terms. This allows to exploit this redundancy by storing the
repeated intermediate steps in memory and simply invoke them when
necessary, instead of computing the whole graph every time. This
division of the program into sub-routines to store the intermediate
results—which are invoked several times—is called dynamic programming
:cite:`b-bellman2013dynamic`. The simplified symbolic
differentiation graph is ultimately what is called automatic
differentiation :cite:`b-baydin2015automatic`. Additionally,
in a multivariate/multi-objective case the benefits of using AD increase
linearly as the difference between the number of parameters :math:`N`
and the number of objective functions :math:`M` get larger. By applying
the same principle of redundancy explained above—this time between
intermediate steps shared across multiple variables or multiple
objective function—it is possible to reduce the number of evaluations
necessary to compute the Jacobian either to :math:`N` in
forward-propagation or to :math:`M` in back-propagation, plus a small
overhead on the evaluations for a more detailed
description of the two modes of AD see :cite:`b-cohen2003computer`.

*Theano* provides a direct implementation of the back-propagation
algorithm, which means in practice that a new graph of similar size is
generated per cost function (or, in the probabilistic inference, per
likelihood function). Therefore, the computational time is independent
of the number of input parameters, opening the door to solving
high-dimensional problems.



.. bibliography:: small.bib
   :cited:
   :labelprefix: B
   :keyprefix: b-
   :style: unsrt