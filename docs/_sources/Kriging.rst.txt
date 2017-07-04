
Kriging structure of the *Potential field method*
-------------------------------------------------

The potential field method (cite Lajaunie) serves as the core of method
used in GeMpy to generate the 3-D geological model. The idea is to
exploit the stratigrafic nature of geological settings describing the
deposition direction with an scalar field. This means that every
layer---or in a sense every synchronal deposition---will have the same
value forming a sequence of isosurfaces :math:`k`. In addition, the
direction of the layers can be provided either in the form of dips (i.e.
tangential to the isosurfaces) or pole (i.e. perpendicular to the
isosurface) allowing to utilize more information as input in the forward
modeling step.

\*\* Mathematical description \*\* page 574 laujaunie

Mathematically the method is based on a specific universal co-kriging
where the variables to interpolate that bears algebraic relation between
them, which is one parameter is the first derivative of the other.

.. math::
   :nowrap:

   \begin{equation}
   \frac{\partial Z_\it{i}}{\partial u}(x) = \lim_{\it{p}\to 0} \frac{ Z(x+pu)-Z(x)}{p}
   \end{equation}

Kriging or Gaussian process regression (cite Matheron) is a spatial
interpolation method that makes use of a given covariance function to
compute the best linear unbiased prediction between the data. This
method can be easily extended to multivariate methods---i.e.
cokriging--- and the consideration of drifts in the mean values---i.e.
universal kriging.

Normally, notation of cokriging parameters is complicated since it has
to be considered *p* random functions :math:`\bf{Z}_\it{i}` (i.e. every
parameter involved in the interpolation), sampled at different points
:math:`\bf{x}` of the three-dimensional domain :math:`\mathbb{R}^3`.
Therefore for clarity in this paper we will refer to the potential field
random function :math:`\bf{Z}` and its set of samples
:math:`{\bf{x}}_{\alpha}` while the second random function will be
:math:`\partial {\bf{Z}}/ \partial u` and its samples
:math:`{\bf{x}}_{\beta}`. In addition, samples that belong to a singular
layer or foliation will be denoted as a subset denoted by a superscript
as :math:`{\bf{x}}_\alpha ^k` and every individual point by a
subscript,\ :math:`{\bf{x}}_{\alpha \, i}^k`.

Universal co-kriging allows to modify the unbiased conditions of kriging
making use of two or more variables exploiting their relative drift
dependencies. In our particular case, the main advantage is to be able
to utilized two different type of data sampled from different locations
to estimate both parameters---potential field, :math:`\bf{Z}`, and pole,
:math:`\partial {\bf{Z}}/ \partial u`---as if they were sampled in all
the involved locations at any given point :math:`{\bf{x}}_0`.Due to the
mathematical dependencies between the two variables allows to express
the universal drift as

.. math::
   :nowrap:

   \begin{equation}
   \lambda F_1 + \lambda F_2 = f_10
   \end{equation}

resulting a cokriging system of the form:

.. math::
   :nowrap:

   \begin{equation}
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
   {\mu} & {\mu} \end{array} \right] =
   \left[ \begin{array}{cc}
   {\bf{c_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}} & {\bf{c_{\partial {\bf{Z}}/ \partial u, \, Z}}} \\
   {\bf{c_{Z, \,\partial {\bf{Z}}/ \partial u}}} &  {\bf{c_{\bf{Z}, \,\bf{Z}}}} \\
   {\bf{f_{10}}} & {\bf{f_{20}}} \end{array} \right]
   \label{krig_sys}
   \end{equation}

As we can see in Eq , it is possible to solve the kriging system for the
potential field, **Z** as well as its derivative
:math:`\partial {\bf{Z}}/ \partial u`. Whether the main goal is the
segmentation of the layers which is done using the value of **Z**, the
gradient of the potential field could be used for further mathematical
application such as meshing or geophysical forward calculations.
