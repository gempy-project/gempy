Advance theory
==============

Kriging system expansion:
-------------------------
Gradient Covariance-Matrix :math:`{\bf{C_{\partial {\bf{Z}}/ \partial u}}}`
---------------------------------------------------------------------------
The following equations have been derived from the work in
:cite:`d-Aug.2004,Lajaunie.1997,chiles2009geostatistics


The gradient covariance-matrix,
:math:`{\bf{C_{\partial {\bf{Z}}/ \partial u}}}`, is made up of as many
variables as gradient directions that are taken into consideration. In
3-D, we would have the Cartesian coordinates
dimensions—\ :math:`{\bf{Z}}/ \partial x`, :math:`{\bf{Z}}/ \partial y`,
and :math:`{\bf{Z}}/ \partial z`—and therefore, they will derive from
the partial differentiation of the covariance function
:math:`\sigma(x_i,x_j)` of :math:`\bf{Z}`.

.. figure:: figs/gradients2.png
   :alt: 

   2-D representation of the decomposition of the orientation vectors
   into Cartesian axis. Each Cartesian axis represent a variable of a
   sub CoKriging system. The dotted green line represent the covariance
   distance, :math:`r`.

As in our case the directional derivatives used are the 3 Cartesian
directions we can rewrite gradients covariance,
:math:`{\bf{C_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}}`
for our specific case as:

.. math::

   {\bf{C_{\partial {\bf{Z}}/ \partial x, \, \partial {\bf{Z}}/ \partial y \, \partial {\bf{Z}}/ \partial z}}} =
   \left[ \begin{array}{ccc}
   {\bf{C_{\partial {\bf{Z}}/ \partial x, \, \partial {\bf{Z}}/ \partial x}}} &
   {\bf{C_{\partial {\bf{Z}}/ \partial x, \, \partial {\bf{Z}}/ \partial y}}} &
   {\bf{C_{\partial {\bf{Z}}/ \partial x, \, \partial {\bf{Z}}/ \partial z}}} \\
   {\bf{C_{\partial {\bf{Z}}/ \partial y, \, \partial {\bf{Z}}/ \partial x}}} &
   {\bf{C_{\partial {\bf{Z}}/ \partial y, \, \partial {\bf{Z}}/ \partial y}}} &
   {\bf{C_{\partial {\bf{Z}}/ \partial y, \, \partial {\bf{Z}}/ \partial z}}}\\
   {\bf{C_{\partial {\bf{Z}}/ \partial z, \, \partial {\bf{Z}}/ \partial x}}} &
   {\bf{C_{\partial {\bf{Z}}/ \partial z, \, \partial {\bf{Z}}/ \partial y}}} &
   {\bf{C_{\partial {\bf{Z}}/ \partial z, \, \partial {\bf{Z}}/ \partial z}}} \end{array} \right]
   \label{C_g}


Notice, however, that covariance functions by definition are described
in a polar coordinate system, and therefore it will be necessary to
apply the chain rule for *directional derivatives*. Considering an
isotropic and stationary covariance we can express the covariance
function as:

.. math:: \sigma(x_i,x_j) = C(r)


with:

.. math:: r = \sqrt{h^2_x+h^2_y}


therefore we need to apply the chain rule in partial differentiation.
For the case of the covariance in a single direction:

.. math::

   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial u}}} =
   \frac{\partial^2 C_{Z} (r)}{\partial h^2_u} =
   \frac{\partial C_{Z} (r)}{\partial r}
   \frac{\partial}{\partial h_u} \left( \frac{\partial r}{\partial h_u} \right) +
   \frac{\partial}{\partial h_u} \left( \frac{\partial C_Z (r)}{\partial r}\right)
   \frac{\partial r}{\partial h_u} \\


where:

.. math:: \frac{\partial C_{Z} (r)}{\partial r} = \frac{\partial C_{Z} (r)}{\partial r} = C'_Z(r)

.. math::

   \frac{\partial r}{\partial h_u} = \frac{h_u}{\sqrt{h^2_u+h^2_v}} = -\frac{h_u}{r}
   \label{eq:}

.. math::

   \frac{\partial}{\partial h_u} \left( \frac{\partial r}{\partial h_u} \right) =
   \frac{\partial}{\partial h_u} \left(\frac{h_u}{\sqrt{h^2_u+h^2_v}}\right) =
   -\frac{2 h_u^2}{2\sqrt{h^2_u+h^2_v}} + \frac{1}{\sqrt{h^2_u+h^2_v}} = -\frac{h_u^2}{r^3} + \frac{1}{r}

.. math::

   \frac{\partial}{\partial h_u} \left( \frac{\partial C_Z (r)}{\partial r}\right) =
   \frac{\partial C'_Z(r)}{\partial h_u} = \frac{\partial C'_{Z} (r)}{\partial r}  \frac{\partial r}{\partial h_u} = - \frac{h_u}{r}C''_Z


Substituting:

.. math::

   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial u}}} = C'_Z(r) \left( -\frac{h_u^2}{r^3} + \frac{1}{r} \right) - \frac{h_u}{r}C''_Z \frac{h_u}{r} = C'_Z(r) \left( -\frac{h_u^2}{r^3} + \frac{1}{r} \right) + \frac{h_u^2}{r^2} C''_Z
   \label{huhu}


While in case of two different directions the covariance will be:

.. math::

   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}} =
   \frac{\partial^2 C_{Z}(r)}{\partial h_u h_v} =
   \frac{\partial C_{Z} (r)}{\partial r}
   \frac{\partial}{\partial h_v} \left( \frac{\partial r}{\partial h_u} \right) +
   \frac{\partial}{\partial h_v} \left( \frac{\partial C_Z (r)}{\partial r}\right)
   \frac{\partial r}{\partial h_u}


with:

.. math::

   \frac{\partial}{\partial h_v} \left( \frac{\partial r}{\partial h_u} \right) =
   \frac{\partial}{\partial h_v} \left(\frac{h_u}{\sqrt{h^2_u+h^2_v}}\right) =
   -\frac{h_u h_v}{r^3}

.. math::

   \frac{\partial}{\partial h_v} \left( \frac{\partial C_Z (r)}{\partial r}\right) =
   \frac{\partial C'_Z(r)}{\partial h_v} = - C''_Z(r) \frac{h_v}{r}


we have:

.. math::

   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}} = C'_Z(r) \left(-\frac{h_u h_v}{r^3} \right) + C''_Z(r) \frac{h_u h_v}{r^2}= \frac{h_u h_v}{r^2} \left( C''_Z (r) - \frac{C'_Z (r)}{r} \right)
   \label{huhv}

This derivation is independent to the covariance function choice,
however, some covariances may lead to mathematical indeterminations.

Interface Covariance-Matrix
---------------------------

.. figure:: figs/interfaces.png
   :alt:

   Distances :math:`r` involved in the computation of the interface
   subsystem of the interpolation. Because all covariances are relative
   to a reference point :math:`x_{\alpha, \, 0}^i`, all four covariances
   must be taken into account (equation [eq:int\_cov])

In a practical sense, keeping the value of the scalar field at every
interface unfixed forces us to consider the covariance between the
points within an interface as well as the covariance between different
layers following equation,

.. math::

   {{C}}_{{\bf{x}}_{\alpha \, i}^r, \, {\bf{x}}_{\alpha \,j}^s} =
   C_{x^r_{\alpha, \,i} \, x^s_{\alpha, \,j}} - C_{x^r_{\alpha, \,0} \, x^s_{\alpha, \,j}} -
   C_{x^r_{\alpha, \,i} \, x^s_{\alpha, \,0}} + C_{x^r_{\alpha, \,0} \, x^s_{\alpha, \,0}}
   \label{eq:int_cov}


This lead to the subdivision of the CoKriging system respecting the
interfaces:

.. math::

   {\bf{C_{\bf{Z}, \, \bf{Z}}}} =
   \begin{bmatrix}
   {\bf{C}}_{{\bf{x}}_{\alpha}^1, \, {\bf{x}}_{\alpha}^1}&
   {\bf{C}}_{{\bf{x}}_{\alpha}^1, \, {\bf{x}}_{\alpha}^2}&
   ... &
   {\bf{C}}_{{\bf{x}}_{\alpha}^1, \, {\bf{x}}_{\alpha}^s}\\
   {\bf{C}}_{{\bf{x}}_{\alpha}^2, \, {\bf{x}}_{\alpha}^1}&
   {\bf{C}}_{{\bf{x}}_{\alpha}^2, \, {\bf{x}}_{\alpha}^2}&
   ... &
   {\bf{C}}_{{\bf{x}}_{\alpha}^2, \, {\bf{x}}_{\alpha}^s}\\
   \vdots &
   \vdots &
   \ddots &
   \vdots \\
   {\bf{C}}_{{\bf{x}}_{\alpha}^r, \, {\bf{x}}_{\alpha}^1}&
   {\bf{C}}_{{\bf{x}}_{\alpha}^r, \, {\bf{x}}_{\alpha}^2}&
   ... &
   {\bf{C}}_{{\bf{x}}_{\alpha}^r, \, {\bf{x}}_{\alpha}^s}
   \end{bmatrix}
   \label{C_i}


Combining Eq [one\_val] and Eq [C\_i] the covariance for the property
*potential field* will look like:

.. math::

   {\bf{C}}_{{\bf{x}}_{\alpha}^r, \, {\bf{x}}_{\alpha}^s} =
   \begin{bmatrix}
   C_{x^1_1 \, x^1_1} - C_{x^1_0 \, x^1_1} -
   C_{x^1_1 \, x^1_0} + C_{x^1_0 \, x^1_0} &
   C_{x^1_1 \, x^1_2} - C_{x^1_0 \, x^1_2} -
   C_{x^1_1 \, x^1_0} + C_{x^1_0 \, x^1_0} &
   ... &
   C_{x^1_1 \, x^s_j} - C_{x^1_0 \, x^s_j} -
   C_{x^1_1 \, x^s_0} + C_{x^1_0 \, x^s_0}\\
   C_{x^1_2 \, x^1_1} - C_{x^1_0 \, x^1_1} -
   C_{x^1_2 \, x^1_0} + C_{x^1_0 \, x^1_0} &
   C_{x^1_2 \, x^1_2} - C_{x^1_0 \, x^1_2} -
   C_{x^1_2 \, x^1_0} + C_{x^1_0 \, x^1_0} &
   ... &
   C_{x^1_2 \, x^s_j} - C_{x^1_0 \, x^s_j} -
   C_{x^1_j \, x^s_0} + C_{x^1_0 \, x^s_0}\\
   \vdots &
   \vdots &
   \ddots &
   \vdots \\
   C_{x^r_i \, x^s_1} - C_{x^r_0 \, x^s_1} -
   C_{x^r_i \, x^s_0} + C_{x^r_0 \, x^s_0} &
   C_{x^r_i \, x^s_2} - C_{x^r_0 \, x^s_2} -
   C_{x^r_i \, x^s_0} + C_{x^r_0 \, x^s_0} &
   ... &
   C_{x^r_i \, x^s_j} - C_{x^r_0 \, x^s_j} -
   C_{x^r_i \, x^s_0} + C_{x^r_0 \, x^s_0}
   \end{bmatrix}


Cross-Covariance
----------------

In a CoKriging system, the relation between the interpolated parameters
is given by a cross-covariance function. As we saw above, the gradient
covariance is subdivided into covariances with respect to the three
Cartesian directions (Eq [C\_g]), while the interface covariance is
detached from the covariances matrices with respect to each individual
interface (Eq [C\_i]). In the same manner, the cross-covariance will
reflect the relation of every interface to each gradient direction,

.. figure:: figs/cross.png
   :alt: 

   Distances :math:`r` involved in the computation of the
   cross-covariance function. In a similar fashion as before, all
   interface covariance are computed relative to a reference point in
   each layer :math:`x_{\alpha, \, 0}^i`\

.. math::

   {\bf{C_{Z, \,\partial {\bf{Z}}/ \partial u}}} =
   \begin{bmatrix}
   {\bf{C}}_{{\bf{x}}_{\alpha \, 1}^1, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, 1})/ \partial x}&
   {\bf{C}}_{{\bf{x}}_{\alpha \, 2}^1, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, 1})/ \partial x}&
   ... &
   {\bf{C}}_{{\bf{x}}_{\alpha \, 1}^1, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, 1})/ \partial y}&
   ... &
   {\bf{C}}_{{\bf{x}}_{\alpha \, i}^1, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, j})/ \partial z}& \\
   {\bf{C}}_{{\bf{x}}_{\alpha \, 1}^2, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, 1})/ \partial x}&
   {\bf{C}}_{{\bf{x}}_{\alpha \, 2}^2, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, 1})/ \partial x}&
   ... &
   {\bf{C}}_{{\bf{x}}_{\alpha \, 1}^2, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, 1})/ \partial y}&
   ... &
   {\bf{C}}_{{\bf{x}}_{\alpha \, i}^2, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, j})/ \partial z}& \\
   \vdots &
   \vdots &
   \ddots &
   \vdots &
   \ddots &
   \vdots \\
   {\bf{C}}_{{\bf{x}}_{\alpha \, 1}^r, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, 1})/ \partial x}&
   {\bf{C}}_{{\bf{x}}_{\alpha \, 2}^r, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, 1})/ \partial x}&
   ... &
   {\bf{C}}_{{\bf{x}}_{\alpha \, 2}^r, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, 1})/ \partial y}&
   ... &
   {\bf{C}}_{{\bf{x}}_{\alpha \, i}^r, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, j})/ \partial z}& \\
   \end{bmatrix}
   
   
As the interfaces are relative to a point
:math:` {\bf{x}}_{\alpha_\, 0}^k` the value of the covariance function:

.. math::

   {\bf{C}}_{{\bf{x}}_{\alpha \, i}^r, \, \partial {\bf{Z}}({\bf{x}}_{\beta \, j})/ \partial x} =
   C_{Z({{\bf{x}}_{\alpha \,i}^r)}, \, \partial Z({\bf{x_{\beta \, j}}})/\partial x} -
   C_{Z({{\bf{x}}_{\alpha \,0}^r)}, \, \partial Z({\bf{x_{\beta \, j}}})/\partial x}


with the covariance of the scalar field being function the vector r,
its directional derivative is analogous to the previous derivations:

.. math::

   {\bf{C}{_{\bf{Z}, \, \partial {\bf{Z}}/ \partial u}}} =
   \frac{\partial C_{\bf{Z}} (r)}{\partial r}
   \frac{\partial r}{\partial h_u} =  - \frac{h_u}{r}C'_Z

Universal matrix
----------------

As the mean value of the scalar field is going to be always unknown, it
needs to be estimated from data itself. The simplest approach is to
consider the mean constant for the whole domain, i.e. ordinary Kriging.
However, in the *scalar field* case we can assume certain drift towards
the direction of the orientations. Therefore, the mean can be written as
function of known basis functions:

.. math:: \mu({\bf{x}}) = \sum^L_{l=0} a_lf^l({\bf{x}})


where *l* is the grade of the polynomials used to describe the drift.
Because of the algebraic dependence of the variables, there is only one
drift and therefore the unbiasedness can be expressed as:

.. math:: {\bf{U}}_{\bf{Z}} \lambda_{1} + {\bf{U}}_{\partial {\bf{Z}}/ \partial u} \lambda_{2} = f_{10}


Consequently, the number of equations are determined according to the
grade of the polynomial and the number of equations forming the
properties matrices equations [C\_i] and [C\_g]:

.. math::

   U_Z =
   \begin{bmatrix}
   x^1_1     - x^1_0 & x^1_2 - x^1_0 & ... &
   x^2_1     - x^2_0 & x^2_2 - x^2_0 & ... &
   x^r_{i-1} - x^r_0 & x^r_i - x^r_0 \\
   y^1_1     - y^1_0 & y^1_2 - y^1_0 & ... &
   y^2_1     - y^2_0 & y^2_2 - y^2_0 & ... &
   y^r_{i-1} - y^r_0 & y^r_i - y^r_0 \\
   z^1_1     - z^1_0 & z^1_2 - z^1_0 & ... &
   z^2_1     - z^2_0 & z^2_2 - z^2_0 & ... &
   z^r_{i-1} - z^r_0 & z^r_i - z^r_0 \\
   x^1_1    x^1_1     - x^1_0x^1_0 & x^1_2 x^1_2 - x^1_0 x^1_0 & ... &
   x^2_1    x^2_1     - x^2_0x^2_0 & x^2_2 x^2_2 - x^2_0 x^2_0 & ... &
   x^r_{i-1}x^r_{i-1} - x^r_0x^r_0 & x^r_i x^r_i - x^r_0 x^r_0 \\
   y^1_1    y^1_1     -
   y^1_0    y^1_0 &
   y^1_2    y^1_2 -
   y^1_0    y^1_0 & ... &
   y^2_1    y^2_1     -
   y^2_0    y^2_0 &
   y^2_2    y^2_2 -
   y^2_0    y^2_0 & ... &
   y^r_{i-1}y^r_{i-1} -
   y^r_0    y^r_0 &
   y^r_i    y^r_i -
   y^r_0    y^r_0 \\
   z^1_1    z^1_1     -
   z^1_0    z^1_0 &
   z^1_2    z^1_2 -
   z^1_0    z^1_0 & ... &
   z^2_1    z^2_1     -
   z^2_0    z^2_0 &
   z^2_2    z^2_2 -
   z^2_0    z^2_0 & ... &
   z^r_{i-1}z^r_{i-1} -
   z^r_0    z^r_0 &
   z^r_i    z^r_i -
   z^r_0    z^r_0 \\
   x^1_1    y^1_1     -
   x^1_0    y^1_0 &
   x^1_2    y^1_2 -
   x^1_0    y^1_0 & ... &
   x^2_1    y^2_1     -
   x^2_0    y^2_0 &
   x^2_2    y^2_2 -
   x^2_0    y^2_0 & ... &
   x^r_{i-1}y^r_{i-1} -
   x^r_0    y^r_0 &
   x^r_i    y^r_i -
   x^r_0    y^r_0 \\
   x^1_1    z^1_1     -
   x^1_0    z^1_0 &
   x^1_2    z^1_2 -
   x^1_0    z^1_0 & ... &
   x^2_1    z^2_1     -
   x^2_0    z^2_0 &
   x^2_2    z^2_2 -
   x^2_0    z^2_0 & ... &
   x^r_{i-1}z^r_{i-1} -
   x^r_0    z^r_0 &
   x^r_i    z^r_i -
   x^r_0    z^r_0 \\
   y^1_1    z^1_1     -
   y^1_0    z^1_0 &
   y^1_2    z^1_2 -
   y^1_0    z^1_0 & ... &
   y^2_1    z^2_1     -
   y^2_0    z^2_0 &
   y^2_2    z^2_2 -
   y^2_0    z^2_0 & ... &
   y^r_{i-1}z^r_{i-1} -
   y^r_0    z^r_0 &
   y^r_i    z^r_i -
   y^r_0    z^r_0 \\
   \end{bmatrix}

.. math::

   \bf{U_{\partial {\bf{Z}}/ \partial u}}  =
   \begin{matrix}
   \begin{matrix}
   {\bf{x}_{\beta 1}} & {\bf{x}_{\beta 2}} & ...& {\bf{x}_{\beta 1}} & {\bf{x}_{\beta 2}}& ...& {\bf{x}_{\beta i-1}} & {\bf{x}_{\beta i}}
   \end{matrix}\\
   \begin{bmatrix}
   1 &
   1 &
   ... &
   0 &
   0 &
   ... &
   0 &
   0\\
   0 &
   0 &
   ... &
   1 &
   1 &
   ... &
   0 &
   0\\
   0 &
   0 &
   ... &
   0 &
   0 &
   ... &
   1 &
   1\\
   2x_1 &
   2x_2 &
   ... &
   0 &
   0 &
   ... &
   0 &
   0\\
   0 &
   0 &
   ... &
   2y_1 &
   2y_2 &
   ... &
   0 &
   0\\
   0 &
   0 &
   ... &
   0 &
   0 &
   ... &
   2z_{i-1} &
   2z_i\\
   y_1 &
   y_2 &
   ... &
   x_1 &
   x_2 &
   ... &
   0 &
   0\\
   y_1 &
   y_2 &
   ... &
   0 &
   0 &
   ... &
   x_{i-1} &
   x_i\\
   0 &
   0 &
   ... &
   z_1 &
   z_2 &
   ... &
   x_{i-1} &
   x_i\\
   \end{bmatrix} &
   \begin{aligned}[l]
   &\left\}\begin{matrix}
   \partial {\bf{x}_{\beta i}}/\partial x \\
   \partial {\bf{x}_{\beta i}}/\partial y \\
   \partial {\bf{x}_{\beta i}}/\partial z \\
   \partial^2 {\bf{x}_{\beta i}}/\partial x^2 \\
   \partial^2 {\bf{x}_{\beta i}}/\partial y^2 \\
   \partial^2 {\bf{x}_{\beta i}}/\partial z^2 \\
   \partial^2 {\bf{x}_{\beta i}}/\partial x \partial y\\
   \partial^2 {\bf{x}_{\beta i}}/\partial x \partial z\\
   \partial^2 {\bf{x}_{\beta i}}/\partial y \partial z\\
   \end{matrix}\right.\end{aligned}
   \end{matrix}

Kriging Estimator
-----------------

In normal Kriging the right hand term of the Kriging system (Eq.
[krig\_sys]) corresponds to covariances and drift matrices of dimensions
:math:`m \times n` where *m* is the number of elements of the data
sets—either :math:`{\bf{x}}_\alpha` or :math:`{\bf{x}}_\beta`—and *n*
the number of locations where the interpolation is performed,
:math:`{\bf{x}}_0`.


Since, in this case, the parameters of the variogram functions are
arbitrarily chosen, the Kriging variance does not hold any physical
information of the domain. As a result of this, being interested only in
the mean value, we can solve the Kriging system in the dual form
:cite:`d-matheron1981splines`:

.. math::

   Z({\bf{x}}_0)=
   \begin{bmatrix}
   a'_{{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}  &
   b'_{{\bf{Z}, \,\bf{Z}}} &
   c'
   \end{bmatrix}
   \left[ \begin{array}{cc}
   {\bf{c_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}} & {\bf{c_{\partial {\bf{Z}}/ \partial u, \, Z}}} \\
   {\bf{c_{Z, \,\partial {\bf{Z}}/ \partial u}}} &  {\bf{c_{\bf{Z}, \,\bf{Z}}}} \\
   {\bf{f_{10}}} & {\bf{f_{20}}} \end{array} \right]


where:

.. math::

   \begin{bmatrix}
   a_{{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}  \\
   b_{{\bf{Z}, \,\bf{Z}}} \\
   c
   \end{bmatrix} =
   \begin{bmatrix}
   \partial {\bf{Z}}  \\ 0 \\ 0
   \end{bmatrix}
   \left[ \begin{array}{ccc}
   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}} &
   {\bf{C_{\partial {\bf{Z}}/ \partial u, \, Z}}} &
   \bf{U_{\partial {\bf{Z}}/ \partial u}} \\
   {\bf{C_{Z, \, \partial {\bf{Z}}/ \partial u }}} &
   {\bf{C_{\bf{Z}, \, \bf{Z}}}} &
   {\bf{U_{Z}}} \\
   \bf{U'_{\partial {\bf{Z}}/ \partial u}} &
   {\bf{U'_{Z}}} &
   {\bf{0}} \end{array} \right]^{-1}


noticing that the 0 on the second row appears due to we are
interpolation the difference of scalar fields instead the scalar field
itself [eq\_rel].

Example of covariance function: cubic
-------------------------------------

The choice of the covariance function will govern the shape of the
iso-surfaces of the scalar field. As opposed to other Kriging uses, here
the choice cannot be based on empirical measurements. Therefore, the
choice of the covariance function is merely arbitrary trying to mimic as
far as possible coherent geological structures.

.. figure:: figs/covariance.png
   :alt: 

   Representation of a cubic variogram and covariance for an arbitrary
   range and nugget effect.

The main requirement to take into consideration when the time comes to
choose a covariance function is that it has to be twice differentiable,
:math:`h^2` in origin to be able to calculate
:math:`{\bf{C_{\partial {\bf{Z}}/ \partial u, \, \partial {\bf{Z}}/ \partial v}}}`
as we saw in equation [huhv]. The use of a Gaussian model
:math:`C(r) = \exp{-(r/a)^2}` and the non-divergent spline
:math:`C(r) = r^4
Log(r)` and their correspondent flaws are explored in
:cite:`d-Lajaunie.1997`.


The most widely used function in the potential field method is the cubic
covariance due to mathematical robustness and its coherent geological
description of the space.

.. math::

   C(r) = \begin{cases}
   C_0(1-7(\frac{r}{a})^2+ \frac{35}{4}(\frac{r}{a})^3
   - \frac{7}{2}(\frac{r}{a})^5 +\frac{3}{4}(\frac{r}{a})^7) &
   \text{for } 0  \leq r \leq a \\
   0 & \text{for } r  \geq a
   \end{cases}


with :math:`a` being the range and :math:`C_0` the variance of the data.
The value of :math:`a` determine the maximum distance that a data point
influence another. Since, we assume that all data belong to the same
depositional phase it is recommended to choose values close to the
maximum extent to interpolate in order to avoid mathematical artifacts.
for the values of the covariance at 0 and nuggets effects so far only
*ad hoc* values have been used so far. It is important to notice that
the only effect that the values of the covariance in the potential-field
method has it is to weight the relative influence of both CoKriging
parameters (interfaces and orientations) since te absolut value of the
field is meaningless. Regarding the nugget effect, the authors
recommendation is to use fairly small nugget effects to give stability
to the computation—since we normally use the kriging mean it should not
have further impact to the result.

Example of Probabilistic Graphical Model
----------------------------------------

Here we can see the probabilistic graphical model of the Bayesian
inference of Section [sec:geol-invers-prob]:

.. figure:: figs/paper_graph.png
   :alt: 

   Probabilistic graphical model generated with pymc2. Ellipses
   represent stochastic parameters, while triangles are deterministic
   functions that return intermediated states of the probabilistic model
   such as the GemPy model


.. bibliography:: small.bib
   :cited:
   :labelprefix: D
   :keyprefix: d-
   :style: unsrt