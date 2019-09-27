r"""Approximating the Hertzian model with a spherical indenter

There is no closed form for the Hertzian model with a spherical indenter.
The force :math:`F` does not directly depend on the indentation depth
:math:`\delta`, but has an indirect dependency via the radius of the circular
contact area between indenter and sample :math:`a` :cite:`Sneddon1965`:

.. math::

    F &= \frac{E}{1-\nu^2} \left( \frac{R^2+a^2}{2} \ln \! \left(
         \frac{R+a}{R-a}\right) -aR  \right) \label{eq:1}\\
    \delta &= \frac{a}{2} \ln
         \! \left(\frac{R+a}{R-a}\right) \label{eq:2}

Here, :math:`E` is the Young's modulus, :math:`R` is the radius of the
indenter, and :math:`\nu` is the Poisson's ratio of the probed material.

Because of this indirect dependency, fitting this model to experimental
data can be time-consuming. Therefore, it is beneficial to approximate this
model with a polynomial function around small values of :math:`\delta/R`
using the Hertz model for a parabolic indenter as a starting point
:cite:`Dobler`:

.. math::

    F = \frac{4}{3} \frac{E}{1-\nu^2} \sqrt{R} \delta^{3/2}
        \left(1
         - \frac{1}{10} \frac{\delta}{R}
         - \frac{1}{840} \left(\frac{\delta}{R}\right)^2
         + \frac{11}{15120} \left(\frac{\delta}{R}\right)^3
         + \frac{1357}{6652800} \left(\frac{\delta}{R}\right)^4
         \right)

This example illustrates the error made with this approach. In nanite, the
model for a spherical indenter has the identifier
:ref:`"sneddon_spher" <sec_ref_model_sneddon_spher>` and the
approximate model has the identifier
:ref:`"sneddon_spher_approx" <sec_ref_model_sneddon_spher_approx>`.

The plot shows the error for the parabolic indenter model
:ref:`"hertz_para" <sec_ref_model_hertz_para>` and for the
approximation to the spherical indenter model.
The maximum indentation depth is set to :math:`R`.
The error made by the approximation of the spherical indenter is more than
four magnitudes lower than the maximum force during indentation.
"""
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np

from nanite.model import models_available

# models
exact = models_available["sneddon_spher"]
approx = models_available["sneddon_spher_approx"]
para = models_available["hertz_para"]
# parameters
params = exact.get_parameter_defaults()
params["E"].value = 1000

# radii
radii = np.linspace(2e-6, 100e-6, 20)

# plot results
plt.figure(figsize=(8, 5))

# overview plot
ax = plt.subplot()
for ii, rad in enumerate(radii):
    params["R"].value = rad
    # indentation range
    x = np.linspace(0, -rad, 300)
    yex = exact.model(params, x)
    yap = approx.model(params, x)
    ypa = para.model(params, x)
    ax.plot(x*1e6, np.abs(yex - yap)/yex.max(),
            color=mpl.cm.get_cmap("viridis")(ii/radii.size),
            zorder=2)
    ax.plot(x*1e6, np.abs(yex - ypa)/yex.max(), ls="--",
            color=mpl.cm.get_cmap("viridis")(ii/radii.size),
            zorder=1)

ax.set_xlabel(r"indentation depth $\delta$ [µm]")
ax.set_ylabel("error in force relative to maximum $F/F_{max}$")
ax.set_yscale("log")
ax.grid()

# legend
custom_lines = [Line2D([0], [0], color="k", ls="--"),
                Line2D([0], [0], color="k", ls="-"),
                ]
ax.legend(custom_lines, ['parabolic indenter',
                         'approximation of spherical indenter'])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)

norm = mpl.colors.Normalize(vmin=radii[0]*1e6, vmax=radii[-1]*1e6)
mpl.colorbar.ColorbarBase(ax=cax,
                          cmap=mpl.cm.viridis,
                          norm=norm,
                          orientation='vertical',
                          label="indenter radius [µm]"
                          )

plt.tight_layout()
plt.show()
