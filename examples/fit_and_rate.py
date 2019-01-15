"""Fitting and rating

This example uses a force-distance curve of a zebrafish spinal cord
section to illustrate basic data fitting and rating with nanite.
The dataset is part of a study on spinal cord stiffness in zebrafish
:cite:`Moellmert2019`.
"""
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt

import nanite

# load the data
group = nanite.load_group("data/zebrafish-head-section-gray-matter.jpk-force")
idnt = group[0]  # this is an instance of `nanite.Indentation`
# apply preprocessing
idnt.apply_preprocessing(["compute_tip_position",
                          "correct_force_offset",
                          "correct_tip_offset"])
# set the fit model ("sneddon_spher_approx" is faster than "sneddon_spher"
# and sufficiently accurate)
idnt.fit_properties["model_key"] = "sneddon_spher_approx"
# get the initial fit parameters
params = idnt.get_initial_fit_parameters()
# set the correct indenter radius
params["R"].value = 18.64e-06
# perform the fit with the edited parameters
idnt.fit_model(params_initial=params)
# obtain the Young's modulus
emod = idnt.fit_properties["params_fitted"]["E"].value
# obtain a rating for the dataset
# (using default regressor and training set)
rate = idnt.rate_quality()

# overview plot
plt.figure(figsize=(8, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

# only plot the approach part (`1` would be retract)
where_approach = idnt["segment"] == 0

# plot force-distance data (nanite uses SI units)
ax1.plot(idnt["tip position"][where_approach] * 1e6,
         idnt["force"][where_approach] * 1e9,
         label="data")
ax1.plot(idnt["tip position"][where_approach] * 1e6,
         idnt["fit"][where_approach] * 1e9,
         label="fit (spherical indenter)")
ax1.text(.2, 2.05,
         "apparent Young's modulus: {:.0f} Pa\n".format(emod)
         + "rating: {:.1f}".format(rate),
         ha="center")
ax1.legend()
# plot resiudals
ax2.plot(idnt["tip position"][where_approach] * 1e6,
         (idnt["force"] - idnt["fit"])[where_approach] * 1e9)

# update plot parameters
ax1.set_xlim(-4.5, 3)
ax1.set_ylabel("force [pN]")
ax1.grid()
ax2.set_xlim(-4.5, 3)
ax2.set_ylim(-.2, .2)
ax2.set_ylabel("residuals [pN]")
ax2.set_xlabel("tip position [µm]")
ax2.grid()

plt.tight_layout()
plt.show()
