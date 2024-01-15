import matplotlib.pylab as plt
import numpy as np


def plot_data(idnt, figure=None, add_text="", path=None):
    if figure is None:
        figure = plt.figure(figsize=(10, 7))
    else:
        figure.clear()
    ax = figure.add_axes([.07, .1, .90, .87])
    ax.set_xlabel("tip position [µm]")
    ax.set_ylabel("force [nN]")
    # overview plot
    ax.plot(idnt["tip position"] * 1e6, idnt["fit"] * 1e9)
    ax.plot(idnt["tip position"] * 1e6, idnt["force"] * 1e9)
    # inset with indentation
    axin = figure.add_axes([.3, .3, .65, .65])
    axin.patch.set_alpha(0.5)
    # inset limits depend on contact point
    axin.plot(idnt["tip position"] * 1e6, idnt["fit"] * 1e9, label="fit")
    axin.plot(idnt["tip position"] * 1e6, idnt["force"] * 1e9, label="data")
    axin.legend(loc="upper right")
    axin.grid()
    cp = idnt.fit_properties["params_fitted"]["contact_point"].value * 1e6
    xmin = idnt["tip position"].min() * 1e6
    dx = np.abs(cp - xmin)
    ymin = idnt["force"][idnt["segment"] == 0].min() * 1e9
    ymax = idnt["force"][idnt["segment"] == 0].max() * 1e9
    dy = ymax - ymin
    axin.set_xlim(xmin - dx/7, cp + dx/2)
    axin.set_ylim(ymin - dy/7, ymax + dy/7)
    axin.set_xticks([])
    axin.set_ylabel("close-up")
    # inset with residuals
    axin2 = figure.add_axes([.3, .15, .65, .14])
    axin2.patch.set_alpha(0.5)
    axin2.plot(idnt["tip position"] * 1e6,
               idnt["fit residuals"] * 1e9, label="fit")
    axin2.set_xlim(xmin - dx/7, cp + dx/2)
    axin2.set_ylabel("fit residuals")
    axin2.axhline(0, color="gray")
    axin2.set_ylim(-dy/10, dy/10)
    # display fitted parameters
    text = "Fit parameters:\n"
    text += "model: {}\n".format(idnt.fit_properties["model_key"])
    params = idnt.fit_properties["params_fitted"]
    for p in params:
        text += "{}={:.2e}\n".format(p, params[p].value)
    text += "\n\n" + add_text
    axin.text(cp, ymax, text, horizontalalignment="right",
              verticalalignment="top")

    if path is not None:
        figure.savefig(path)
        plt.close()
