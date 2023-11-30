import numpy as np
from matplotlib import pyplot as plt
import scipy

from src import config, utils


def plot():
    dim_scale = 100
    y_line = np.exp(-1)
    try:
        utils.load_data(aperture_size=0.1)
        x = [0.004, 0.006, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    except FileNotFoundError:
        x = [0.006, 0.02, 0.2, 0.3]
    s = []
    _, ax = plt.subplots(1, 1, figsize=(4, 3))

    for i, a in enumerate(x):
        df = utils.load_data(a)
        pear = utils.pearson_df(df)
        s.append(dim_scale * utils.get_intersect(df.columns, pear, y_line))
    plt.plot(dim_scale * np.asarray(x), s, markersize=3, marker='o',
             color=config.LINE_COLORS[1])
    plt.ylim(0, plt.ylim()[1])
    plt.xlim(left=0)
    plt.xlabel(r'Aperture radius $R_\mathrm{ap}$ [cm]')
    plt.ylabel(r"Spatial coherence radius $\rho_0$ [cm]")

    smooth = 1
    try:
        df = utils.load_data(aperture_size=0.1)
    except FileNotFoundError:
        df = utils.load_data(aperture_size=0.2)
    axin = plt.gca().inset_axes([0.48, 0.21, 0.45, 0.35])
    eta_pearson = utils.pearson_df(df)

    t = np.linspace(0, df.columns[-1], 100)
    values = scipy.interpolate.interp1d(df.columns, eta_pearson)(t)
    axin.plot(dim_scale * t, scipy.ndimage.gaussian_filter1d(values, smooth),
              c=config.LINE_MAIN_COLOR)
    corrr_length = dim_scale * utils.get_intersect(df.columns, eta_pearson, y_line)
    axin.hlines(y_line, 0, corrr_length, color='k', ls=':')
    axin.vlines(corrr_length, 0, y_line, color='k', ls=':')
    axin.scatter(corrr_length, y_line, s=10, color='k')
    axin.text(0.1, y_line + 0.03, r"$e^{-1}$")

    axin.set_ylabel("Pearson\ncorrelation")
    axin.set_xlabel(r"Wind-driven shift $s$ [cm]")
    axin.set_ylim(0, 1.05)
    axin.set_xlim(0, axin.get_xlim()[1])
    axin.text(4.9, 0.06, r'$\rho_0$')

    plt.savefig("plots/3_corr_length.pdf", **config.SAVE_KWARGS)
