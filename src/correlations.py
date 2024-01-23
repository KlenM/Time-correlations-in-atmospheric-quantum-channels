import numpy as np
import scipy
from matplotlib import pyplot as plt 

from src import utils


def get_correlations(channel_name, y_line=np.exp(-1)):
    apertures = utils.get_apertures(channel_name)
    spatial_coherence = []
    for a in apertures:
        df = utils.load_data(channel_name, a)
        pear = utils.pearson_df(df)
        spatial_coherence.append(utils.get_intersect(df.columns, pear, y_line))
    return apertures, spatial_coherence


class SpatialCoherencePlot():
    def __init__(self, dim_scale=(100, 100)):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_xlabel(r'Aperture radius $R_\mathrm{ap}$ (cm)')
        ax.set_ylabel(r"Spatial coherence radius $\rho_0$ (cm)")
        self.dim_scale = dim_scale
        self.fig = fig
        self.ax = ax
        self.line_color_id = 1

    def plot(self, channel_name, **kwargs):
        apertures, spatial_coherence = get_correlations(channel_name)
        scaled_apertures = self.dim_scale[0] * np.asarray(apertures)
        scaled_spatial_coherence = self.dim_scale[1] * np.asarray(spatial_coherence)
        kwargs = {'color': utils.LINE_COLORS[self.line_color_id], 'marker': 'o', 'markersize': 3, **kwargs}
        self.ax.plot(scaled_apertures, scaled_spatial_coherence, label=channel_name, **kwargs)
        self.line_color_id += 1

    def plot_example(self, channel_name, aperture_size, pos=(0.51, 0.16, 0.47, 0.4)):
        smooth = 1
        axin = self.ax.inset_axes(pos)
        df = utils.load_data(channel_name, aperture_size)
        eta_pearson = utils.pearson_df(df)
    
        t = np.linspace(0, df.columns[-1], 100)
        values = scipy.interpolate.interp1d(df.columns, eta_pearson)(t)
        axin.plot(self.dim_scale[0] * t, scipy.ndimage.gaussian_filter1d(values, smooth), c=utils.LINE_MAIN_COLOR)
        y_line = np.exp(-1)
        corrr_length = self.dim_scale[0] * utils.get_intersect(df.columns, eta_pearson, y_line)
        axin.hlines(y_line, 0, corrr_length, color='k', ls=':')
        axin.vlines(corrr_length, 0, y_line, color='k', ls=':')
        axin.scatter(corrr_length, y_line, s=10, color='k', zorder=10)
        axin.text(0.2, y_line + 0.03, r"$e^{-1}$", fontsize='x-small')
    
        axin.set_ylabel("Pearson\ncorrelation", fontsize='x-small')
        axin.set_xlabel(r"Wind-driven shift $s$ (cm)", fontsize='x-small')
        axin.set_ylim(0, 1.05)
        axin.set_xlim(0, axin.get_xlim()[1])
        axin.tick_params(axis='both', which='major', labelsize='x-small')
        axin.xaxis.set_label_coords(0.5, -0.22)
        axin.yaxis.set_label_coords(-0.18, 0.5)
        axin.text(9.2, 0.06, r'$\rho_0$', fontsize='x-small')
        self.axin = axin

    def savefig(self, file_path, **kwargs):
        self.fig.tight_layout()
        kwargs = {**utils.SAVE_KWARGS, **kwargs}
        self.fig.savefig(file_path, **kwargs)
        