import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import scipy

from src import utils
from src.correlations import get_correlations


def witnessess(etaa_mean, etab_mean, sqrt_corr, r):
    return ((1 - etaa_mean + etaa_mean * np.cosh(r)**2) * (1 - etab_mean + etab_mean * np.cosh(r)**2) *
            (-sqrt_corr**2 * np.cosh(r)**2 * np.sinh(r)**2 + etaa_mean * etab_mean * np.sinh(r)**4) -
            sqrt_corr * np.cosh(r) * np.sinh(r) *
            (-sqrt_corr**3 * np.cosh(r)**3 * np.sinh(r)**3 +
             sqrt_corr * etaa_mean * etab_mean * np.cosh(r) * np.sinh(r)**5))


def delayed_witnessess(etaa_mean, etab_mean, sqrt_corr, r, t, eta_writing=0.85, db_p_ms=3):
    eta_delay = lambda t: 10**(-db_p_ms * (t * 1000) / 10)
    etab_mean *= eta_writing * eta_delay(t)
    sqrt_corr_value = np.sqrt(eta_writing * eta_delay(t)) * sqrt_corr(t)
    return witnessess(etaa_mean, etab_mean, sqrt_corr_value, r)


def _gaussian_entanglement_threshold(channel_name, aperture_size, r_squeezing):
    df = utils.load_data(channel_name, aperture_size)
    eta_mean = df.mean().mean()
    sqrt_corr_data = np.sqrt(df).mul(np.sqrt(df[0]), axis='index').mean(axis='index')
    sqrt_corr = scipy.interpolate.Akima1DInterpolator(df.columns, sqrt_corr_data)
    res = scipy.optimize.fsolve(lambda t: delayed_witnessess(eta_mean, eta_mean, sqrt_corr, r_squeezing, t), 
                                 x0=1e-4, xtol=1e-4, full_output=True)
    if res[1]['fvec'] < 0:
        if np.abs(res[0][0] / df.columns[-1]) < 0.98:
            return res[0][0]
        else: 
            return np.nan
    else:
        return 0


def gaussian_entanglement_threshold(channel_name, aperture_size, r_squeezing):
    return np.asarray([_gaussian_entanglement_threshold(channel_name, aperture_size, r) for r in r_squeezing])


class GaussianEntanglementPlot():
    def __init__(self, dim_scale=(100, 1)):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_xlabel(r'Wind-driven shift $s$ (cm)')
        ax.set_ylabel(r"Squeezing parameter $\xi$")
        self.dim_scale = dim_scale
        self.fig = fig
        self.ax = ax
        self.line_color_id = 0

    def plot(self, channel_name, aperture_radius, fill_kwargs=None, **kwargs):
        r_squeezing = np.linspace(1, 3, 300)
        threshold = gaussian_entanglement_threshold(channel_name, aperture_radius, r_squeezing)
        scaled_data = self.dim_scale[0] * np.asarray(threshold)
        scaled_r = self.dim_scale[1] * r_squeezing

        kwargs = {'color': utils.LINE_COLORS[self.line_color_id], **kwargs}
        self.ax.plot(scaled_data, r_squeezing, label=channel_name, **kwargs)
        if fill_kwargs:
            # matplotlib bugfix
            fill_e_kwargs = {**fill_kwargs, 'edgecolor': fill_kwargs['color'], 'color': 'None'}
            self.ax.fill_between([*scaled_data, 0], 0, [*r_squeezing, 10], **fill_e_kwargs)
            fill_kwargs = {**fill_kwargs, 'hatch': None}
            self.ax.fill_between([*scaled_data, 0], 0, [*r_squeezing, 10], **fill_kwargs)
        
        self.line_color_id += 1
        if self.line_color_id >= len(utils.LINE_COLORS):
            self.line_color_id = 0

    def annotate(self, pos=((0.2, 0.3), (0.6, 0.6))):
        t = self.ax.annotate(r'$\mathcal{W} > 0$', xy=pos[0], xycoords='axes fraction')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='none'))
        self.ax.annotate(r'$\mathcal{W} < 0$', xy=pos[1], xycoords='axes fraction')

    def savefig(self, file_path, **kwargs):
        self.fig.tight_layout()
        kwargs = {**utils.SAVE_KWARGS, **kwargs}
        self.fig.savefig(file_path, **kwargs)


class WitnessCoherenceCalculator:
    def __init__(self, channels=('strong_d2', 'strong', 'strong_x1_5'), squeezing=(1.8, 2)):
        self.channels = channels
        self.squeezing = squeezing

    def get(self):
        result = []
        for channel_name in self.channels:
            data = {'channel_name': channel_name}
            data['aperture_radii'], data['spatial_coherence_radii'] = get_correlations(channel_name)
            
            max_wind_shift = np.asarray([gaussian_entanglement_threshold(channel_name, a, self.squeezing).tolist() for a in data['aperture_radii']]).T.tolist()
            data[f'max_wind_shift_for_xi'] = [{'xi': xi, 'max_wind_shift': value} for xi, value in zip(self.squeezing, max_wind_shift)]
            result.append(data)
        return result


class WitnessCoherencePlot:
    def __init__(self, dim_scale=(100, 100)):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_xlabel(r"Spatial coherence radius $\rho_0$ (cm)")
        ax.set_ylabel(r'Wind-driven shift threshold $s_\mathrm{th}$ (cm)')
        self.dim_scale = dim_scale
        self.fig = fig
        self.ax = ax
        self.line_color_id = 0

    def plot(self, data, channel_name, per_xi_kwargs=None, **kwargs):
        rho0, sm_data = [(d['spatial_coherence_radii'], d['max_wind_shift_for_xi']) 
                    for d in data if d['channel_name'] == channel_name][0]
        scaled_rho0 = self.dim_scale[0] * np.asarray(rho0)
        for i, xi_data in enumerate(sm_data):
            xi, sm = xi_data['xi'], xi_data['max_wind_shift']
            scaled_sm = self.dim_scale[1] * np.asarray(sm)
            kwargs = {'color': utils.LINE_COLORS[self.line_color_id], **kwargs, **(per_xi_kwargs[i] if per_xi_kwargs else {})}
            sm_interp = scipy.interpolate.PchipInterpolator(
                scaled_rho0[~np.isnan(scaled_sm)], scaled_sm[~np.isnan(scaled_sm)], extrapolate=False)
            _x = np.linspace(scaled_rho0[0], scaled_rho0[-1])
            self.ax.plot(_x, sm_interp(_x), **kwargs)
            self.ax.scatter(scaled_rho0, scaled_sm, s=15, zorder=10, **{**kwargs, 'ls': '-'})
            self.line_color_id += 1
            if self.line_color_id >= len(utils.LINE_COLORS):
                self.line_color_id = 0

    def ellipse_annotate(self, xy, wh, text, textpos, **kwargs):
        ellipse = Ellipse(xy=xy, width=wh[0], height=wh[1], edgecolor='k', fc='None')
        self.ax.add_patch(ellipse)
        arrowprops = {'arrowstyle': "->", 'color': 'k', 'shrinkA': 5, 'shrinkB': 1,
                          'patchA': None, 'patchB': None, 'connectionstyle': "arc3,rad=0.2", **kwargs}
        self.ax.annotate(text, xy, textpos, arrowprops=arrowprops)

    def savefig(self, file_path, **kwargs):
        self.fig.tight_layout()
        kwargs = {**utils.SAVE_KWARGS, **kwargs}
        self.fig.savefig(file_path, **kwargs)

        