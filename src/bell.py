import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src import utils


def get_p_B(time_eta_df):
    return time_eta_df.mul(time_eta_df[0], axis='index').mean(axis=0)


def get_p_0(time_eta_df):
    return (1 - time_eta_df).mul((1 - time_eta_df[0]), axis='index').mean(axis=0)


def get_p_1(time_eta_df):
    return (time_eta_df.mul(1 - time_eta_df[0], axis='index') +
            (1 - time_eta_df).mul(time_eta_df[0], axis='index')).mean(axis=0)


def ideal_B(p_B, p_0, p_1, eta_c, nu):
    numer = 2 * np.sqrt(2) * p_B * eta_c**2 * np.exp(2 * nu)
    denom1 = p_B * (np.exp(2 * nu) + eta_c - 1)**2
    denom2 = p_0 * (np.exp(2 * nu) - 1)**2
    denom3 = p_1 * (np.exp(2 * nu) - 1) * (np.exp(2 * nu) + eta_c - 1)
    return numer / (denom1 + denom2 + denom3)


def P(same_or_diff, theta_A, theta_B, nu, xi, eta_c, eta_A, eta_B):
    eta_A = np.asarray(eta_A)
    eta_B = np.asarray(eta_B)

    def C_same_diff(same_or_diff):
        _sin_cos = np.sin(theta_A - theta_B) if same_or_diff == 'same' else np.cos(theta_A - theta_B)
        return (eta_c**2 * eta_A * eta_B * np.tanh(xi)**2 * (1 - np.tanh(xi)**2)**2 *
                ((1 - eta_c * eta_A) * (1 - eta_c * eta_B) * np.tanh(xi)**2 - _sin_cos**2))

    def C_1AB(A_or_B):
        eta_1 = eta_A if A_or_B == 'A' else eta_B
        eta_2 = eta_B if A_or_B == 'A' else eta_A
        return (eta_c * eta_2 * (1 - eta_c * eta_1) *
                (1 - np.tanh(xi)**2) * np.tanh(xi)**2 *
                (eta_c**2 * eta_A * eta_B * np.tanh(xi)**2 -
                 (1 + (eta_c * eta_A - 1) * np.tanh(xi)**2) *
                  (1 + (eta_c * eta_B - 1) * np.tanh(xi)**2)))

    C_0 = (eta_c**2 * eta_A * eta_B * np.tanh(xi)**2 -
           (1 + (eta_c * eta_A - 1) * np.tanh(xi)**2) *
           (1 + (eta_c * eta_B - 1) * np.tanh(xi)**2))**2
    C_1A = C_1AB('A')
    C_1B = C_1AB('B')
    C_i = C_same_diff('same') if same_or_diff == 'same' else C_same_diff('diff')
    C_j = C_same_diff('diff') if same_or_diff == 'same' else C_same_diff('same')

    AVG_1 = (1 / (C_0 + C_1A + C_1B + C_i)).mean()
    AVG_2 = (C_0 / (C_0 + C_1A)**2).mean()
    AVG_3 = (C_0 / (C_0 + C_1B)**2).mean()
    AVG_4 = (1 / (C_0 + C_1A + C_1B + C_j)).mean()
    AVG_5 = (1 / C_0).mean()
    P_i = 1 / 2 + np.exp(-4 * nu) / 2 * (1 - np.tanh(xi)**2)**4 * \
        (np.exp(2 * nu) * (2 * AVG_1 - AVG_2 - AVG_3 - 2 * AVG_4) + AVG_5)
    return P_i


def E(theta_A, theta_B, nu, xi, eta_c, eta_A, eta_B):
    P_same = P('same', theta_A, theta_B, nu, xi, eta_c, eta_A, eta_B)
    P_diff = P('diff', theta_A, theta_B, nu, xi, eta_c, eta_A, eta_B)
    return (P_same - P_diff) / (P_same + P_diff)


def get_B(theta_A1, theta_B1, theta_A2, theta_B2, nu, xi, eta_c, eta_A, eta_B):
    E_11 = E(theta_A1, theta_B1, nu, xi, eta_c, eta_A, eta_B)
    E_12 = E(theta_A1, theta_B2, nu, xi, eta_c, eta_A, eta_B)
    E_22 = E(theta_A2, theta_B2, nu, xi, eta_c, eta_A, eta_B)
    E_21 = E(theta_A2, theta_B1, nu, xi, eta_c, eta_A, eta_B)
    return np.abs(E_11 - E_12) + np.abs(E_22 + E_21)


def smooth(x, y, smooth, num=100):
    x = np.asarray(x)
    y = np.asarray(y)

    nan_mask = np.isnan(x)
    x = x[~nan_mask]
    y = y[~nan_mask]

    t = np.linspace(x[0], x[-1], num=num)
    values = interp1d(x, y)(t)
    return t, gaussian_filter1d(values, smooth)


class BellCalculator():
    def __init__(self, **kwargs):
        default_params = {
            'channel_name': 'strong',
            'aperture_radius': 0.2,
            'channel_length_km': 50,
            'wind_speed': 10,
            'theta': (0, np.pi / 8, np.pi / 4, 3 * np.pi / 8),
            'nu': 3e-4,
            'eta_c': 1/2 * 0.85,
            'eta_writing': 0.85,
            'db_p_ms': 3,
            '_xi': np.linspace(0.01, 1.2, 100),
        }
        self.params = {**default_params, **kwargs}
    
    def eta_d(self, length_km): 
        return 10**(-0.1 * length_km / 10)
    
    def eta_delay(self, time, db_p_ms): 
        return 10**(-db_p_ms * (time * 1000) / 10)
        
    def get(self, **kwargs):
        params = {**self.params, **kwargs}

        df = utils.load_data(params['channel_name'], params['aperture_radius'])
        eta_d = self.eta_d(params['channel_length_km'])
        eta_memory = lambda s: (1 if s == 0 else params['eta_writing']) * self.eta_delay(s / params['wind_speed'], params['db_p_ms'])
        df *= eta_d * pd.Series([eta_memory(s) for s in df.columns], df.columns)
    
        eta_mean = df.mean()
        eta2_mean = (df**2).mean()
        etacorr_mean = (df.mul(df[0], axis='index')).mean()
        p_B, p_0, p_1 = get_p_B(df), get_p_0(df), get_p_1(df)
        max_B = [np.max([get_B(*params['theta'], params['nu'], xi, params['eta_c'], df[0], df[t]) for xi in params['_xi']])
            for t in df.columns]
        B_Bell = [ideal_B(p_B[t], p_0[t], p_1[t], params['eta_c'], params['nu']) for t in df.columns]
        
        return {'wind_shift': np.asarray(df.columns), 'max_B': np.asarray(max_B), 'B_Bell': np.asarray(B_Bell)}


class BellPlot():
    def __init__(self, ax=None):
        self.ax = ax or plt.subplots(1, 1, figsize=(4, 3))[1]
        self.dim_scale = (100, 1)
        self.ax.set_ylabel(r'Bell parameter $\mathcal{B}$')
        self.ax.set_xlabel('Wind-driven shift $s$ (cm)')
        self.line_color_id = 0
        self.ax.axhline(y = 2, c='k', alpha=0.5, lw=0.5, antialiased=None, snap=True)
        self.axin = None
    
    def plot(self, bell_data, smooth_value=1, pdc_kwargs=None, bell_kwargs=None, **kwargs):
        scaled_x = self.dim_scale[0] * bell_data['wind_shift']
        _pdc_kwargs = {'color': utils.LINE_COLORS[self.line_color_id], **kwargs, **(pdc_kwargs or {})}
        self.ax.plot(*smooth(scaled_x, bell_data['max_B'], smooth_value), **_pdc_kwargs)
        _bell_kwargs = {'color': utils.LINE_COLORS[self.line_color_id], **kwargs, **(bell_kwargs or {})}
        self.ax.plot(*smooth(scaled_x, bell_data['B_Bell'], smooth_value), **_bell_kwargs)
        self.line_color_id += 1

    def _example(self, bell_calculator, time, xi_max=1.2):
        params = bell_calculator.params
        _xi = np.linspace(0, xi_max, 100)
        data = utils.load_data(params['channel_name'], params['aperture_radius'])
        return {'xi': _xi, 'B': [get_B(*params['theta'], params['nu'], xi, params['eta_c'], data[0], data[time]) for xi in _xi]}

    def plot_example(self, bell_calculator, time, xi_max=1.2):
        data = self._example(bell_calculator, time, xi_max=1.2)
        axin = self.ax.inset_axes([0.62, 0.72, 0.35, 0.25])
        self.axin = axin
        axin.plot([data['xi'][0], data['xi'][-1]], [2, 2], c='k', alpha=0.5, lw=0.5, antialiased=None, snap=True)
        axin.plot(data['xi'], data['B'], c=utils.LINE_MAIN_COLOR)
        axin.scatter(data['xi'][np.argmax(data['B'])], np.max(data['B']), c='k', s=12, zorder=10)
        axin.hlines(np.max(data['B']), 0, data['xi'][np.argmax(data['B'])], color='k', ls=':', lw=0.7)
        axin.set_ylim(1, 2.8)
        axin.set_xlim(0, 1)
        axin.tick_params(axis='both', which='major', labelsize='x-small')
        axin.set_xlabel(r'Squeezing parameter $\xi$', fontsize='x-small')
        axin.set_ylabel('Bell\n parameter $\mathcal{B}$', fontsize='x-small')
        axin.xaxis.set_label_coords(0.5, -0.33)
        axin.text(-0.18, 2.35, r'$\mathcal{B}_\mathrm{m}$', fontsize='x-small')

    def savefig(self, file_path, **kwargs):
        fig = self.ax.get_figure()
        fig.tight_layout()
        kwargs = {**utils.SAVE_KWARGS, **kwargs}
        fig.savefig(file_path, **kwargs)


class BellPlotTime(BellPlot):
    def __init__(self, ax=None):
        super().__init__(ax)
        self.dim_scale = (1000, 1)
        self.ax.set_xlabel('Time $\\tau$ (ms)')
    
    def plot(self, bell_data, wind_speed, smooth_value=1, **kwargs):
        time = bell_data['wind_shift'] / wind_speed
        super().plot({**bell_data, 'wind_shift': time}, smooth_value, **kwargs)
