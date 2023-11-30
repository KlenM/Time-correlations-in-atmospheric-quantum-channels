import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src import config, utils


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


def plot():
    theta_A1, theta_B1, theta_A2, theta_B2 = 0, np.pi / 8, np.pi / 4, 3 * np.pi / 8
    nu = 3e-4

    eta_c = 1/2 * 0.85
    eta_d = 10**(-0.1 * 50 / 10)
    eta_writing = 0.85
    eta_delay = lambda t, db_p_ms=3: 10**(-db_p_ms * (t * 1000) / 10)

    df = utils.load_data(aperture_size=0.2)
    df *= eta_d
    df *= pd.Series([(eta_writing if s else 1) * eta_delay(s / 10) for s in df.columns], df.columns)

    eta_mean = df.mean()
    eta2_mean = (df**2).mean()
    etacorr_mean = (df.mul(df[0], axis='index')).mean()

    _xi = np.linspace(0.01, 1.2, 100)

    p_B = get_p_B(df)
    p_0 = get_p_0(df)
    p_1 = get_p_1(df)

    max_B = [np.max([get_B(theta_A1, theta_B1, theta_A2, theta_B2, nu, xi, eta_c, df[0], df[t]) for xi in _xi])
        for t in df.columns]
    B_Bell = [ideal_B(p_B[t], p_0[t], p_1[t], eta_c, nu) for t in df.columns]


    dim_scale = 100

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(*smooth(dim_scale * df.columns, max_B, 2), c=config.LINE_COLORS[0])
    plt.plot(*smooth(dim_scale * df.columns, B_Bell, 2), c=config.LINE_COLORS[1], ls='--')
    plt.ylabel(r'Bell parameter $\mathcal{B}$')
    plt.xlabel('Wind-driven shift $s$ [cm]')
    plt.ylim(*(np.asarray(plt.ylim()) * [1, 1]))
    plt.xlim(0, 6)
    plt.plot([dim_scale * df.columns[0], dim_scale * df.columns[-1]], [2, 2],
            c='k', alpha=0.5, lw=0.5, antialiased=None, snap=True)


    axins = ax.inset_axes([0.22, 0.3, 0.41, 0.28])

    axins.plot([_xi[0], _xi[-1]], [2, 2], c='k', alpha=0.5, lw=0.5, antialiased=None, snap=True)
    time = 0.028
    for a in [0.2]:
        _B = [get_B(theta_A1, theta_B1, theta_A2, theta_B2, nu, xi, eta_c, df[0], df[time]) for xi in _xi]
        axins.plot(_xi, _B, c=config.LINE_MAIN_COLOR)
        axins.scatter(_xi[np.argmax(_B)], np.max(_B), c='k', s=12)
        axins.hlines(np.max(_B), 0, _xi[np.argmax(_B)], color='k', ls=':', lw=0.7)
        axins.set_ylim(1, 2.6)
        axins.set_xlim(0, 1)
    axins.set_xlabel(r'Squeezing parameter $\xi$', fontsize=9)
    axins.set_ylabel('Bell\n parameter $\mathcal{B}$', fontsize=9)
    axins.xaxis.set_label_coords(0.47, -0.44)
    axins.text(-0.18, 2.35, r'$\mathcal{B}_\mathrm{m}$')

    plt.savefig('plots/5_bell.pdf', **config.SAVE_KWARGS)
