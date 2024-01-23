import warnings
import math

import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.special import binom

from src import utils


def preselection(data, eta_min, min_size_warn=1000):
    mask = (data[0] > eta_min)
    pre_selection_size = mask.sum()
    if min_size_warn and pre_selection_size < min_size_warn:
        warnings.warn(f"Small dataset after pre-selection: {pre_selection_size}", RuntimeWarning)
    return data[mask]


# Pns

def get_Pn(eta, n, N, alpha_0, r, xi=0):
    eta = np.asarray(eta)

    def CNk(k):
        _numer = 2 * N * np.exp(-xi * (N - k) / N)
        _d1 = eta**2 * (N - k)**2
        _d2 = 2 * np.cosh(2 * r) * ((2 - eta) * N + eta * k) * eta * (N - k)
        _d3 = ((2 - eta) * N + eta * k)**2

        _sigma_inv_coef = eta * (N - k) / (_d1 + _d2 + _d3)
        _alpha_vec = np.asarray([np.conjugate(alpha_0), -alpha_0])
        _matrix = np.asarray([[
            eta * (N - k) * np.cosh(2 * r) + (2 - eta) * N + eta * k,
            -np.sinh(2 * r) * eta * (N - k)
        ], [
            -np.sinh(2 * r) * eta * (N - k),
            eta * (N - k) * np.cosh(2 * r) + (2 - eta) * N + eta * k
        ]])
        _expon = np.exp(_sigma_inv_coef * np.tensordot(
            np.tensordot(_alpha_vec, _matrix, axes=(0, 0)),
            np.conjugate(-_alpha_vec), axes=(0, 0)))
        return _numer / np.sqrt(_d1 + _d2 + _d3) * _expon
    adds = [binom(n, k) * (-1)**(n - k) * CNk(k) for k in range(n + 1)]
    return binom(N, n) * sum(adds)


def get_Pns(eta, N, alpha_0, r, xi=0):
    return [get_Pn(eta=eta, n=n, N=N, alpha_0=alpha_0, r=r, xi=xi) for n in range(N + 1)]


def mean_Pns(eta, N, alpha_0, r, xi=0):
    return [get_Pn(eta, n, N, alpha_0, r, xi=0).mean() for n in range(N + 1)]


# Mandel

def get_n_mean(alpha_0, r):
    return np.sinh(r)**2 + np.real(alpha_0)**2


def get_dn2_ord_mean(alpha_0, r):
    return np.sinh(r)**2 * np.cosh(2 * r) + np.real(alpha_0)**2 * (np.exp(-2 * r) - 1)


def get_Q(alpha_0, r):
    return get_dn2_ord_mean(alpha_0, r) / get_n_mean(alpha_0, r)


def Qout(alpha_0, r, eta_mean, eta2_mean, nu):
    Qin = get_Q(alpha_0, r)
    n_mean = get_n_mean(alpha_0, r)
    return eta2_mean * n_mean / (eta_mean * n_mean + nu) * Qin + (eta2_mean - eta_mean**2) * n_mean**2 / (eta_mean * n_mean + nu)


def mandel_array(Pn):
    N = np.asarray(Pn).shape[0]
    Pn = np.asarray([*Pn, 1 - np.sum(Pn)])
    k = np.arange(Pn.shape[0])
    c_mean = np.sum(k * Pn)
    c2_mean = np.sum(k**2 * Pn)
    dc2_mean = c2_mean - c_mean**2
    Q_B = N * dc2_mean / (c_mean * (N - c_mean)) - 1
    return Q_B


# Inequality

def _povmv_my(t, N):
    t = np.asarray(t)
    n = np.arange(N).reshape(-1, 1)
    povm = scipy.special.binom(N, n) * t**(N - n) * (1 - t)**n
    return povm.T if np.asarray(t).shape else povm.T[0]


def _dpovmv_my(t, N):
    t = np.asarray(t)
    n = np.arange(1, N).reshape(-1, 1)
    dpovm_0 = (N * t**(N - 1)).reshape(-1, 1)
    dpovm_1N = (scipy.special.binom(N, n) * ((N - n) * t**(N - n - 1) * (1 - t)**n - n * t**(N-n) * (1 - t)**(n - 1))).T
    dpovm = np.hstack([dpovm_0, dpovm_1N])
    return dpovm if np.asarray(t).shape else dpovm[0]


def _povmv(t, N):
    def povm(t, n, D=1):
        if n > 0:
            return math.comb(D, n) * t ** n * (1 - t) ** (D - n)
        else:
            return (1 - t) ** D
    return np.asarray([povm(t, j, N) for j in range(N)])


def _dpovmv(t, N):
    def dpovm(t, n, D=1):
        if n > 0:
            return math.comb(D, n) * (
                    n * t ** (n - 1) * (1 - t) ** (D - n) - (D - n) * t ** n * (1 - t) ** (D - n - 1)
            )
        else:
            return -D * (1 - t) ** (D - 1)
    return np.asarray([dpovm(t, j, N) for j in range(N)])


def _lambda_3d(t, tau, d):
    I = np.zeros(3)
    I[d] = 1
    Pi = _povmv(t, 3)
    dPi = _dpovmv(t, 3)
    Pi0 = _povmv(tau, 3)
    m = np.vstack((I, dPi, Pi0 - Pi))
    return (-1)**tau * np.linalg.det(m)


def err_3d(Pns, t, tau, S):
    Lambda = np.asarray([_lambda_3d(t, tau, d) for d in range(3)])
    return np.sqrt(np.dot(Lambda**2, Pns) - np.dot(Lambda, Pns)**2) / np.sqrt(S)


def _lbound_3d(Pns, t, tau):
    vPi = _povmv(t, 3)
    Lambda = [_lambda_3d(t, tau, d) for d in range(3)]
    return np.dot(Lambda, (Pns - vPi))


def violation_3d(Pns, S):
    epsilon = 0.005
    mu, t1, tau = max([
        max([[_lbound_3d(Pns, t1, 0), t1, 0] for t1 in np.arange(0, 1, epsilon)], key=lambda x: x[0]),
        max([[_lbound_3d(Pns, t1, 1), t1, 1] for t1 in np.arange(0, 1, epsilon)], key=lambda x: x[0]),
    ], key=lambda x: x[0])
    return mu, err_3d(Pns, t1, tau, S)


def _lambda_5D(t1: float, t2: float, tau: float, d: int):
    I = np.zeros(5)
    I[d] = 1
    Pi1 = _povmv(t1, 5)
    dPi1 = _dpovmv(t1, 5)
    Pi2 = _povmv(t2, 5)
    dPi2 = _dpovmv(t2, 5)
    Pi0 = _povmv(tau, 5)
    m = np.vstack((I, Pi1 - Pi0, Pi1 + dPi1 - Pi0, Pi2 - Pi0, Pi2 + dPi2 - Pi0))
    return (-1) ** tau * np.linalg.det(m)


def err_5d(Pns, t1, t2, tau, S):
    Lambda = np.asarray([_lambda_5D(t1, t2, tau, d) for d in range(5)])
    return np.sqrt(np.dot(Lambda**2, Pns) - np.dot(Lambda, Pns)**2) / np.sqrt(S)


def lbound5D(Pns, t1: float, t2: float, tau: float):
    vPi = _povmv(t1, 5)
    Lambda = [_lambda_5D(t1, t2, tau, d) for d in range(5)]
    return np.dot(Lambda, (Pns - vPi))


def violation_5d(Pns, S):
    epsilon = 0.01
    mu, t1, t2, tau = max([
        max([[lbound5D(Pns, t1, t2, 0), t1, t2, 0] for t1 in np.arange(0, 1, epsilon)
             for t2 in np.arange(0, 1, epsilon)], key=lambda x: x[0]),
        max([[lbound5D(Pns, t1, t2, 1), t1, t2, 1] for t1 in np.arange(0, 1, epsilon)
             for t2 in np.arange(0, 1, epsilon)], key=lambda x: x[0]),
    ], key=lambda x: x[0])
    return mu, err_5d(Pns, t1, t2, tau, S)


def lbound2D(Pns, t):
    Pns = np.asarray(Pns)
    vPi = _povmv_my(t, 2)
    Lambda = [(-1) * _dpovmv_my(t, 2)[1], _dpovmv_my(t, 2)[0]]
    return np.dot(Lambda, (Pns - vPi))


def err_2d(Pns, t, S):
    Lambda = np.asarray([(-1) * _dpovmv_my(t, 2)[1], _dpovmv_my(t, 2)[0]])
    return np.sqrt(np.dot(Lambda**2, Pns) - np.dot(Lambda, Pns)**2) / np.sqrt(S)


def violation_2d(Pns, S=1e6):
    epsilon = 0.01
    m, t = max([[lbound2D(Pns, t1), t1] for t1 in np.arange(0, 1, epsilon)], key=lambda x: x[0])
    return m, err_2d(Pns, t, S)


class NonclassicalityPlot():
    def __init__(self, dim_scale=(100, 1)):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_xlabel('Wind-driven shift $s$ (cm)')
        ax.set_ylabel('Maximum violation')
        self.dim_scale = dim_scale
        self.fig = fig
        self.ax = ax
        self.line_color_id = 1
        
        self.eta_d = lambda length_km: 0.8 * 10**(-0.1 * length_km / 10)
        self.eta_min = 0.1
        self.nu = 0
        self.r = 0.59
        self.alpha_0 = 1.15

    def plot(self, channel_name, aperture_radius, *, channel_length_km, xlim=(0, 29)):
        df = utils.load_data(channel_name, aperture_radius)
        df *= self.eta_d(channel_length_km)
    
        pre_sel_data = preselection(df, self.eta_min)
    
        Q = Qout(self.alpha_0, self.r, pre_sel_data.mean(axis=0), (pre_sel_data**2).mean(axis=0), self.nu)
        QB2 = [mandel_array(mean_Pns(pre_sel_data[t], N=2, alpha_0=self.alpha_0, r=self.r)[:-1]) for t in df.columns]
        QB3 = [mandel_array(mean_Pns(pre_sel_data[t], N=3, alpha_0=self.alpha_0, r=self.r)[:-1]) for t in df.columns]
        QB5 = [mandel_array(mean_Pns(pre_sel_data[t], N=5, alpha_0=self.alpha_0, r=self.r)[:-1]) for t in df.columns]
    
    
        E2, E2_e = np.asarray([violation_2d(mean_Pns(pre_sel_data[t], N=2, alpha_0=self.alpha_0, r=self.r)[:-1], S=1e6) for t in df.columns]).T
        E3, E3_e = np.asarray([violation_3d(mean_Pns(pre_sel_data[t], N=3, alpha_0=self.alpha_0, r=self.r)[:-1], S=1e6) for t in df.columns]).T
        E5, E5_e = np.asarray([violation_5d(mean_Pns(pre_sel_data[t], N=5, alpha_0=self.alpha_0, r=self.r)[:-1], S=1e6) for t in df.columns]).T

        scaled_windshifts = self.dim_scale[0] * df.columns
        self.ax.plot(*utils.smooth(scaled_windshifts, E2, 3), label=f"E2", c="#402602", ls=':')
        self.ax.plot(*utils.smooth(scaled_windshifts, E3, 3), label=f"E3", c="#542120", ls='--')
        self.ax.plot(*utils.smooth(scaled_windshifts, E5, 3), label=f"E5", c="#052530", ls='-')
    
        self.ax.fill_between(scaled_windshifts, E2 - E2_e, E2 + E2_e, alpha=0.2, color=utils.LINE_COLORS[2])
        self.ax.fill_between(scaled_windshifts, E3 - E3_e, E3 + E3_e, alpha=0.2, color=utils.LINE_COLORS[1])
        self.ax.fill_between(scaled_windshifts, E5 - E5_e, E5 + E5_e, alpha=0.2, color=utils.LINE_COLORS[0])
    
        Q_max = utils.get_intersect(scaled_windshifts, Q)
        QB2_max = utils.get_intersect(scaled_windshifts, QB2)
        QB3_max = utils.get_intersect(scaled_windshifts, QB3)
        QB5_max = utils.get_intersect(scaled_windshifts, QB5)
    
        ylim = self.ax.get_ylim()
    
        self.ax.vlines(QB2_max, *ylim, color='k', ls=':', alpha=0.7, lw=1.5)
        self.ax.vlines(QB3_max, *ylim, color='k', ls='--', alpha=0.7, lw=1.5)
        self.ax.vlines(QB5_max, *ylim, color='k', alpha=0.7, lw=1.5)
    
        self.ax.vlines(Q_max, *ylim, color='k', ls='-.', alpha=0.7, lw=1.5)
        self.ax.set_ylim(*ylim)
    
        self.ax.plot(self.ax.get_xlim(), [0, 0], c='k', alpha=1, lw=0.5, antialiased=None, snap=True)
        self.ax.set_xlim(*xlim)
    
        ax_t = self.ax.twiny()
        ax_t.set_xlim(self.ax.get_xlim())
        ax_t.set_xticks([Q_max, QB5_max, QB3_max, QB2_max])
        ax_t.set_xticklabels([r"$Q$", r"$Q_\mathrm{5}$", r" $Q_\mathrm{3}$", r"$Q_\mathrm{2}$"])
    

    def savefig(self, file_path, **kwargs):
        self.fig.tight_layout()
        kwargs = {**utils.SAVE_KWARGS, **kwargs}
        self.fig.savefig(file_path, **kwargs)
