import numpy as np
from matplotlib import pyplot as plt
import scipy

from src import config, utils


def witnessess(etaa_mean, etab_mean, sqrt_corr, r):
    return ((1 - etaa_mean + etaa_mean * np.cosh(r)**2) * (1 - etab_mean + etab_mean * np.cosh(r)**2) *
            (-sqrt_corr**2 * np.cosh(r)**2 * np.sinh(r)**2 + etaa_mean * etab_mean * np.sinh(r)**4) -
            sqrt_corr * np.cosh(r) * np.sinh(r) *
            (-sqrt_corr**3 * np.cosh(r)**3 * np.sinh(r)**3 +
             sqrt_corr * etaa_mean * etab_mean * np.cosh(r) * np.sinh(r)**5))


def delayed_witnessess(etaa_mean, etab_mean, sqrt_corr, r, t):
    eta_writing = 0.85
    eta_delay = lambda t, db_p_ms=3: 10**(-db_p_ms * (t * 1000) / 10)
    etab_mean *= eta_writing * eta_delay(t)
    sqrt_corr_value = np.sqrt(eta_writing * eta_delay(t)) * sqrt_corr(t)
    return witnessess(etaa_mean, etab_mean, sqrt_corr_value, r)


def plot():
    dim_scale = 100

    df = utils.load_data(aperture_size=0.2)
    eta_mean = df.mean().mean()
    sqrt_corr_data = np.sqrt(df).mul(np.sqrt(df[0]), axis='index').mean(axis='index')
    sqrt_corr = scipy.interpolate.Akima1DInterpolator(df.columns, sqrt_corr_data)


    _r = np.linspace(1, 3, 300)
    van = np.asarray([scipy.optimize.fsolve(lambda t: delayed_witnessess(eta_mean, eta_mean, sqrt_corr, r, t), x0=1e-4)[0] for r in _r])

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(dim_scale * van, _r, c=config.LINE_COLORS[0])
    plt.gca().fill_between([*(dim_scale * van), 0], 0, [*_r, 10], color=config.LINE_COLORS[0], alpha=0.1)
    ax.set_xlabel(r'Wind-driven shift $s$ [cm]')
    ax.set_ylabel(r'Squeezing parameter $\xi$')
    ax.set_xlim(0, 14)
    ax.set_ylim(1.51, 2.4)

    ax.text(10, 1.79, r'$\mathcal{W} > 0$')
    ax.text(5.3, 1.67, r'$\mathcal{W} < 0$')

    plt.tight_layout()
    plt.savefig('plots/4_entanglement.pdf', **config.SAVE_KWARGS)
