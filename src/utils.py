import math
from pathlib import Path
# from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


LINE_MAIN_COLOR = "#363649"
LINE_COLORS = ["#16a9da", "#fe6361", "#ea8b06", "#9430b0", "#f7ca49", "#7d7059"]
SAVE_KWARGS = {"format": "pdf", "dpi": 300, "bbox_inches": "tight", "pad_inches": 0.005}

plt.rcParams['axes.axisbelow'] = True
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = "STIX"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


# @lru_cache()
def load_data(channel_name, aperture_size):
    file_name = "aperture_" + str(aperture_size).replace(".", "_") + '.csv'
    path = Path('data') / channel_name / file_name
    df = pd.read_csv(path)
    df.columns = [float(c) for c in df.columns]
    return df


def get_apertures(channel_name):
    apertures = []
    for file in (Path('data') / channel_name).glob('*.csv'):
        if '.raw' in file.name or '.adhoc' in file.name:
            continue
        a = float(file.name[9:-4].replace('_', '.'))
        apertures.append(a)
    apertures.sort()
    return apertures


def hist(transmittance, bins=200, smooth=1, restore_scale=(200, 200), restore_shift=(0, 1)):
    density, bin_edges = np.histogram(transmittance, bins=bins, density=True)
    eta = (bin_edges[1:] + bin_edges[:-1]) / 2
    smoothed = scipy.ndimage.gaussian_filter1d(density, smooth)
    restore_shift = (0.9 * np.min(transmittance), np.max(transmittance))
    mask = 2 / (1 + np.exp(-restore_scale[0] * (eta - restore_shift[0]))) / (1 + np.exp(restore_scale[1] * (eta - restore_shift[1]))) - 1
    restored = smoothed * mask
    return eta, restored


def smooth(x, y, smooth, num=100):
    x = np.asarray(x)
    y = np.asarray(y)
    nan_mask = np.isnan(x)
    x = x[~nan_mask]
    y = y[~nan_mask]
    t = np.linspace(x[0], x[-1], num=num)
    values = interp1d(x, y)(t)
    return t, gaussian_filter1d(values, smooth)


def round_n(x, n):
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def pearson_df(df, df2=None):
    df2 = df if df2 is None else df2
    return [scipy.stats.pearsonr(df[0], df2[i])[0] for i in df.columns]


def get_intersect(x, y, y_line=0):
    x = np.asarray(x)
    y = np.asarray(y)
    ascending = sum([(y2-y0)/(x2-x0) for x2, x0, y2, y0 in zip(x[2:], x, y[2:], y)]) > 0
    try:
        mask = y < y_line if ascending else y > y_line
        van_index = np.nonzero(mask)[0][-1]
    except IndexError:
        return np.nan
    if van_index == 0 or van_index == (len(x) - 1):
        return np.nan
    x2, x1, y2, y1 = x[van_index + 1], x[van_index], y[van_index + 1], y[van_index]
    k = (y2 - y1) / (x2 - x1)
    return (y_line - y1) / k + x1
