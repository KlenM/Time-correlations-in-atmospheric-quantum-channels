from matplotlib import pyplot as plt
import numpy as np

from src import utils

class ConditionalPDTPlot():
    def __init__(self, eta_min):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_xlabel(r"Transmittance $\eta$")
        ax.set_ylabel(r"Conditional PDT $\mathcal{P}(\eta_\tau|\eta_0 > \eta_\mathrm{min})$")
        self.fig = fig
        self.ax = ax
        self.line_color_id = 0
        self.eta_min = eta_min
        
        self.ax.axvline(x=eta_min, c='k', lw=0.7)
        self.ax.text(eta_min, 0.15, f"$\eta_\mathrm{{min}}={eta_min}$")

    def _plot_0(self, channel_name, aperture_radius, **kwargs):
        df = utils.load_data(channel_name, aperture_radius)
        _eta, _pdt = utils.hist(df[0][df[0] > self.eta_min], smooth=13, restore_scale=(1e4, 1e4))
        eta = [self.eta_min, *_eta]
        pdt = [0, *_pdt]
        kwargs = {'color': utils.LINE_COLORS[self.line_color_id], 'label': f"$s = 0$ cm", **kwargs}
        self.ax.plot(eta, pdt, **kwargs)
        self.line_color_id += 1

    def _plot_inf(self, channel_name, aperture_radius, **kwargs):
        df = utils.load_data(channel_name, aperture_radius)
        eta, pdt = utils.hist(df[0], smooth=5, restore_scale=(2e2, 1e4))
        kwargs = {'color': utils.LINE_MAIN_COLOR, 'label': f"$s = \infty$", **kwargs}
        self.ax.plot(eta, pdt, **kwargs)
    
    def plot(self, channel_name, aperture_radius, wind_shift, smooth=5, **kwargs):
        if wind_shift == 0:
            return self._plot_0(channel_name, aperture_radius, **kwargs)
        if wind_shift == np.inf:
            return self._plot_inf(channel_name, aperture_radius, **kwargs)
                    
        df = utils.load_data(channel_name, aperture_radius)
        eta, pdt = utils.hist(df[wind_shift][df[0] > self.eta_min], smooth=smooth, restore_scale=(1e4, 1e4))
        kwargs = {'color': utils.LINE_COLORS[self.line_color_id], 
                  'label': f"$s = {utils.round_n(wind_shift * 100, 3)}$ cm", **kwargs}
        self.ax.plot(eta, pdt, **kwargs);
        self.line_color_id += 1

    def legend(self):
        self.ax.set_ylim(bottom=0)
        self.ax.legend()

    def savefig(self, file_path, **kwargs):
        self.fig.tight_layout()
        kwargs = {**utils.SAVE_KWARGS, **kwargs}
        self.fig.savefig(file_path, **kwargs)
        
