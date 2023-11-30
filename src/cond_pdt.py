from matplotlib import pyplot as plt

from src import config, utils


def plot():
    eta_min = 0.5
    aperture = 0.3

    df = utils.load_data(aperture)
    df_adhoc = utils.load_adhoc_data(aperture_size=0.3)
    _line_colors = [*config.LINE_COLORS[:-2]]
    _, ax = plt.subplots(1, 1, figsize=(4, 3))

    _x, _y = utils.hist(df[0][df[0] > eta_min], smooth=13, restore_scale=(1e4, 1e4))
    plt.plot([eta_min, *_x], [0, *_y], ls=':', c=_line_colors.pop(), label=f"$s = 0$ cm");
    plt.plot(*utils.hist(df_adhoc[0.01][df_adhoc[0] > eta_min], smooth=10, restore_scale=(1e4, 1e4)), ls='--', c=_line_colors.pop(), label=f"$s = 1$ cm");

    for s, ls in zip([0.056, 0.196], ['-.', (5, (10, 3))]):
        plt.plot(*utils.hist(df[s][df[0] > eta_min], smooth=5, restore_scale=(1e4, 1e4)), ls=ls, c=_line_colors.pop(), label=f"$s = {utils.round_n(s * 100, 3)}$ cm");

    plt.plot(*utils.hist(df[0], smooth=5, restore_scale=(2e2, 1e4)), c=config.LINE_MAIN_COLOR, label=f"$s = \infty$ ")

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])

    plt.ylim(bottom=0)

    plt.gca().axvline(x=eta_min, c='k', lw=0.7)
    plt.text(eta_min, 0.15, f"$\eta_\mathrm{{min}}={eta_min}$")
    plt.xlabel(r"Transmittance $\eta$")
    plt.ylabel(r"Conditional PDT $\mathcal{P}(\eta_\tau|\eta_0 > \eta_\mathrm{min})$")
    plt.savefig("plots/2_cond_pdt.pdf", **config.SAVE_KWARGS)
