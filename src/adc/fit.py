import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import poisson as _poisson

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def poisson(
    numbers: np.ndarray,
    max_value=None,
    xlabel="Initial number of cells",
    title="",
    plot=True,
    save_fig_path=None,
):
    if max_value is None:
        max_value = int(numbers.mean() + 3 * numbers.std())
        logger.info(
            f"Max value: {numbers.max()}, truncating to {max_value} based on mean + 3 * std"
        )
    bins = np.arange(max_value + 1) - 0.5
    vector = bins[:-1] + 0.5
    hist, bins = np.histogram(numbers, bins=bins, density=True)
    popt, pcov = curve_fit(_poisson.pmf, vector, hist, p0=(1.0,))
    l = popt[0]
    if plot:
        plt.hist(numbers, bins=bins, fill=None)
        plt.plot(
            vector,
            len(numbers) * _poisson.pmf(vector, l),
            ".-",
            label=f"Poisson fit λ={l:.1f}",
            color="tab:red",
        )
        plt.xlabel(xlabel)
        plt.title(title)
        plt.legend()
        if save_fig_path is not None:
            try:
                plt.savefig(save_fig_path)
                logger.info(f"Save histogram {save_fig_path}")
            except Exception as e:
                logger.error("saving histogram failed", e.args)
    return l
