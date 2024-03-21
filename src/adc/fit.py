import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import poisson as _poisson

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import pandas as pd
import seaborn as sns
from scipy.special import erf

# class Exponent:

#     def __call__(self, x, b, c):
#         return b * np.exp(x / c)

#     def fmt(self, b, c):
#         return f'{b} * np.exp(x / {c})'


def prob_survive(
    n_cells_range: list,
    single_cell_prob: float,
):
    return [1 - (1 - single_cell_prob) ** n for n in n_cells_range]


def single_prob_to_final_states(
    dft,
    plot=False,
    concentration_col="ng",
):
    """
    For every concentration, fits `prob_survive(n_cells)` curve to final_state counts

    Params:
    -------
    dft: pandas.DataFrame
        with coluns ['n_cells', 'concentration, ng', 'final_state']
    plot : bool
        if true, shows the lineplot of data and optimised fit
    Return:
    -------
    pandas.DataFrame
        with [Concentration, single_prob, precision, q(prob to die)]
    Raises:
    -------
    AssertionError if input columns missing
    scipy.optimize.curve_fit exceptions
    """

    assert all(
        [
            c in dft.columns
            for c in ["n_cells", concentration_col, "final_state"]
        ]
    )
    concentrations = sorted(dft[concentration_col].unique())

    def fit_concentration(c):
        n_f = (
            single_c := dft.query(f"`{concentration_col}` == {c}")[
                ["n_cells", "final_state"]
            ]
        ).values.astype("int")
        n_cells = n_f[:, 0]
        final_states = n_f[:, 1]
        popt, pcov = curve_fit(
            prob_survive,
            n_cells,
            final_states,
        )
        N = sorted(np.unique(n_cells))
        if plot:
            sns.lineplot(x="n_cells", y="final_state", data=single_c)
            plt.plot(
                N,
                prob_survive(N, popt[0]),
                label=f"fit q={(1-popt[0]):.2f} +- {np.sqrt(pcov[0][0]):.4f}",
            )
            plt.legend()
            plt.title(f"[AB] = {c} ng")
            plt.show()
        return {
            concentration_col: c,
            "single_prob": popt[0],
            "precision": np.sqrt(pcov[0][0]),
            "q": 1 - popt[0],
            "MSE": np.mean(
                (final_states - prob_survive(n_cells, popt[0])) ** 2
            ),
        }

    return pd.DataFrame(list(map(fit_concentration, concentrations)))


def show_probs(probs, *args, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(*args, **kwargs)
    try:
        probs.plot(ax=ax, x="concentration, ng", y="q")
        ax.fill_between(
            probs["concentration, ng"],
            probs.q - 1 * probs.precision,
            probs.q + 1 * probs.precision,
            # color='tab:blue',
            alpha=0.2,
        )
    except KeyError:
        ax = probs.plot(ax=ax, x="concentration, ng", y="q")
    finally:
        return ax


def single_prob(
    stats: pd.DataFrame, show_individual_fits=False
) -> pd.DataFrame:
    """
    Fits probability of single cell to survive to the curve of probabilities for range of cells numbers
    Input:
        stats: DataFrame with 'n_cells', 'prob_survive', 'concentration' columns.
    Return:
        DataFrame with 'concentration', prob_survive and q (prob to die)
    """
    assert all(
        [
            c in stats.columns
            for c in [
                "n_cells",
                "prob_survive",
                "ng",
            ]
        ]
    )
    dfs_table = stats.pivot(
        index="ng", columns="n_cells", values="prob_survive"
    )
    vector = dfs_table.columns.values
    probs = [
        {
            "ng": i,
            "prob_survive": (
                p := (
                    fit := curve_fit(
                        prob_survive,
                        vector,
                        dfs_table.loc[i].values,
                    )
                )[0][0]
            ),
            "q": 1 - p,
            "precision": np.sqrt(fit[1][0]),
            "MSE": np.mean(
                (dfs_table.loc[i].values - prob_survive(vector, p)) ** 2
            ),
        }
        for i in dfs_table.index
    ]

    dfp = pd.DataFrame(probs)

    dfp.plot(x="ng", y="q")
    plt.ylim(0, 1)
    plt.show()

    if show_individual_fits:
        for c, p, q, v, _ in dfp.values:
            plt.plot(vector, dfs_table.loc[c], "o", label="data")
            plt.plot(vector, prob_survive(vector, p), label=f"fit q={q:.2f}")
            plt.title(f"{c} ng")
            plt.xlabel("n_cells")
            plt.legend()
            plt.show()

    return dfp


def exponent(x, b, c):
    """b * np.exp(x / c)"""
    return b * np.exp(x * c)


def exp_on_baseline(x, a, b, c):
    """
    a + b * np.exp(c * x)
    """
    return a + b * np.exp(c * x)


# def lin_exp_fun(x, a, b, c, d, cut=0):
#     ''' a + bx, a + c exp(d x), concatenate at point `cut` computed automatically'''
#     try:
#         acut = int(abs(-1 / d * lambertw(- c * d / b)))
#         print(f'cut {acut}')
#     except ValueError:
#         acut = cut
#         print('unable to find intersection lin-exp')
#     return np.concatenate((a + b * x[:acut], exp_on_baseline(x[acut:], a, c, d)))


def lin_exp_fun(x, a, b, c, d, cut):
    cut = int(cut)
    return np.concatenate((a + b * x[:cut], exp_on_baseline(x[cut:], a, c, d)))


def exp_exp_fun(x, a, b, c, d, e, cut):
    """
    Double-exponent fit

    Parameters:
    -----------
    a - common baseline
    b - first pivot point
    c - first power
    d - second pivot point
    e - second power
    cut - juction point

    Return:
    -------
    a curve
    """
    cut = int(cut)
    return np.concatenate(
        (exp_on_baseline(x[:cut], a, b, c), exp_on_baseline(x[cut:], a, d, e))
    )


def exp_exp_fit(
    curve: list, plot=False, fun=exp_exp_fun, base_limit=-2, **kwargs
):
    """fits dual exponent with optimizing a junction"""

    x = np.arange(len(curve))
    params = dict(
        fun=fun, bounds=((base_limit, 0, 0, 0, 0, 0), (0, 10, 1, 1, 1, np.inf))
    )

    def find_cut(cut):
        try:
            popt = fit_exponent(curve, p0=(0, 1, 0.1, 0.1, 0.1, cut), **params)
            fit_result = fun(x, *popt)
            chi2 = np.mean((fit_result - curve) ** 2)
            return chi2
        except RuntimeError:
            return np.inf

    chi2s = list(map(find_cut, range(len(curve) - 10)))

    cut = np.argmin(chi2s)
    popt = fit_exponent(curve, p0=(0, 1, 0.1, 0.1, 0.1, cut), **params)

    if plot:
        plt.semilogy(curve)
        plt.plot(x, fun(x, *popt), ".")
        plt.plot(cut, fun(x, *popt)[cut], "o")
        plt.show()

    return popt


def lin_exp_fit(curve: list, plot=False, fun=lin_exp_fun, **kwargs):
    """fits linear fuction in the beginning of the curve and exponent to the end"""

    x = np.arange(len(curve))

    def find_cut(cut):
        try:
            popt = fit_exponent(curve, fun=fun, p0=(0, 0, 0, 0, cut))
            fit_result = fun(x, *popt)
            chi2 = np.mean((fit_result - curve) ** 2)
            return chi2
        except RuntimeError:
            return np.inf

    chi2s = list(map(find_cut, range(len(curve))))

    cut = np.argmin(chi2s)
    popt = fit_exponent(curve, fun=fun, p0=(0, 0, 0, 0, cut))

    if plot:
        plt.semilogy(curve)
        plt.plot(x, fun(x, *popt), ".")
        plt.plot(cut, fun(x, *popt)[cut], "o")
        plt.show()

    return popt


def fit_exp_on_baseline(data, x=None, p0=(1, 1, 0), plot=False, **kwargs):
    """fits a + b * exp(c * x)
    kwargs: same as in fit.fit_exponent and fit.plot_fit
    returns:
    (a, b, c)
    """
    try:
        popt, bins = fit_exponent(
            data,
            bins=x,
            fun=exp_on_baseline,
            p0=p0,
            plot=False,
            return_bins=True,
        )
        if plot == "log":
            curve = data - popt[0]
            fit = exp_on_baseline(bins, 0, popt[1], popt[2])
            plot_fit(curve, fit, bins, plot=plot, **kwargs)
        if plot == "linear":
            curve = data
            fit = exp_on_baseline(bins, *popt)
            plot_fit(curve, fit, bins, plot=plot, **kwargs)
        return popt
    except RuntimeError:
        # if plot:
        #     plt.plot(x, data)
        print("No fit")
        return (None, None, None)


def lag_exponent(x, lag, c):
    """np.exp((x - lag) / c)"""
    return np.exp((x - lag) / c)


def gompertz(t, baseline, a, b, c):
    """
    https://en.wikipedia.org/wiki/Gompertz_function
    """
    return baseline + a * np.exp(-b * np.exp(-c * t))


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(a * x + b))


def hill(x, n, K):
    return x**n / (x**n + K)


def erf_fun(x, a, b):
    return (1 + erf(a * (x - b))) / 2


def fit_sigmoid(
    probs, ax=None, fun=sigmoid, fit_name="sigmoid", p0=(2.0, -2.0)
):
    probs = probs.copy()
    vector = probs.ng
    #     print(vector)
    #     probs.loc[:, 'negative'] = 1 - probs.positive
    popt, pcov = curve_fit(fun, vector, probs.q, p0=p0)
    a, b = popt
    da, db = np.sqrt(np.diag(pcov))
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(vector, probs.q, ".", label="data")
    ax.plot(
        (x := sorted(vector)),
        fun(x, *popt),
        lw=10,
        alpha=0.5,
        label=f"{fit_name} fit",
    )
    plt.legend()
    return popt, (da, db)


def fit_exponent(
    curve,
    bins=None,
    fun=exponent,
    p0=(
        1.0,
        0.5,
    ),
    bounds=(-np.inf, np.inf),
    plot=False,
    plot_init=False,
    return_bins=False,
    **kwargs,
):
    """
    Fits exponent to 1D curve
    plot: ['linear', 'log', None]
    """
    if bins is None:
        bins = np.arange(len(curve))
    popt, _ = curve_fit(f=fun, xdata=bins, ydata=curve, p0=p0, bounds=bounds)

    fit_result = fun(bins, *popt)
    plot_fit(curve, fit_result, bins, plot=plot, **kwargs)
    if plot_init:
        plot_fit(
            curve,
            fun(bins, *p0),
            bins,
            plot=plot,
            labels=["init", "data"],
            **kwargs,
        )
    if return_bins:
        return popt, bins
    return popt


def plot_fit(
    curve=None,
    fit=None,
    vector=None,
    plot="linear",
    labels=["data", "fit"],
    markers=[".", "-"],
    legend=True,
    **kwargs,
):
    """
    plot: ['linear', 'log', None]
    """

    def plot_curves(fun):
        [
            fun(vector, c, marker, label=label)
            for c, marker, label in zip([curve, fit], markers, labels)
        ]
        if legend:
            plt.legend(loc=(1, 0))

    if plot == "log":
        plot_curves(plt.semilogy)
    elif plot == "linear":
        plot_curves(plt.plot)
    elif plot == False:
        pass
    else:
        print(f"plot type `{plot}` not understood. Use `log` or `linear`")

    return True


def add_doubling_time(
    df: pd.DataFrame,
    rate_column: str = "c",
    frame_rate: float = 1.0,
    new_column="Doubling time, min",
    convert_rate_time=lambda c, frame_rate: np.log(2) / c * frame_rate,
):
    """
    adds a column with doubling time using rate 'c'
    """
    table = df.copy()
    table[new_column] = convert_rate_time(
        table[rate_column].values, frame_rate
    )
    return table


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
