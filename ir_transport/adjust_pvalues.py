"""This module contains multiple testing corrections for controlling the family-wise
error rate or the false discovery rate.
"""

import warnings
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline

METHODS = {
    "bonferroni",
    "sidak",
    "empirical_null",
    "holm",
    "hommel",
    "simes-hochberg",
    "bh",
    "by",
    "storey",
}


def is_iterable(obj: Any) -> bool:
    """
    Check if input is an iterable.

    Parameter
    ---------
    obj : anything

    Returns
    -------
    bool
        Whether the input is iterable.
    """
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def estimate_fraction_null(
    pvalues: NDArray[np.float64],
    lambdas: NDArray[np.float64] = np.arange(0.05, 1, 0.05),
    estimate_method: str = "smoother",
    log_transform: bool = False,
    **kwargs: Dict[str, Any],
) -> Tuple[np.float64, NDArray[np.float64], NDArray[np.float64], UnivariateSpline]:
    """
    Estimate the overall proportion of null p values.

    Notably, this function is only valid if the null distribution of p values
    is assumed to be uniform. This algorithm was first described in _[1].
    Code is inspired and adapted from _[2].

    Parameters
    ----------
    pvalues : numpy.ndarray
        An array of p values. These numbers must be in [0, 1].
    lambdas : numpy.ndarray, default np.arange(0.05, 1, 0.05)
        The points (notably, must be in [0, 1)) at which the proportion of null
        p values will be estimated.
    method: str, default 'smoother'
        The final estimate can be inferred using a spline ('smoother') described
        in _[1] or by using a boostrap method ('bootstrap') described in _[3].
    log_transform : bool, default False
        If True, the spline estimating the proportion of null p values
        will be estimated using log pvalue against lambdas.
    **kwargs
        Keyword arguments to scipy.interpolate.UnivariateSpline.

    Returns
    -------
    pi0_hat : numpy.float64
        The estimated overall proportion of null p values.
    pi0_ests : numpy.ndarray
        The estimated proportions of null p values evaluated using the p values
        given by lambdas.
    lambdas : numpy.ndarray
        The values at which the proportion of null p values was estimated.
    spline : scipy.interpolate.UnivariateSpline
        If len(lambdas) >= 4 and method == 'smoother', this will return the
        ensuing spline used to estimate the fraction of null p values.

    References
    ----------
    .. [1] Storey JD, Tibshirani R. (2003) "Statistical significance for
           genomewide studies." Proc Natl Acad Sci U S A. 100(16):9440-5.
           https://doi.org/10.1073/pnas.1530509100
    .. [2] Storey JD (2023) qvalue. https://github.com/StoreyLab/qvalue/tree/master
    .. [3] Storey, JD et al. (2004) "Strong control, conservative point estimation
           and simultaneous conservative consistency of false discovery rates: a
           unified approach." J. R. Stat. Soc., B: Stat. 66: 187-205.
           https://doi.org/10.1111/j.1467-9868.2004.00439.x
    """
    if estimate_method != "smoother" and estimate_method != "boostrap":
        raise ValueError(
            f"{estimate_method} is an invalid option. estimate_method must be "
            "either 'smoother' or 'boostrap'."
        )

    num_tests = len(pvalues)

    if not is_iterable(lambdas):
        len_lambdas = 1
        lambdas = np.array([lambdas])
    else:
        len_lambdas = len(lambdas)
        if len_lambdas > 1 and len_lambdas < 4:
            raise RuntimeError(
                f"The amount of lambdas is {len_lambdas}. Either "
                "one lambda is used or lambdas should have at least "
                "four values."
            )

    if np.any((lambdas < 0) | (lambdas >= 1)):
        raise ValueError("lambdas must be in [0, 1).")

    if np.max(pvalues) < np.max(lambdas):
        raise RuntimeError(
            "The maximum p value is smaller than the lambda range. "
            "Change the range of lambda."
        )

    if len_lambdas == 1:
        spline = None
        pi0_ests = np.mean(pvalues >= lambdas) / (1 - lambdas)
        pi0_hat = np.minimum(pi0_ests, 1)[0]
    else:
        lambdas = lambdas[lambdas < np.max(pvalues)]
        numer = np.count_nonzero(pvalues >= lambdas[:, None], axis=1)
        denom = num_tests * (1 - lambdas)
        pi0_ests = numer / denom

        if estimate_method == "smoother":
            if log_transform:
                spline = UnivariateSpline(lambdas, np.log(pi0_ests), **kwargs)
                pi0_hat = min(np.exp(spline(lambdas[-1])), 1)
            else:
                spline = UnivariateSpline(lambdas, pi0_ests, **kwargs)
                pi0_hat = min(spline(lambdas[-1]), 1)
        else:
            spline = None

            quantile_p1 = np.quantile(pi0_ests, 0.1)
            mse = (
                numer / (num_tests**2 * (1 - lambdas) ** 2) * (1 - numer / num_tests)
                + (pi0_ests - quantile_p1) ** 2
            )
            pi0_hat = np.minimum(pi0_ests[mse == np.min(mse)], 1)[0]

    if pi0_hat <= 0:
        warnings.warn(
            "The estimated probability of null p values <= 0. This "
            "estimate is being set to 1. Check that the p values are "
            "in [0, 1] or use a different range of lambdas."
        )
        pi0_hat = 1

    return pi0_hat, pi0_ests, lambdas, spline


def get_adjusted_pvalues(
    pvalues: NDArray[np.floating],
    method: str,
    fraction_null: np.floating = None,
    null_pvalues: NDArray[np.floating] = None,
    lambdas: NDArray[np.floating] = np.arange(0.05, 1, 0.05),
    estimate_method: str = "smoother",
    log_transform: bool = False,
    **kwargs: Dict[str, Any],
) -> NDArray[np.floating]:
    """
    Adjust p values using multiple testing corrections.

    Source code here is inspired by or adapted from _[1], _[2], and _[3].

    Parameters
    ----------
    pvalues : numpy.ndarray
        An array of p values. These numbers must be in [0, 1].
    method : str
        Method using for adjusting the p values. Available methods:
            'bonferroni' _[4]
            'sidak' _[5]
            'empirical_null'
            'holm' _[6]
            'hommel' _[7]
            'simes-hochberg' _[8] _[9]
            'bh' _[10]
            'by' _[11]
            'storey' _[12] _[13]
    fraction_null : numpy.float64, optional
        The overall proportion of null p values in the supplied pvalues array.
    null_pvalues : numpy.ndarray, optional
        An array of p values which are known to be associated with a null process,
        e.g., p values from replicate experiments. Used in method = 'empirical_null'.
    lambdas : numpy.ndarray, default np.arange(0.05, 1, 0.05)
        The points (notably, must be in [0, 1)) at which the proportion of null
        p values will be estimated. Used in method = 'storey'.
    estimate_method: str, default 'smoother'
        Choices are 'smoother', 'boostrap', 'empirical'.
        If method = 'empiricall_null', 'smoother' will construct a spline from
        the esimated false discovery rates whereas 'empirical' will merely use
        the estimates from data. If method = 'storey', 'smoother' will ensue in
        using a spline to infer the overall proportion of null (see _[9])
        whereas 'bootstrap' will use a method described in _[10].
    log_transform : bool, default False
        If True, the spline estimated will using a log y domain.
    **kwargs
        Keyword arguments to scipy.interpolate.UnivariateSpline.

    Returns
    -------
    numpy.ndarray
        The adjusted p values. Notably, these values can be q values when
        using false discovery rate procedures. Therefore, be mindful of the
        interpretation based on the method selected. The values in this
        array are ordered exactly with the input pvalues array.

    References
    ----------
    .. [1] R Core Team (2020) p.adjust. (Version 4.4.0)
           https://github.com/wch/r-source/blob/master/src/library/stats/R/p.adjust.R
           https://www.r-project.org/
    .. [2] Perktold J (2023) statsmodels.stats.multitest (Version 0.15.0)
           https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html#multipletests
           https://www.statsmodels.org/dev/index.html
    .. [3] Storey JD (2023) qvalue. https://github.com/StoreyLab/qvalue/tree/master
    .. [4] Neyman J, Pearson ES (1928) "On the use and interpretation of certain
           test criteria for purposes of statistical inference: Part I." Biometrika
           20A(1/2): 175-240. https://doi.org/10.2307/2331945
    .. [5] Sidak Z (1967) "Rectangular confidence regions for the means of
           multivariate normal distributions." J Am Stat Assoc 62: 626-633.
           https://doi.org/10.2307/2283989
    .. [6] Holm S (1979) "A Simple Sequentially Rejective Multiple Test Procedure."
           Scand Stat Theory Appl 6(2): 65-70. https://www.jstor.org/stable/4615733
    .. [7] Hommel G (1988) "A stagewise rejective multiple test procedure based on
           a modified Bonferroni test." Biometrika 75(2): 383-386.
           https://doi.org/10.1093/biomet/75.2.383
    .. [8] Simes RJ (1986) "An improved Bonferroni procedure for multiple tests of
           significance." Biometrika 73(3): 751-754.
           https://doi.org/10.1093/biomet/73.3.751
    .. [9] Hochberg Y (1988) "A sharper Bonferroni procedure for multiple tests of
           significance." Biometrika 75(4): 800-802.
           https://doi.org/10.1093/biomet/75.4.800
    .. [10] Hochberg Y, Benjamini Y (1990) "More powerful procedures for multiple
            significance testing." Stat Med 9(7): 811-818.
            https://doi.org/10.1002/sim.4780090710
    .. [11] Benjamini Y, Yekutieli D (2001) "The control of the false discovery
            rate in multiple testing under dependency." Ann Statist 29(4): 1165-1188.
            https://doi.org/10.1214/aos/1013699998
    .. [12] Storey JD, Tibshirani R. (2003) "Statistical significance for
            genomewide studies." Proc Natl Acad Sci U S A. 100(16):9440-5.
            https://doi.org/10.1073/pnas.1530509100
    .. [13] Storey, JD et al. (2004) "Strong control, conservative point estimation
            and simultaneous conservative consistency of false discovery rates: a
            unified approach." J. R. Stat. Soc., B: Stat. 66: 187-205.
            https://doi.org/10.1111/j.1467-9868.2004.00439.x
    """
    if method not in METHODS:
        to_print = ", ".join(METHODS)
        raise ValueError(
            f"{method} is an invalid method. Available methods: "
            f"{to_print}. See documentation for further information."
        )

    pvalues = np.asarray(pvalues)

    gt_1 = pvalues > 1
    if np.any(gt_1):
        num_gt_1 = np.count_nonzero(gt_1)
        raise RuntimeError(
            f"There are {num_gt_1} erroneous p values greater "
            "than 1. p values must be in [0, 1]."
        )

    lt_0 = pvalues < 0
    if np.any(lt_0):
        num_lt_0 = np.count_nonzero(lt_0)
        raise RuntimeError(
            f"There are {num_lt_0} erroneous p values less than "
            "0. p values must be in [0, 1]."
        )

    nan_check = np.isnan(pvalues)
    if np.any(nan_check):
        num_nan = np.count_nonzero(nan_check)
        raise RuntimeError(f"There are {num_nan} erroneous p values which are nan.")

    method = method.lower()
    num_tests = len(pvalues)

    if method == "bonferroni":
        return np.minimum(pvalues * num_tests, 1)

    elif method == "sidak":
        return -np.expm1(num_tests * np.log1p(-pvalues))

    elif method == "empirical_null":
        if null_pvalues is None:
            raise RuntimeError("empirical_null cannot be used if null_pvalues is None.")
        null_pvalues = np.asarray(null_pvalues)

        if estimate_method == "smoother":
            sig_discovery = np.count_nonzero(pvalues < lambdas[:, None], axis=1)
            bkgd_discovery = np.count_nonzero(null_pvalues < lambdas[:, None], axis=1)
            false_discovery_rate = bkgd_discovery / (bkgd_discovery + sig_discovery)

            if log_transform:
                spline = UnivariateSpline(
                    lambdas, np.log(false_discovery_rate), s=0, **kwargs
                )
                p_adj = np.exp(spline(pvalues))
            else:
                spline = UnivariateSpline(lambdas, false_discovery_rate, s=0, **kwargs)
                p_adj = spline(pvalues)

            p_adj = np.clip(p_adj, 0, 1)
            return p_adj

        elif estimate_method == "empirical":
            concat_pvalues = np.concatenate((pvalues, null_pvalues))
            signal_labels = np.concatenate(
                (
                    np.ones(num_tests, dtype=bool),
                    np.zeros(len(null_pvalues), dtype=bool),
                )
            )
            argsort = np.argsort(concat_pvalues)
            pvalues_ascend = concat_pvalues[argsort]
            signal_labels_ascend = signal_labels[argsort]
            sig_discovery = signal_labels_ascend.cumsum()
            bkgd_discovery = (~signal_labels_ascend).cumsum()
            p_adj = bkgd_discovery / (sig_discovery + bkgd_discovery)
            p_adj_unsrt = np.empty_like(p_adj)
            p_adj_unsrt[argsort] = p_adj
            return p_adj_unsrt[:num_tests]
        else:
            raise ValueError(
                f"{estimate_method} is an invalid option. "
                "estimate_method must be either 'smoother' or "
                "'empirical' when method is 'empirical_null'."
            )

    arange = np.arange(1, num_tests + 1)

    # Check monotonicity of p values to see if they are sorted.
    if np.all(pvalues[1:] >= pvalues[:-1]):
        argsort = arange - 1
        input_sorted = True
    else:
        argsort = np.argsort(pvalues)
        input_sorted = False

    if method == "holm" or method == "hommel":
        p_ascend = np.take(pvalues, argsort)

        if method == "holm":
            to_max_acc = (num_tests - arange + 1) * p_ascend
            p_adj = np.minimum(1, np.maximum.accumulate(to_max_acc))

        else:
            p_adj = p_ascend.copy()
            for i in range(num_tests, 1, -1):
                min_right_q = np.min(i * p_ascend[-i:] / arange[:i])
                p_adj[-i:] = np.maximum(p_adj[-i:], min_right_q)
                p_adj[:-i] = np.maximum(
                    p_adj[:-i], np.minimum(i * p_ascend[:-i], min_right_q)
                )

    else:
        arange = arange[::-1]
        argsort = argsort[::-1]
        p_desc = np.take(pvalues, argsort)

        if method == "simes-hochberg":
            to_min_acc = (num_tests - arange + 1) * p_desc
            p_adj = np.minimum(1, np.minimum.accumulate(to_min_acc))

        elif method == "bh":
            to_min_acc = num_tests / arange * p_desc
            p_adj = np.minimum(1, np.minimum.accumulate(to_min_acc))

        elif method == "by":
            q = np.sum(1 / arange)
            to_min_acc = q * num_tests / arange * p_desc
            p_adj = np.minimum(1, np.minimum.accumulate(to_min_acc))

        elif method == "storey":
            if fraction_null is None:
                fraction_null = estimate_fraction_null(
                    p_desc, lambdas, estimate_method, log_transform, **kwargs
                )[0]
            else:
                if fraction_null <= 0 or fraction_null > 1:
                    raise ValueError("fraction_null must be in (0, 1].")
            to_min_acc = num_tests / arange * fraction_null * p_desc
            p_adj = np.minimum.accumulate(to_min_acc)

        if input_sorted:
            p_adj = p_adj[::-1]

    if not input_sorted:
        p_adj_unsrt = np.empty_like(p_adj)
        p_adj_unsrt[argsort] = p_adj
        return p_adj_unsrt
    else:
        return p_adj
