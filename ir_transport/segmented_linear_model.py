from __future__ import annotations

from typing import *
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import factorial, stdtr

if TYPE_CHECKING:
    from numpy.typing import NDArray


def preprocess_input(
    x: Sequence[float],
    y: Sequence[float],
) -> Tuple[NDArray[np.float64]]:
    """
    Check x and y have the same length and return finite values only.

    Parameters
    ----------
    x : sequence of float
        Independent variables that will be used for fitting the model.
    y : sequence of float
        Ordinate variables that will be used for fitting the model.

    Returns
    -------
    tuple of numpy.ndarray of numpy.float64
        The input x and y with nonfinite values removed.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        msg = "x and y must have the same length."
        raise RuntimeError(msg)

    mask_finite = np.isfinite(x) & np.isfinite(y)
    return x[mask_finite], y[mask_finite]


def _create_design_matrices(
    x: NDArray[np.float64],
    breakpoints: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Return design matrices for testing different breakpoints.

    Parameters
    ----------
    x : numpy.ndarray of numpy.float64
        Independent variables.
    breakpoints : numpy.ndarray of numpy.float64
        The breakpoints to be tested.

    Returns
    -------
    design_mats : numpy.ndarray of numpy.float64
        Array of design matrices.
    """
    x_diff_bkpts = x - breakpoints[..., None]
    max0_vals = np.maximum(x_diff_bkpts, 0)
    ones = np.ones(
        (
            breakpoints.shape[0],
            x.shape[0],
        )
    )
    x_for_mat = np.tile(x, breakpoints.shape[0]).reshape(
        breakpoints.shape[0], x.shape[0]
    )
    design_mats = np.hstack(
        (
            ones,
            x_for_mat,
            max0_vals,
        )
    ).reshape(breakpoints.shape[0], 3, x.shape[0])
    return design_mats.swapaxes(1, 2)


def _evaluate(
    x: Sequence[float],
    params: NDArray[np.float64],
    breakpoints: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute an estimate of y for the given x using the fitted parameters.

    Parameters
    ----------
    x : sequence of float
        Independent variables.
    params : numpy.ndarray of numpy.float64
        The intercept, slope, and slope adjustment parameters.
    breakpoints : numpy.ndarray of numpy.float64
        The breakpoints

    Returns
    -------
    numpy.ndarray of numpy.float64
        The estimated ordinate values using the fitted model parameters.
    """
    x = np.asarray(x)
    max0_vals = np.maximum(x - breakpoints[:, None], 0)
    return (
        params[0]
        + params[1] * x
        + np.sum(max0_vals * params[2 : len(breakpoints) + 2, None], 0)
    )


# TODO Actually report this.
def compute_r_squared(
    y: NDArray[np.float64], y_est: NDArray[np.float64], num_params: int
) -> Tuple[np.float64]:
    """ """
    y_mean = np.mean(y)
    total_sum_squares = np.sum((y - y_mean) ** 2)
    residual_sum_squares = np.sum((y - y_est) ** 2)
    r_squared = 1 - residual_sum_squares / total_sum_squares

    num_data = len(y)
    coef = (num_data - 1) / (len_data - num_params - 1)
    adj_r_squared = 1 - (1 - r_squared) * coef

    return residual_sum_squares, total_sum_squares, r_squared, adj_r_squared


def compute_standard_errors(
    params: NDArray[np.float64],
    cov_mat: NDArray[np.float64],
) -> Tuple[NDArray[np.float64]]:
    """
    Return the standard errors for the intercept, slope, slope-change, and breakpoints.

    Parameters
    ----------
    params : numpy.ndarray of numpy.float64
        The inferred segmented linear model parameters.
    cov_mat : numpy.ndarray of numpy.float64
        The covariance matrix from the linear regression used to infer the model.

    Returns
    -------
    tuple of numpy.ndarray of numpy.float64
        The standard errors for the parameters.
    """
    num_breakpoints = (len(cov_mat) - 2) // 2
    variances = np.diagonal(cov_mat)

    intercept_se = np.sqrt(variances[0])
    max0_var = variances[2 : num_breakpoints + 2]
    max0_se = np.sqrt(max0_var)

    indicator_var = variances[2 + num_breakpoints :]
    max0_params = params[2 : num_breakpoints + 2]
    ratio = params[2 + num_breakpoints :] / max0_params
    idxs = np.arange(0, num_breakpoints)
    breakpoint_se = np.sqrt(
        (
            indicator_var
            + max0_var * ratio**2
            - 2 * cov_mat[idxs + 2, idxs + 2 + num_breakpoints] * ratio
        )
        / max0_params**2
    )

    slope_se = np.sqrt(
        [np.sum(cov_mat[1:k, 1:k]) for k in range(2, 3 + num_breakpoints)]
    )

    return intercept_se, slope_se, max0_se, breakpoint_se


def compute_davies(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_points: int = 10,
    bounds: Optional[Sequence[np.float64]] = None,
    alternative: str = "two-sided",
) -> Tuple[np.float64]:
    """
    Compute the Davies statistic and p value to determine if there is evidence
    for at least one breakpoint in the data.

    Parameters
    ----------
    x : sequence of float
        Independent variables that will be used for fitting the model.
    y : sequence of float
        Ordinate variables that will be used for fitting the model.
    n_points : int, default 10
        How many points will be used to probe where breakpoints could be.
    bounds : sequence of numpy.float64, optional
        The lower and upper bound of the interval used for testing.
    alternative : str, default 'two-sided'
        Which alternative hypothesis is being tested.

    Returns
    -------
    x_best : numpy.float64
        The x at which there is the most evidence for a breakpoint.
    p : numpy.float64
        The Davies p value.

    References
    ----------
    .. [1] Davies R (2002) "Hypothesis testing when a nuisance parameter
           is present only under the alternative: Linear model case."
           Biometrika 89(2), 484-489, https://doi.org/10.1093/biomet/89.2.484
    """
    if bounds is None:
        bounds = (x[1], x[-2])
    interval = np.linspace(bounds[0], bounds[1], n_points)
    design_mats = _create_design_matrices(x, interval)

    df = len(x) - 3

    pinvs = np.linalg.pinv(design_mats)
    sols = np.tensordot(pinvs, y, [2, 0])
    y_ests = np.einsum("ijk,ik->ij", design_mats, sols)
    residuals = np.sum((y_ests - y) ** 2, 1)
    cov_mats = np.einsum("ijk,ilk->ijl", pinvs, pinvs)
    max0_param_se = np.sqrt(cov_mats[:, 2, 2] * residuals / df)

    # Eq. 4, Davies (2002).
    stats = sols[:, 2] / max0_param_se

    # Eq. 5, Davies (2002).
    z_sq = sols[:, 2] ** 2 / cov_mats[:, 2, 2]
    beta_analogue = z_sq / (z_sq + residuals)

    # Total variation (between Eq. 11 and Eq. 12 in Davies (2002)).
    v = np.abs(np.diff(np.arcsin(beta_analogue**0.5))).sum()

    # Follow Muggeo in davies.test.r in R segmented package.
    if alternative == "two-sided":
        abs_stats = np.abs(stats)
        argbest = np.argmax(abs_stats)
        stat = abs_stats[argbest]
        x_best = interval[argbest]
        p = 1 - stdtr(df, stat)
    elif alternative == "less":
        argbest = np.argmin(stats)
        stat = stats[argbest]
        x_best = interval[argbest]
        p = stdtr(df, stat)
    else:
        argbest = np.argmax(stats)
        stat = stats[argbest]
        stat = np.max(stats)
        x_best = interval[argbest]
        p = 1 - stdtr(df, stat)

    # u and adjustment defined after Eq. 12 in Davies (2002).
    u = stat**2 / (df + stat**2)
    adjustment = (
        v
        * (1 - u) ** ((df - 1) * 0.5)
        * factorial(df / 2 - 0.5)
        / (2 * factorial(df / 2 - 1) * np.pi**0.5)
    )
    p = p + adjustment

    if alternative == "two-sided":
        p *= 2

    return x_best, np.minimum(p, 1)


def search_min(
    momentum: np.float64,
    new_breakpoints: NDArray[np.float64],
    old_breakpoints: NDArray[np.float64],
    ones_x_mat: NDArray[np.float64],
    y: NDArray[np.float64],
) -> np.float64:
    """
    Return the segmented linear loss for a given momentum.

    This is used when finding the best momentum when adjusting an update
    of the breakpoints in SegmentedLinearModel._fit().

    Parameters
    ----------
    momentum : numpy.float64
        A number in [0, 1].
    new_breakpoints : numpy.ndarray of numpy.float64
        A set of breakpoints.
    old_breakpoints : numpy.ndarray of numpy.float64
        Another set of breakpoints.
    ones_x_mat : numpy.ndarray of numpy.float64
        A portion of the design matrix which is independent of the breakpoints.
    y : numpy.ndarray of numpy.float64
        The dependent variables.

    Returns
    -------
    numpy.float64
        The sum of squared residuals.
    """
    breakpoints_mix = momentum * new_breakpoints + (1 - momentum) * old_breakpoints
    breakpoints_diff_x = ones_x_mat[1] - breakpoints_mix[:, None]
    design_mat = np.vstack((ones_x_mat, np.maximum(breakpoints_diff_x, 0)))
    pinv = np.linalg.pinv(design_mat.T)
    sol = pinv @ y
    y_ests = design_mat.T @ sol
    return np.sum((y - y_ests) ** 2)


class SegmentedLinearModel:
    """Class to fit piecewise linear models by inferring breakpoints."""

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.SeedSequence = None,
    ) -> None:
        """
        Initialize a SegmentedLinearModel object.

        Parameters
        ----------
        x : sequence of float
            Independent variables that will be used for fitting the model.
        y : sequence of float
            Ordinate variables that will be used for fitting the model.
        seed : int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            The seed for the random number generator.
        """
        self.x, self.y = preprocess_input(x, y)
        if len(self.x) == 1:
            msg = "SegmentedLinearModel cannot be initialized. Only one datapoint is given"
            raise RuntimeError(msg)

        self.x_min = self.x.min()
        self.x_max = self.x.max()
        self.lower_breakpoint_bound = self.x[0] + (self.x[1] - self.x[0]) / 2
        self.upper_breakpoint_bound = self.x[-1] - (self.x[-1] - self.x[-2]) / 2
        self.rng = np.random.default_rng(seed)
        self.len_data = len(self.x)

    def _fit(
        self,
        breakpoints: Sequence[float] = None,
        num_breakpoints: int = 1,
        maxiter: int = 30,
        tol: float = 1e-5,
        h: float = 1.25,
        x: Sequence[float] = None,
        y: Sequence[float] = None,
    ) -> Tuple[np.float64, NDArray[np.float64]]:
        """
        Fit a segmented linear model using the algorithm by Muggeo [1]_.

        Function is adapted from the seg.lm.fit.r script from the segmented R
        package at https://cran.r-project.org/web/packages/segmented/.

        Parameters
        ----------
        breakpoints : sequence of float, optional
            The initial guesses for the breakpoints. If None, then the initial
            guesses are the quantiles specified by num_breakpoints.
        num_breakpoints : int, default 1
            The number of breakpoints to be estimated. If breakpoints is given,
            then num_breakpoints is ignored.
        maxiter : int, default 30
            Maximum number of iterations to perform.
        tol : float, 1e-5
            Tolerance for convergence.
        h : float, default 1.25
            Positive factor modifying the increments in breakpoint updates
            during the estimation process.
        x : sequence of float, optional
            Independent variables. If None, use self.x.
        y : sequence of float, optional
            Ordinate variables. If None, use self.y.

        Returns
        -------
        params : numpy.ndarray of numpy.float64
            The parameters of the segmented linear model.
        breakpoints : numpy.ndarray of numpy.float64
            The inferred breakpoints
        residuals : numpy.float64
            The sum of squared residuals.
        cov_mat : numpy.ndarray of numpy.float64
            The covariance matrix of the inferred parameters.

        References
        ----------
        .. [1] Muggeo V (2003) "Estimating regression models with unknown break-points."
               Statist. Med 22(19), 3055-3071, https://doi.org/10.1002/sim.1545
        """
        if x is None and y is None:
            x = self.x
            y = self.y
            x_min = self.x_min
            x_max = self.x_max
        elif x is not None and y is not None:
            x, y = preprocess_input(x, y)
            x_min = x.min()
            x_max = x.max()
        elif x is not None and y is None:
            msg = "If x is given, y must be given too."
            raise RuntimeError(msg)
        else:
            msg = "If y is given, x must be given too."
            raise RuntimeError(msg)

        breakpoint_min = np.min(breakpoints)
        breakpoint_max = np.max(breakpoints)
        if breakpoint_min <= x_min:
            msg = (
                "Breakpoint analysis failed. The minimum breakpoint "
                f"({breakpoint_min}) is at or below the minimum x "
                f"({x_min}). Ensure that the minimum initial guess is in "
                "(x_min, x_max)."
            )
            raise RuntimeError(msg)
        if breakpoint_max >= x_max:
            msg = (
                "Breakpoint analysis failed. The maximum breakpoint "
                f"({breakpoint_min:.3g}) is at or beyond the maximum  x "
                f"({x_max:.3g}). Ensure that the minimum initial guess is in "
                "(x_min, x_max)."
            )
            raise RuntimeError(msg)

        num_breakpoints = len(breakpoints)
        ones = np.ones_like(x)
        ones_x_mat = np.vstack((ones, x))
        x_diff_bkpts = x - breakpoints[:, None]
        max0_vals = np.maximum(x_diff_bkpts, 0)
        design_mat = np.vstack((ones_x_mat, max0_vals)).T
        pinv = np.linalg.pinv(design_mat)
        sol = pinv @ y
        rss = np.sum(((design_mat @ sol) - y) ** 2)

        design_mat = np.vstack(
            (ones_x_mat, max0_vals, (max0_vals > 0).astype(np.float64))
        ).T
        for _ in range(maxiter):
            pinv = np.linalg.pinv(design_mat)
            sol = pinv @ y
            intercept, slope = sol[0:2]
            max0_params = sol[2 : 2 + num_breakpoints]

            indicator_params = sol[2 + num_breakpoints :]

            if np.isclose(0, sol[2:]).any():
                msg = (
                    "At least one segmented linear model parameter is close to 0. "
                    "Are too many breakpoints being estimated? Is there enough "
                    "evidence for breakpoints in the data?"
                )
                raise RuntimeError(msg)
            delta = indicator_params / max0_params

            new_breakpoints = breakpoints - h * delta
            new_breakpoints = np.clip(
                new_breakpoints,
                self.lower_breakpoint_bound,
                self.upper_breakpoint_bound,
            )

            # Although not in Muggeo (2003), the optimal momentum is computed
            # using the updated breakpoints and the previous breakpoints.
            res = minimize_scalar(
                search_min,
                bracket=(0, 1),
                bounds=(0, 1),
                args=(new_breakpoints, breakpoints, ones_x_mat, y),
            )
            opt_momentum = res.x
            res_loss = res.fun
            breakpoints = (
                opt_momentum * new_breakpoints + (1 - opt_momentum) * breakpoints
            )
            breakpoints = np.clip(
                breakpoints, self.lower_breakpoint_bound, self.upper_breakpoint_bound
            )

            # Muggeo has 0.1 in the denominator, but this increases oscillatory
            # behavior.
            epsilon = np.abs(rss - res_loss) / (rss + 1e-8)
            rss = res_loss

            x_diff_bkpts = x - breakpoints[:, None]
            max0_vals = np.maximum(x_diff_bkpts, 0)
            design_mat = np.vstack(
                (ones_x_mat, max0_vals, (max0_vals > 0).astype(np.float64))
            ).T

            if epsilon < tol:
                break

        # Muggeo computes final coefficients without the indicator part of the
        # design matrix. This shouldn't matter if the indicator params are close
        # to 0. However, the algorithm can converge if the indicator params are
        # not close to 0. Thus, this is a way of implicitly checking that the
        # indicator params -> 0.
        # TODO Match Muggeo. How does this impact indicator params estimate for
        #      computing standard errors for breakpoints?
        design_mat = design_mat[:, : 2 + num_breakpoints]
        pinv = np.linalg.pinv(design_mat)
        sol = pinv @ y
        residuals = np.sum((design_mat @ sol - y) ** 2)
        dof = self.len_data - design_mat.shape[1]
        cov_mat = (pinv @ pinv.T) * residuals / dof
        to_return = (sol, breakpoints, residuals, cov_mat)
        return to_return

    def fit(
        self,
        breakpoints: Optional[Sequence[float]] = None,
        num_breakpoints: int = 1,
        maxiter: int = 30,
        tol: float = 1e-8,
        num_davies: int = 10,
        init_breakpoints: str = "range",
        num_bootstraps: int = 30,
        early_stopping: Optional[int] = None,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.SeedSequence = None,
    ) -> None:
        """
        Fit a segmented linear model to the data with the option of using bootstraps
        to escape local minima.

        Parameters
        ----------
        breakpoints : sequence of float, optional
            The initial guesses for the breakpoints. If None, then the initial
            guesses are the quantiles specified by num_breakpoints.
        num_breakpoints : int, default 1
            The number of breakpoints to be estimated. If breakpoints is given,
            then num_breakpoints is ignored.
        maxiter : int, default 30
            Maximum number of iterations to perform.
        tol : float, optional
            Tolerance for convergence.
        num_davies : int, default 10
            The size of the interval used for computing the Davies test p value.
        init_breakpoints : str, default 'range'
            How the breakpoints should be initialized if none are provided.
            'range': breakpoints are intialized equally across the range of x.
            'quantile': breakpoints are initialized using quantiles of x.
            'random': breakpoints are initialized randomly.
        num_bootstraps : int, default 30
            The number of bootstrap inferences for escaping local minima.
        early_stopping: int, optional
            If this amount of iterations has the same loss, then the bootstrapping
            local minima escape loop will end.
        seed : int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            The seed for the random number generator. If None, the random number
            generator used is self.rng.

        Returns
        -------
        None

        References
        ----------
        .. [1] Muggeo V (2003) "Estimating regression models with unknown break-points."
               Statist. Med 22(19), 3055-3071, https://doi.org/10.1002/sim.1545
        .. [2] Wood S N (2001) "Minimizing model fitting objectives that contain
               spurious local minima by bootstrap restarting." Biometrics 57(1),
               240-4. https://doi.org/10.1111/j.0006-341X.2001.00240.x
        """
        for arg, arg_label, lowest_val in zip(
            [num_breakpoints, maxiter, num_davies, num_bootstraps],
            ["num_breakpoints", "maxiter", "num_davies", "num_bootstraps"],
            [1, 0, 1, 0],
        ):
            if int(arg) != arg:
                msg = f"{arg_label} must be an integer."
                raise TypeError(msg)
            if arg < lowest_val:
                msg = f"{arg_label} must be >= {lowest_val}."
                raise ValueError(msg)

        if tol <= 0:
            msg = "tol must be > 0."
            raise ValueError(msg)

        if early_stopping is not None:
            if int(early_stopping) != early_stopping:
                msg = "early_stopping must be an integer"
                raise TypeError(msg)
            if early_stopping <= 0:
                msg = "early_stopping must be >= 0."
                raise ValueError(msg)
        else:
            early_stopping = np.inf

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        if breakpoints is not None:
            if not hasattr(breakpoints, "__len__"):
                breakpoints = [breakpoints]
            breakpoints = np.sort(np.asarray(breakpoints))
            num_breakpoints = len(breakpoints)
        else:
            if init_breakpoints == "quantile":
                breakpoints = np.linspace(0, 1, 2 + num_breakpoints)[1:-1]
            elif init_breakpoints == "range":
                breakpoints = self.x_min + (self.x_max - self.x_min) * np.arange(
                    1, num_breakpoints + 1
                ) / (num_breakpoints + 1)
            elif init_breakpoints == "random":
                breakpoints = np.sort(
                    rng.uniform(
                        low=self.lower_breakpoint_bound,
                        high=self.upper_breakpoint_bound,
                        size=num_breakpoints,
                    )
                )
            else:
                msg = "init_breakpoints must be 'quantile', 'range', or 'random'."
                raise ValueError(msg)

        if self.len_data <= 2 + 2 * num_breakpoints:
            msg = "Breakpoint analysis failed. Too few datapoints."
            raise RuntimeError(msg)

        self.davies_x, self.davies_p = compute_davies(self.x, self.y, num_davies)

        # Compute fit on initial parameters.
        params, breakpoints, residuals, cov_mat = self._fit(
            breakpoints, num_breakpoints, maxiter, tol
        )

        alpha = 0.1
        same_loss = 0
        tried_quantile = False
        same_loss_thresh = 3
        # Escape local minima using Wood (2001) and Muggeo.
        for _ in range(num_bootstraps):
            bootstrap_idxs = rng.integers(low=0, high=self.len_data, size=self.len_data)
            x_bootstrap = self.x[bootstrap_idxs]
            y_bootstrap = self.y[bootstrap_idxs]

            # Fit bootstrapped data.
            try:
                # If the loss hasn't been changing with bootstraps, perturb the
                # breakpoints.
                if same_loss >= same_loss_thresh:
                    if not tried_quantile:
                        # Muggeo uses breakpoints obtained from quantiles.
                        breakpoint_quantiles = np.mean(
                            breakpoints[:, None] >= self.x, axis=1
                        )
                        breakpoint_quantiles[
                            np.abs(breakpoint_quantiles - 0.5) < 0.1
                        ] = alpha
                        alpha = 1 - alpha
                        breakpoints_to_try = np.sort(
                            np.quantile(self.x, 1 - breakpoint_quantiles)
                        )
                        tried_quantile = True
                    else:
                        # Once the quantile method is tried, use random breakpoints
                        # to try escaping local minima further.
                        breakpoints_to_try = np.sort(
                            rng.uniform(
                                low=self.lower_breakpoint_bound,
                                high=self.upper_breakpoint_bound,
                                size=num_breakpoints,
                            )
                        )
                    _, breakpoints_boot, _, _ = self._fit(
                        breakpoints_to_try,
                        maxiter=maxiter,
                        tol=tol,
                        x=x_bootstrap,
                        y=y_bootstrap,
                    )
                else:
                    _, breakpoints_boot, _, _ = self._fit(
                        breakpoints,
                        maxiter=maxiter,
                        tol=tol,
                        x=x_bootstrap,
                        y=y_bootstrap,
                    )
            except Exception:
                breakpoints_boot = np.sort(
                    rng.uniform(
                        low=self.lower_breakpoint_bound,
                        high=self.upper_breakpoint_bound,
                        size=num_breakpoints,
                    )
                )

            # Fit original data using parameters from bootstrap fit.
            try:
                new_params, new_breakpoints, new_residuals, new_cov_mat = self._fit(
                    breakpoints_boot, maxiter=maxiter, tol=tol
                )
            except Exception:
                same_loss += 1
                continue

            if new_residuals < residuals:
                params = new_params
                breakpoints = new_breakpoints
                residuals = new_residuals
                cov_mat = new_cov_mat
                same_loss = 0
            else:
                same_loss += 1
            if same_loss >= early_stopping:
                break

        self.params = params
        self.breakpoints = breakpoints
        self.residuals = residuals
        self.cov_mat = cov_mat

        (self.intercept_se, self.slope_se, self.max0_se, self.breakpoint_se) = (
            compute_standard_errors(self.params, self.cov_mat)
        )

    def evaluate(self, x: Sequence[float]) -> NDArray[np.float64]:
        """
        Compute an estimate of y for the given x using the fitted parameters.

        Parameters
        ----------
        x : sequence of float
            Independent variables.

        Returns
        -------
        numpy.ndarray of numpy.float64
            The estimated ordinate values using the fitted model parameters.
        """
        return _evaluate(x, self.params, self.breakpoints)

    def print_segmented_input(self) -> None:
        """
        Print statements to screen for running R segmented package.

        This is useful for comparing this module's inference to Muggeo's
        segmented package.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        to_print = "library('segmented')\n"
        to_print += "x <- c(" + str(list(self.x))[1:-1] + ")\n"
        to_print += "y <- c(" + str(list(self.y))[1:-1] + ")\n"
        to_print += "lm_fit <- lm(y~x, data=list(x, y))\n"
        to_print += "s <- segmented(lm_fit, npsi=1)\n"
        to_print += "summary(s)\n" + "davies.test(lm_fit)\n"
        to_print += "sum(s$residual^2)\n"
        print(to_print)
