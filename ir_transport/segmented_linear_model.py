from typing import *

import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial, stdtr
from scipy.optimize import minimize_scalar

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
        raise RuntimeError(
            'x and y must have the same length.'
        )

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
    indicator_vals = (x_diff_bkpts > 0).astype(np.float64)
    ones = np.ones((breakpoints.shape[0], x.shape[0], ))
    x_for_mat = np.tile(x, breakpoints.shape[0]).reshape(
        breakpoints.shape[0], x.shape[0]
    )
    design_mats = np.hstack((ones, x_for_mat, max0_vals, indicator_vals)).reshape(
        breakpoints.shape[0], 4, x.shape[0]
    )
    return design_mats.swapaxes(1, 2)

def _eval(
    x: Sequence[float],
    intercept: np.float64,
    slope: np.float64,
    breakpoints: NDArray[np.float64],
    max0_params: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute an estimate of y for the given x using the fitted parameters.

    Parameters
    ----------
    x : sequence of float
        Independent variables.
    intercept : numpy.float64
        The intercept.
    slope : numpy.float64
        The slope.
    breakpoints : numpy.ndarray of numpy.float64
        The breakpoints
    max0_params : numpy.ndarray of numpy.float64
        The difference-in-slopes parameters for the different breakpoints.

    Returns
    -------
    numpy.ndarray of numpy.float64
        The estimated ordinate values using the fitted model parameters.
    """
    x = np.asarray(x)
    max0_vals = np.maximum(x - breakpoints[:, None], 0)
    return intercept + slope * x + np.sum(max0_vals * max0_params[:, None], 0)

# TODO Actually report this.
def compute_r_squared(
    y: NDArray[np.float64],
    y_est: NDArray[np.float64],
    num_params: int
) -> Tuple[np.float64]:
    """
    """
    y_mean = np.mean(y)
    total_sum_squares = np.sum((y - y_mean)**2)
    residual_sum_squares = np.sum((y - y_est)**2)
    r_squared = 1 - residual_sum_squares / total_sum_squares

    num_data = len(y)
    coef = (num_data - 1) / (len_data - num_params - 1)
    adj_r_squared = 1 - (1 - r_squared) * coef

    return residual_sum_squares, total_sum_squares, r_squared, adj_r_squared

def compute_standard_errors(
    params: Tuple[NDArray[np.float64]],
    cov_mat: NDArray[np.float64],
) -> Tuple[NDArray[np.float64]]:
    """
    Return the standard errors for the intercept, slope, slope-change, and breakpoints.

    Parameters
    ----------
    params : tuple fo numpy.ndarray of numpy.float64
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
    max0_var = variances[2:num_breakpoints + 2]
    max0_se = np.sqrt(max0_var)

    indicator_var = variances[2 + num_breakpoints:]
    ratio = params[4] / params[3]
    idxs = np.arange(0, num_breakpoints)
    breakpoint_se = np.sqrt((
        indicator_var
        + max0_var * ratio**2
        - 2 * cov_mat[idxs + 2, idxs + 2 + num_breakpoints] * ratio
    ) / params[3]**2)

    slope_se = np.sqrt(
        [np.sum(cov_mat[1:k, 1:k]) for k in range(2, 3 + num_breakpoints)]
    )

    return intercept_se, slope_se, max0_se, breakpoint_se

def compute_davies(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_points: int = 10,
    bounds: Optional[Sequence[np.float64]] = None,
    alternative: str = 'two-sided',
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
    design_mats = _create_design_matrices(x, interval)[:, :, :3]

    df = len(x) - 3

    pinvs = np.linalg.pinv(design_mats)
    sols = np.tensordot(pinvs, y, [2, 0])
    y_ests = np.einsum('ijk,ik->ij', design_mats, sols)
    residuals = np.sum((y_ests - y)**2, 1)
    cov_mats = np.einsum('ijk,ilk->ijl', pinvs, pinvs)
    max0_param_se = np.sqrt(cov_mats[:, 2, 2] * residuals / df)

    # Eq. 4, Davies (2002).
    stats = sols[:, 2] / max0_param_se

    # Eq. 5, Davies (2002).
    z_sq = sols[:, 2]**2 / cov_mats[:, 2, 2]
    beta_analogue = z_sq / (z_sq + residuals)

    # Total variation (between Eq. 11 and Eq. 12 in Davies (2002)).
    v = np.abs(np.diff(np.arcsin(beta_analogue**0.5))).sum()

    # Follow Muggeo in davies.test.r in R segmented package.
    if alternative == 'two-sided':
        abs_stats = np.abs(stats)
        argbest = np.argmax(abs_stats)
        stat = abs_stats[argbest]
        x_best = interval[argbest]
        p = 1 - stdtr(df, stat)
    elif alternative == 'less':
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
    u = stat**2  / (df + stat**2)
    adjustment = (
        v * (1 - u)**((df - 1) * 0.5)
        * factorial(df / 2 - 0.5)
        / (2 * factorial(df / 2 - 1) * np.pi**0.5)
    )
    p = p + adjustment

    if alternative == 'two-sided':
        p *= 2

    return x_best, np.minimum(p, 1)

def search_min(
    momentum: np.float64,
    new_breakpoints: NDArray[np.float64],
    old_breakpoints: NDArray[np.float64],
    ones_x_mat: NDArray[np.float64],
    y: NDArray[np.float64]
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
    return np.sum((y - y_ests)**2)

class SegmentedLinearModel(object):
    """
    Class to fit piecewise linear models by inferring breakpoints.
    """
    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        seed: int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence = None
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
        intercept : float
            The inferred intercept.
        slope : float
            The inferred slope.
        breakpoints : numpy.ndarray of numpy.float64
            The inferred breakpoints
        max0_params : numpy.ndarray of numpy.float64
            The difference-in-slopes parameters for the different breakpoints.
        indicator_params : numpy.ndarray of np.float64
            The parameters associated with updated the estimation of the
            breakpoints. They are used for assessing convergence and statistics
            of the breakpoints, not evaluating ordinate values.

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
            raise RuntimeError(
                'If x is given, y must be given too.'
            )
        else:
            raise RuntimeError(
                'If y is given, x must be given too.'
            )

        if breakpoints.min() <= x_min:
            raise RuntimeError(
                'Breakpoint analysis failed. The minimum breakpoint is at or below '
                'the minimum x. Ensure that the minimum initial guess is in '
                '(x_min, x_max).'
            )
        if breakpoints.max() >= x_max:
            raise RuntimeError(
                'Breakpoint analysis failed. The maximum breakpoint is at or '
                'beyond the maximum x. Ensure that the maximum initial guess '
                'is in (x_min, x_max).'
            )

        num_breakpoints = len(breakpoints)
        ones = np.ones_like(x)
        ones_x_mat = np.vstack((ones, x))
        x_diff_bkpts = x - breakpoints[:, None]
        max0_vals = np.maximum(x_diff_bkpts, 0)
        design_mat = np.vstack((ones_x_mat, max0_vals)).T
        pinv = np.linalg.pinv(design_mat)
        sol = pinv @ y
        rss = np.sum(((design_mat @ sol) - y)**2)

        design_mat = np.vstack((
            ones_x_mat, max0_vals, (max0_vals > 0).astype(np.float64)
        )).T
        for _ in range(maxiter):
            pinv = np.linalg.pinv(design_mat)
            sol = pinv @ y
            intercept, slope = sol[0:2]
            max0_params = sol[2:2 + num_breakpoints]

            indicator_params = sol[2 + num_breakpoints:]

            if np.isclose(0, sol[2:]).any():
                raise RuntimeError(
                    'At least one segmented linear model parameter is close to 0. '
                    'Are too many breakpoints being estimated? Is there enough '
                    'evidence for breakpoints in the data?'
                )
            delta = indicator_params / max0_params

            new_breakpoints = breakpoints - h * delta
            new_breakpoints = np.clip(
                new_breakpoints, self.lower_breakpoint_bound, self.upper_breakpoint_bound
            )

            # Although not in Muggeo (2003), the optimal momentum is computed
            # using the updated breakpoints and the previous breakpoints.
            res = minimize_scalar(
                search_min, bracket=(0, 1), bounds=(0, 1),
                args=(new_breakpoints, breakpoints, ones_x_mat, y)
            )
            opt_momentum = res.x
            res_loss = res.fun
            breakpoints = opt_momentum * new_breakpoints + (1 - opt_momentum) * breakpoints
            breakpoints = np.clip(
                breakpoints, self.lower_breakpoint_bound, self.upper_breakpoint_bound
            )
            epsilon = np.abs(rss - res_loss) / (rss + 0.1)
            rss = res_loss

            x_diff_bkpts = x - breakpoints[:, None]
            max0_vals = np.maximum(x_diff_bkpts, 0)
            design_mat = np.vstack((
                ones_x_mat, max0_vals, (max0_vals > 0).astype(np.float64)
            )).T

            if epsilon < tol:
                break

        pinv = np.linalg.pinv(design_mat)
        sol = pinv @ y
        residuals = np.sum((design_mat @ sol - y)**2)
        dof = self.len_data - design_mat.shape[1]
        cov_mat = (pinv @ pinv.T) * residuals / dof

        to_return = (
            intercept, slope, breakpoints, max0_params, indicator_params,
            residuals, cov_mat,
        )
        return to_return

    def fit(
        self,
        breakpoints: Sequence[float] = None,
        num_breakpoints: int = 1,
        maxiter: int = 30,
        tol: float = 1e-8,
        num_davies: int = 10,
        num_bootstraps: int = 10,
        seed: int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence = None
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
        num_bootstraps : int, default 0
            The number of bootstrap inferences for escaping local minima.
        seed : int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            The seed for the random number generator. If None, the random number
            generator used is self.rng.

        Returns
        -------
        None
        """
        if not num_bootstraps >= 0:
            raise ValueError('num_bootstraps must be >= 0.')
        if num_breakpoints < 1:
            raise ValueError('num_breakpoints must be >= 1.')
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        self.davies_x, self.davies_p = compute_davies(self.x, self.y, num_davies)

        best_rss = None
        best_params = None

        # TODO Write this better.
        if breakpoints is not None:
            if not hasattr(breakpoints, '__len__'):
                breakpoints = [breakpoints]
            breakpoints = np.sort(np.asarray(breakpoints))
            num_breakpoints = len(breakpoints)
            if self.len_data <= 2 + 2 * num_breakpoints:
                raise RuntimeError('Breakpoint analysis failed. Too few datapoints.')
            try:
                params = self._fit(
                    breakpoints, num_breakpoints, maxiter, tol
                )
            except:
                pass
            else:
                best_params = params
                y_est = _eval(self.x, *params[:4])
                best_rss = np.sum((self.y - y_est)**2)

        if self.len_data <= 2 + 2 * num_breakpoints:
            raise RuntimeError('Breakpoint analysis failed. Too few datapoints.')

        # TODO Generalize to multiple breakpoints and clean up.
        # Let number of breakpoints tested be chosen by the user.
        # Necessary anymore?
        if  num_breakpoints == 1:
            breakpoints = self.x[2:-2]
            design_mats = _create_design_matrices(self.x, breakpoints)
            pinvs = np.linalg.pinv(design_mats)
            sols = np.tensordot(pinvs, self.y, [2, 0])[..., None]
            max0_vals = np.maximum(self.x - breakpoints[:, None], 0)
            y_ests = sols[:, 0] + self.x * sols[:, 1] + max0_vals * sols[:, 2]
            rss = np.sum((y_ests - self.y)**2, 1)
            where_best = np.argmin(rss)
            best_rss = rss[where_best]
            best_sol = sols[where_best].ravel()
            breakpoints = np.array([breakpoints[where_best]])
            residuals = np.sum((design_mats[where_best] @ best_sol - self.y)**2)
            cov_mat = (pinvs[where_best] @ pinvs[where_best].T) * residuals / (self.len_data - 4)
            best_params = (best_sol[0], best_sol[1], breakpoints, best_sol[2:3], best_sol[3:4], best_rss, cov_mat)
        try:
            params = self._fit(
                breakpoints, num_breakpoints, maxiter, tol
            )
        except:
            pass
        else:
            rss = np.sum((_eval(self.x, *params[:4]) - self.y)**2)
            if rss < best_rss:
                best_params = params
                best_rss = rss

        # TODO Perturb the breakpoints if the best parameter values haven't been changing
        # for some amount of iterations. Muggeo uses 3.
        for _ in range(num_bootstraps):
            bootstrap_idxs = rng.integers(low=0, high=self.len_data, size=self.len_data)
            x_bootstrap = self.x[bootstrap_idxs]
            y_bootstrap = self.y[bootstrap_idxs]

            breakpoints = best_params[2]

            # Fit bootstrapped data.
            try:
                params_bootstrap = self._fit(
                    best_params[2], maxiter=maxiter, tol=tol, x=x_bootstrap, y=y_bootstrap
                )
            except Exception as e:
                random_breakpoints = np.sort(rng.uniform(
                    low=self.lower_breakpoint_bound,
                    high=self.upper_breakpoint_bound,
                    size=2
                ))
                params_bootstrap = [[], [], random_breakpoints]

            # Fit original data using parameters from bootstrap fit.
            try:
                params = self._fit(
                    params_bootstrap[2], maxiter=maxiter, tol=tol
                )
            except Exception as e:
                continue

            rss = np.sum((self.y - _eval(self.x, *params[:4]))**2)
            if rss < best_rss:
                best_params = params
                best_rss = rss

        self.intercept = best_params[0]
        self.slope = best_params[1]
        self.breakpoints = best_params[2]
        self.max0_params = best_params[3]
        self.indicator_params = best_params[4]
        self.residuals = best_params[5]
        self.cov_mat = best_params[6]

        (self.intercept_se, self.slope_se,
         self.max0_se, self.breakpoint_se) = compute_standard_errors(
             best_params[:5], self.cov_mat
         )

    def eval(
        self,
        x: Sequence[float]
    ) -> NDArray[np.float64]:
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
        return _eval(x, self.intercept, self.slope, self.breakpoints, self.max0_params)

    def print_segmented_input(
        self
    ) -> None:
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
        to_print = 'library(\'segmented\')\n'
        to_print += 'x <- c(' + str(list(self.x))[1:-1] + ')\n'
        to_print += 'y <- c(' + str(list(self.y))[1:-1] + ')\n'
        to_print += 'lm_fit <- lm(y~x, data=list(x, y))\n'
        to_print += 's <- segmented(lm_fit, npsi=1)\n'
        to_print += 'summary(s)\n' + 'davies.test(lm_fit)\n'
        to_print += 'sum(s$residual^2)\n'
        print(to_print)
