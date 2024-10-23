from typing import *

import numpy as np
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
        raise RuntimeError(
            'x and y must have the same length.'
        )

    mask_finite = np.isfinite(x) & np.isfinite(y)
    return x[mask_finite], y[mask_finite]

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
    """
    num_breakpoints = (len(cov_mat) - 2) // 2
    variances = np.diagonal(cov_mat)

    intercept_se = np.sqrt(variances[0])
    max0_var = variances[2:num_breakpoints + 2]
    max0_se = np.sqrt(max0_var)

    indicator_var = variances[2 + num_breakpoints:]
    ratio = params[3] / params[2]
    idxs = np.arange(0, num_breakpoints)
    breakpoint_se = np.sqrt((
        indicator_var
        + max0_var * ratio**2
        - 2 * cov_mat[idxs + 2, idxs + 2 + num_breakpoints] * ratio
    ) / params[2]**2)

    slope_se = np.sqrt(
        [np.sum(cov_mat[1:k, 1:k]) for k in range(2, 3 + num_breakpoints)]
    )

    return intercept_se, slope_se, max0_se, breakpoint_se

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

    def _fit(
        self,
        breakpoints: Sequence[float] = None,
        num_breakpoints: int = 1,
        maxiter: int = 30,
        tol: float = 1e-5,
        x: Sequence[float] = None,
        y: Sequence[float] = None,
    ) -> Tuple[np.float64, NDArray[np.float64]]:
        """
        Fit a segmented linear model using the algorithm by Muggeo [1]_.

        Function is originally based off https://datascience.stackexchange.com/a/32833,
        and the segmented R package is at https://cran.r-project.org/web/packages/segmented/.

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

        if breakpoints is not None:
            breakpoints = np.sort(np.asarray(breakpoints))
        else:
            if num_breakpoints < 1:
                raise ValueError(
                    'num_breakpoints must be >= 1.'
                )
            breakpoints = np.quantile(
                x, np.linspace(0, 1, num_breakpoints + 1, False)[1:]
            )

        if breakpoints.min() <= x_min:
            raise RuntimeError(
                'The minimum breakpoint is at or below the minimum x. Ensure that '
                'the minimum initial guess is in (x_min, x_max).'
            )
        if breakpoints.max() >= x_max:
            raise RuntimeError(
                'The maximum breakpoint is at or beyond the maximum x. Ensure that'
                'the maximum initial guess is in (x_min, x_max).'
            )

        num_breakpoints = len(breakpoints)
        ones = np.ones_like(x)
        previous_delta = 0
        for _ in range(maxiter):
            x_diff_bkpts = x - breakpoints[:, None]
            max0_vals = np.maximum(x_diff_bkpts, 0)
            indicator_vals = (x_diff_bkpts > 0).astype(np.float64)
            coeff_mat = np.vstack((ones, x, max0_vals, indicator_vals))

            pinv = np.linalg.pinv(coeff_mat.T)
            sol = pinv @ y
            intercept, slope = sol[0:2]
            max0_params = sol[2:2 + num_breakpoints]

            indicator_params = sol[2 + num_breakpoints:]
            # TODO What to do when max0_params is 0?
            delta = indicator_params / max0_params

            y_est = (
                intercept + slope * x
                + np.sum(max0_vals * max0_params[:, None], 0)
            )

            # Algorithm is oscillating but has essentially converged.
            if np.allclose(np.abs(previous_delta), np.abs(delta)):
                break

            loss = np.max(np.abs(indicator_params))
            # Change in breakpoint is within precision tolerance of convergence.
            if loss < tol:
                break

            breakpoints = breakpoints - delta
            previous_delta = delta
            breakpoints = np.clip(
                breakpoints, self.lower_breakpoint_bound, self.upper_breakpoint_bound
            )

        residuals = np.sum((coeff_mat.T @ sol - y)**2)
        dof = x.shape[0] - coeff_mat.shape[0]
        cov_mat = (pinv @ pinv.T) * residuals / dof

        to_return = (
            intercept, slope, breakpoints, max0_params, indicator_params,
            residuals, cov_mat,
        )

        return to_return

    # TODO Implement Davies test.
    def fit(
        self,
        breakpoints: Sequence[float] = None,
        num_breakpoints: int = 1,
        maxiter: int = 30,
        tol: float = 1e-8,
        num_bootstraps: int = 10,
        num_restarts: int = 1000,
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
            raise ValueError(
                'num_bootstraps must be >= 0.'
            )
        if not num_restarts >= 0:
            raise ValueError(
                'num_restarts must be >= 0.'
            )

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        if breakpoints is not None:
            if not hasattr(breakpoints, '__len__'):
                breakpoints = [breakpoints]
            breakpoints = np.asarray(breakpoints)
            num_breakpoints = len(breakpoints)

        # Fit the original data.
        # TODO Make use of parallel routines for multiple starting points.
        params = None
        try:
            params = self._fit(
                breakpoints, num_breakpoints, maxiter, tol
            )
        except Exception as e:
            if num_restarts == 0:
                raise e
            # If the original guess or using quantiles doesn't converge,
            # use random points until something converges.
            tries = 0
            while tries < num_restarts:
                breakpoints = self.rng.uniform(
                    low=self.lower_breakpoint_bound, high=self.upper_breakpoint_bound,
                    size=num_breakpoints
                )
                try:
                    params = self._fit(
                        breakpoints, num_breakpoints, maxiter, tol
                    )
                except:
                    tries += 1
                else:
                    break

        if params is None:
            raise RuntimeError(
                'Breakpoint analysis failed. Try increasing num_restarts '
                'or run a davies test to see if any breakpoints might even '
                'exist in the data.'
            )

        score = np.sum((self.y - _eval(self.x, *params[:4]))**2)
        len_data = len(self.x)

        for _ in range(num_bootstraps):
            bootstrap_idxs = rng.integers(low=0, high=len_data, size=len_data)
            x_bootstrap = self.x[bootstrap_idxs]
            y_bootstrap = self.y[bootstrap_idxs]


            # Fit bootstrapped data.
            try:
                params_bootstrap = self._fit(
                    params[2], maxiter=maxiter, tol=tol, x=x_bootstrap, y=y_bootstrap
                )
            except:
                params_bootstrap = params

            # Fit original data using parameters from bootstrap fit.
            try:
                new_params = self._fit(
                    params_bootstrap[2], maxiter=maxiter, tol=tol
                )
            except Exception as e:
                continue

            new_score = np.sum((self.y - _eval(self.x, *new_params[:4]))**2)
            if new_score < score:
                params = new_params
                score = new_score

        self.intercept = params[0]
        self.slope = params[1]
        self.breakpoints = params[2]
        self.max0_params = params[3]
        self.indicator_params = params[4]
        self.residuals = params[5]
        self.cov_mat = params[6]

        (self.intercept_se, self.slope_se,
         self.max0_se, self.breakpoint_se) = compute_standard_errors(
             params[:5], self.cov_mat
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
        print(to_print)
