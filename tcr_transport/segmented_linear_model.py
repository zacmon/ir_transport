from typing import *

import numpy as np
from numpy.linalg import lstsq
from numpy.typing import NDArray

class SegmentedLinearModel(object):
    """
    Class to fit piecewise linear models by inferring breakpoints.
    """
    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
    ) -> None:
        """
        Initialize a SegmentedLinearModel object.

        Parameters
        ----------
        x : sequence of float
            Independent variables that will be used for fitting the model.
        y : sequence of float
            Ordinate variables that will be used for fitting the model.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if len(x) != len(y):
            raise RuntimeError(
                'x and y must have the same length.'
            )

        mask_finite = np.isfinite(x) & np.isfinite(y)
        self.x, self.y = x[mask_finite], y[mask_finite]
        self.x_min = x.min()
        self.x_max = x.max()

    def fit(
        self,
        breakpoints: Sequence[float],
        maxiter: int = 10,
        tol: float = 1e-8,
    ) -> Tuple[np.float64, NDArray[np.float64]]:
        """
        Fit a segmented linear model using the algorithm by Muggeo [1]_.

        Function is originally based off https://datascience.stackexchange.com/a/32833,
        and the segmented R package is at https://cran.r-project.org/web/packages/segmented/.

        Parameters
        ----------
        x : sequence of float
            Independent variables.
        y : sequence of float
            Ordinate variables.
        breakpoints : sequence of float
            The initial guesses for the breakpoints.
        maxiter : int, default 10
            Maximum number of iterations to perform.
        tol : float, optional
            Tolerance for convergence.

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
        if not hasattr(breakpoints, '__len__'):
            breakpoints = [breakpoints]
        breakpoints = np.sort(np.array(breakpoints))

        if breakpoints.min() <= self.x_min:
            raise RuntimeError(
                'The minimum breakpoint is at or below the minimum x. Ensure that '
                'the minimum initial guess is in (x_min, x_max).'
            )
        if breakpoints.max() >= self.x_max:
            raise RuntimeError(
                'The maximum breakpoint is at or beyond the maximum x. Ensure that'
                'the maximum initial guess is in (x_min, x_max).'
            )

        num_breakpoints = len(breakpoints)
        ones = np.ones_like(self.x)
        previous_delta = 0

        for _ in range(maxiter):
            x_diff_bkpts = self.x - breakpoints[:, None]
            max0_vals = np.maximum(x_diff_bkpts, 0)
            indicator_vals = (x_diff_bkpts > 0).astype(np.float64)
            coeff_mat = np.vstack((ones, self.x, max0_vals, indicator_vals))

            sol =  lstsq(coeff_mat.T, self.y, rcond=None)[0]

            intercept, slope = sol[0:2]
            max0_params = sol[2:2 + num_breakpoints]
            indicator_params = sol[2 + num_breakpoints:]

            delta = indicator_params / max0_params

            # Algorithm is oscillating but has essentially converged.
            if np.allclose(np.abs(previous_delta), np.abs(delta)):
                break

            loss = np.max(np.abs(indicator_params))
            # Change in breakpoint is within precision tolerance of convergence.
            if loss < tol:
                break

            breakpoints = breakpoints - delta
            previous_delta = delta

        if breakpoints.min() < self.x_min:
            raise RuntimeError(
                'The minimum inferred breakpoint is smaller than the given x. '
                'Is the model over- or underparameterized? Are there better values '
                'for the initial breakpoint guesses? Inspect y vs. x to get '
                'a better sense of how many breakpoints there might be and at what '
                'values the breakpoints should be initialized.'
            )
        if breakpoints.max() > self.x_max:
            raise RuntimeError(
                'The maximum inferred breakpoint is larger than the given x. '
                'Is the model over- or underparameterized? Are there better values '
                'for the initial breakpoint guesses? Inspect y vs. x to get '
                'a better sense of how many breakpoints there might be and at what '
                'values the breakpoints should be initialized.'
            )

        self.breakpoints = breakpoints
        self.max0_params = max0_params
        self.intercept = intercept
        self.slope = slope
        self.indicator_params = indicator_params

        return intercept, slope, breakpoints, max0_params, indicator_params

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
        x = np.asarray(x)
        max0_vals = np.maximum(x - self.breakpoints[:, None], 0)
        return self.intercept + self.slope * x + np.sum(max0_vals * self.max0_params[:, None], 0)
