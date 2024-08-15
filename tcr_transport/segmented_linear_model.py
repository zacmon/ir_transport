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
    ) -> None:
        pass

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        breakpoints: Sequence[float],
        maxiter: int = 10,
        tol: float = 1e-2,
    ):
        """
        Fit a segmented linear model using the algorithm by Muggeo [1]_.

        Parameters
        ----------
        x : numpy.ndarray of numpy.float64
            Independent variables.
        y : numpy.ndarray of numpy.float64
            Ordinate variables.
        breakpoints : sequence of float
            The initial guesses for the breakpoints.
        maxiter : int, default 10
            How many iterations the optimization algorithm runs for.
        tol : float, optional
            The tolerance at which the algorithm is said to converge.

        Returns
        -------
        None

        References
        ----------
        .. [1] Muggeo V (2003) "Estimating regression models with unknown break-points."
               Statist. Med 22(19), 3055-3071, https://doi.org/10.1002/sim.1545
        """
        # Remove nonfinite values.
        mask_finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask_finite], y[mask_finite]

        breakpoints = np.sort(np.array(breakpoints))
        num_breakpoints = len(breakpoints)
        dt = np.min(np.diff(x))
        ones = np.ones_like(x)

        err = np.inf

        for i in range(maxiter):
            x_diff_bkpts = x - breakpoints[:, None]
            max0_vals = np.maximum(x_diff_bkpts, 0)
            indicator_vals = (x_diff_bkpts > 0).astype(np.float64)
            coeff_mat = np.vstack((ones, x, max0_vals, indicator_vals))

            sol =  lstsq(coeff_mat.T, y, rcond=None)[0]

            # Parameters identification:
            intercept, slope = sol[0:2]
            max0_params = sol[2:2 + num_breakpoints]
            indicator_params = sol[2 + num_breakpoints:]

            new_breakpoints = breakpoints - indicator_params / max0_params
            loss = np.max(np.abs(new_breakpoints - breakpoints) / breakpoints)
            if loss < tol:
                break

            breakpoints = new_breakpoints

        self.breakpoints = breakpoints
        self.max0_params = max0_params
        self.intercept = intercept
        self.slope = slope
        self.indicator_params = indicator_params

    def eval(
        self,
        x: NDArray[np.float64]
    ):
        """
        Compute an estimate of y for the given x using the fitted parameters.

        Parameters
        ----------
        x : numpy.ndarray of numpy.float64
            Independent variables.

        Returns
        -------
        numpy.ndarray of numpy.float64
            The estimated ordinate values using the model fit.
        """
        max0_vals = np.maximum(x - self.breakpoints[:, None], 0)
        return self.intercept + self.slope * x + np.sum(max0_vals * self.max0_params[:, None], 0)
