"""
Lugiato-Lefever Equation (LLE) solver for driven-dissipative Kerr cavities.

Solves the normalized LLE:

    dA/dt = -(1 + i*alpha)*A + i*|A|^2*A + F(theta) + dispersion terms

where:
    A(theta, t)  = intracavity field envelope (theta: azimuthal angle)
    t            = time in units of photon lifetimes (1/dwo)
    alpha        = normalized pump-resonance detuning
    beta[j]      = j-th order dispersion coefficient (mode-number space)
    F(theta)     = pump waveform (can be CW or pulsed/EO comb)

Uses a symmetric split-step Fourier method with an adaptive-step RK4
integrator for the nonlinear part and Richardson extrapolation for error
control. This is the standard numerical approach for LLE simulations.

The linear part (dispersion, loss, detuning) is handled exactly via
exponentiation in Fourier space:

    L(k) = -(1 + i*alpha) + sum_j [i * beta_j / j! * k^j]

The nonlinear part is:

    N(A) = i * |A|^2 * A + F

References:
    - Chembo & Menyuk, Phys. Rev. A 87, 053852 (2013)
    - Coen et al., Opt. Lett. 38, 37 (2013)
"""

from __future__ import annotations

import time as timer
from dataclasses import dataclass
from math import factorial

import numpy as np

# Use scipy.fft if available (supports workers=-1 for parallel FFTs),
# fall back to numpy.fft otherwise.
try:
    from scipy.fft import fft, ifft
    _HAS_SCIPY_FFT = True
except ImportError:
    from numpy.fft import fft, ifft
    _HAS_SCIPY_FFT = False


@dataclass
class LLEResult:
    """Results of an LLE simulation.

    Attributes
    ----------
    t : 1D array (nsaves,)
        Time values in photon lifetimes at each saved snapshot.
    theta : 1D array (Ntheta,)
        Azimuthal angle grid.
    k : 1D array (Ntheta,)
        Mode numbers in FFT order.
    A_t : 2D array (nsaves, Ntheta)
        Intracavity field in the theta (time) domain at each snapshot.
    A_k : 2D array (nsaves, Ntheta)
        Intracavity field in the spectral (mode-number) domain at each
        snapshot (FFT-ordered, not fftshifted).
    F : 1D array (Ntheta,)
        The pump waveform used.
    alpha : float
        Normalized detuning.
    beta : 1D array
        Dispersion coefficients.
    lin_operator : 1D array (Ntheta,)
        The linear operator L(k) used in the simulation.
    elapsed_time : float
        Wall-clock time for the integration in seconds.
    """

    t: np.ndarray
    theta: np.ndarray
    k: np.ndarray
    A_t: np.ndarray
    A_k: np.ndarray
    F: np.ndarray
    alpha: float
    beta: np.ndarray
    lin_operator: np.ndarray
    elapsed_time: float

    @property
    def intensity(self) -> np.ndarray:
        """Intracavity intensity |A|^2 at each snapshot."""
        return np.abs(self.A_t) ** 2

    @property
    def spectrum_dB(self) -> np.ndarray:
        """Power spectrum in dB (normalized to peak) at each snapshot."""
        spec = np.abs(self.A_k) ** 2
        spec_db = 10 * np.log10(spec + 1e-300)
        spec_db -= spec_db.max(axis=1, keepdims=True)
        return spec_db


def _rk4_step(A, F, h, exp_half):
    """One symmetric split-step with RK4 for the nonlinear part.

    Applies: half-linear -> full-nonlinear(RK4) -> half-linear

    Parameters
    ----------
    A : complex array
        Field in theta domain at start of step.
    F : complex array
        Pump waveform in theta domain.
    h : float
        Step size in photon lifetimes.
    exp_half : complex array
        Pre-computed exp(L * h/2) for the linear half-step.

    Returns
    -------
    A_new : complex array
        Field in theta domain at end of step.
    """
    # Half-step linear (Fourier space)
    AI = ifft(exp_half * fft(A))

    # RK4 for nonlinear + pump: dA/dt = i|A|^2*A + F
    # k1 includes the linear half-step on the initial nonlinear term
    k1 = ifft(exp_half * fft(h * (1j * np.abs(A) ** 2 * A + F)))

    Ak2 = AI + k1 * 0.5
    k2 = h * (1j * np.abs(Ak2) ** 2 * Ak2 + F)

    Ak3 = AI + k2 * 0.5
    k3 = h * (1j * np.abs(Ak3) ** 2 * Ak3 + F)

    Ak4 = ifft(exp_half * fft(AI + k3))
    k4 = h * (1j * np.abs(Ak4) ** 2 * Ak4 + F)

    # Combine RK4 + second half-linear step
    return ifft(exp_half * fft(AI + k1 / 6 + k2 / 3 + k3 / 3)) + k4 / 6


def LLE(
    Ntheta: int,
    alpha: float,
    beta: np.ndarray,
    F: np.ndarray,
    tphot_end: float,
    nsaves: int = 200,
    A0: np.ndarray | None = None,
    min_field: float = 0.0,
    h_init: float = 5e-3,
    tol: float = 5e-6,
    print_status: bool = True,
) -> LLEResult:
    """Simulate a Kerr microresonator using the Lugiato-Lefever Equation.

    Uses a symmetric split-step Fourier method with adaptive-step RK4
    and Richardson extrapolation for error control.

    Parameters
    ----------
    Ntheta : int
        Number of angular grid points (should be a power of 2).
    alpha : float
        Normalized pump-resonance detuning. Positive = red-detuned
        (soliton regime).
    beta : array-like
        Dispersion coefficients. beta[0] = 1st order (freq offset),
        beta[1] = 2nd order (GVD), etc.
    F : 1D complex array, length Ntheta
        Pump waveform in theta domain.
    tphot_end : float
        Total simulation time in photon lifetimes.
    nsaves : int
        Number of equally-spaced snapshots to save.
    A0 : 1D complex array, optional
        Initial field. Defaults to zeros.
    min_field : float
        Quantum noise floor per spectral mode. Set to 0 to disable.
    h_init : float
        Initial step size in photon lifetimes.
    tol : float
        Error tolerance for adaptive step-size control.
    print_status : bool
        Print progress during integration.

    Returns
    -------
    result : LLEResult
    """
    beta = np.asarray(beta, dtype=float)
    F = np.asarray(F, dtype=complex)

    if F.shape[0] != Ntheta:
        raise ValueError(
            f"Pump F has length {F.shape[0]}, expected {Ntheta}"
        )

    # -- Build grids --
    dtheta = 2 * np.pi / Ntheta
    theta = np.arange(-Ntheta // 2, Ntheta // 2) * dtheta
    k = np.concatenate([np.arange(0, Ntheta // 2),
                        np.arange(-Ntheta // 2, 0)])  # FFT order

    # -- Linear operator L(k) --
    lin_op = -(1.0 + 1j * alpha) * np.ones(Ntheta, dtype=complex)
    for j in range(len(beta)):
        order = j + 1
        lin_op += 1j * beta[j] / factorial(order) * k.astype(float) ** order

    # -- Initial condition --
    if A0 is None:
        A = np.zeros(Ntheta, dtype=complex)
    else:
        A = np.asarray(A0, dtype=complex).copy()

    # -- Storage --
    save_interval = tphot_end / nsaves
    A_t_all = np.zeros((nsaves, Ntheta), dtype=complex)
    A_k_all = np.zeros((nsaves, Ntheta), dtype=complex)
    t_all = np.zeros(nsaves)

    A_t_all[0] = A.copy()
    A_k_all[0] = fft(A)
    save_idx = 1
    next_save = save_interval

    # -- Adaptive stepping --
    tphot = 0.0
    h = h_init / 2  # will be doubled on first iteration
    iteration = 0

    # Cache for the exponential (recomputed only when h changes)
    cached_h_half = None
    exp_half = None
    exp_full = None

    # Pre-compute constants for step adjustment
    grow = 2 ** (1.0 / 3)
    shrink = 2 ** (-1.0 / 3)

    start = timer.time()

    while tphot < tphot_end:
        iteration += 1
        h *= 2  # undo previous halving

        error_val = 1.0

        while error_val > 2 * tol:
            h_half = h / 2

            # Recompute exponentials only when step size changes
            if cached_h_half != h_half:
                cached_h_half = h_half
                exp_half = np.exp(lin_op * h_half / 2)
                exp_full = np.exp(lin_op * h / 2)

            # -- One full step of size h --
            A1 = _rk4_step(A, F, h, exp_full)

            # -- Two half-steps of size h/2 --
            A2 = _rk4_step(A, F, h_half, exp_half)
            A2 = _rk4_step(A2, F, h_half, exp_half)

            # -- Richardson error estimate --
            norm_A2 = np.sqrt(np.sum(np.abs(A2) ** 2))
            if norm_A2 > 0:
                error_val = np.sqrt(
                    np.sum(np.abs(A2 - A1) ** 2)
                ) / norm_A2
            else:
                error_val = 0.0

            if error_val > 2 * tol:
                h = h_half  # reject step, halve and retry
                cached_h_half = None  # force recompute

        # Accept: Richardson-extrapolated result
        h_phot = h  # effective step
        tphot += h_phot

        Aft = fft(4.0 / 3 * A2 - 1.0 / 3 * A1)

        # Apply quantum noise floor
        if min_field > 0:
            below = np.abs(Aft) < min_field
            if np.any(below):
                noise = min_field * np.exp(
                    1j * 2 * np.pi * np.random.rand(np.sum(below))
                )
                Aft[below] = noise

        A = ifft(Aft)

        # Adaptive step adjustment
        h = (shrink if error_val > tol else grow) * h_half
        cached_h_half = None  # h changed, expire cache

        # Save snapshots
        while next_save <= tphot and save_idx < nsaves:
            A_t_all[save_idx] = A
            A_k_all[save_idx] = Aft
            t_all[save_idx] = tphot
            save_idx += 1
            next_save += save_interval

        if print_status and iteration % 500 == 0:
            pct = tphot / tphot_end * 100
            elapsed = timer.time() - start
            max_I = np.max(np.abs(A) ** 2)
            print(
                f"  {pct:5.1f}% - t_phot = {tphot:.1f}/{tphot_end:.0f} "
                f"- max(I) = {max_I:.4f} - iter {iteration} - "
                f"{elapsed:.1f}s"
            )

    # Fill any remaining save slots
    while save_idx < nsaves:
        A_t_all[save_idx] = A
        A_k_all[save_idx] = Aft
        t_all[save_idx] = tphot
        save_idx += 1

    elapsed = timer.time() - start
    if print_status:
        print(f"\nSimulation complete: {iteration} iterations in {elapsed:.1f}s")

    return LLEResult(
        t=t_all,
        theta=theta,
        k=k,
        A_t=A_t_all,
        A_k=A_k_all,
        F=F,
        alpha=alpha,
        beta=beta,
        lin_operator=lin_op,
        elapsed_time=elapsed,
    )
