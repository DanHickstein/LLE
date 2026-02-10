"""
Analytical benchmarks for the LLE solver.

Three tests that validate the numerical solver against known analytical
results of the Lugiato-Lefever Equation:

1. CW Steady State — exact cubic equation for intracavity power
2. Bright Soliton Profile — approximate sech shape, width, and amplitude
3. Modulational Instability — exact linear growth rates

Run:
    python test_lle_analytical.py

Generates: benchmark_analytical.png with a 3-panel comparison figure.

References:
    - Lugiato & Lefever, Phys. Rev. Lett. 58, 2209 (1987)
    - Coen et al., Opt. Lett. 38, 37 (2013)
    - Godey et al., Phys. Rev. A 89, 063814 (2014)
    - Herr et al., Nature Photonics 8, 145 (2014)
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from scipy.optimize import curve_fit

from lle import LLE

# ══════════════════════════════════════════════════════════════════════
# Analytical helper functions
# ══════════════════════════════════════════════════════════════════════

def cw_steady_state_roots(f_squared, alpha):
    """Solve the cubic: |F|^2 = rho * [1 + (alpha - rho)^2].

    Returns the real positive roots for the intracavity power rho = |A|^2.

    Parameters
    ----------
    f_squared : float
        Squared pump amplitude |F|^2.
    alpha : float
        Normalized detuning.

    Returns
    -------
    roots : array of real positive roots, sorted ascending.
    """
    # Expand: f^2 = rho * (1 + alpha^2 - 2*alpha*rho + rho^2)
    #       = rho^3 - 2*alpha*rho^2 + (1 + alpha^2)*rho
    # So: rho^3 - 2*alpha*rho^2 + (1 + alpha^2)*rho - f^2 = 0
    coeffs = [1, -2 * alpha, 1 + alpha**2, -f_squared]
    all_roots = np.roots(coeffs)

    # Keep only real positive roots
    real_roots = []
    for r in all_roots:
        if np.abs(r.imag) < 1e-10 and r.real > 0:
            real_roots.append(r.real)

    return np.sort(real_roots)


def cw_steady_state_field(rho, alpha, f):
    """Return the complex CW steady-state field A0 given rho = |A0|^2.

    From: (1 + i*(alpha - rho)) * A0 = F, so A0 = F / (1 + i*(alpha - rho))
    """
    return f / (1 + 1j * (alpha - rho))


def mi_growth_rate(rho0, alpha, d2, l):
    """Analytical MI growth rate for perturbation mode number l.

    lambda_+(l) = -1 + sqrt(rho0^2 - sigma_l^2)  if discriminant > 0
    lambda_+(l) = -1  (imaginary eigenvalues) otherwise

    Parameters
    ----------
    rho0 : float
        CW steady-state power |A0|^2.
    alpha : float
        Detuning.
    d2 : float
        Anomalous dispersion magnitude (= |beta_2|, positive).
    l : int or array
        Mode number(s).

    Returns
    -------
    growth_rate : float or array
        Real part of the most-unstable eigenvalue.
    """
    sigma = alpha - 2 * rho0 + d2 * np.asarray(l, dtype=float)**2 / 2
    discriminant = rho0**2 - sigma**2
    
    # Avoid RuntimeWarning by only taking sqrt of positive elements
    growth = np.full_like(discriminant, -1.0)
    pos_mask = discriminant > 0
    growth[pos_mask] = -1 + np.sqrt(discriminant[pos_mask])

    return growth


# ══════════════════════════════════════════════════════════════════════
# Test 1: CW Steady State
# ══════════════════════════════════════════════════════════════════════

def test_cw_steady_state():
    """Verify convergence to the analytical CW steady state."""
    print("=" * 60)
    print("TEST 1: CW Steady State")
    print("=" * 60)

    Ntheta = 128  # small grid (no spatial structure needed)
    tphot_end = 100  # long enough to converge
    results = []

    # Test across multiple detuning and pump values
    test_cases = [
        # (alpha, f, description)
        (0.0, 1.0, "zero detuning"),
        (1.0, 1.5, "moderate detuning"),
        (2.0, 2.0, "bistable regime, lower branch"),
        (3.0, 2.5, "deep bistable, lower branch"),
        (5.0, 4.0, "large detuning"),
    ]

    all_passed = True

    for alpha, f, desc in test_cases:
        F = np.full(Ntheta, f, dtype=complex)
        beta = np.array([0.0])  # no dispersion

        result = LLE(
            Ntheta=Ntheta,
            alpha=alpha,
            beta=beta,
            F=F,
            tphot_end=tphot_end,
            nsaves=10,
            min_field=0,
            print_status=False,
        )

        rho_num = np.mean(np.abs(result.A_t[-1])**2)
        roots = cw_steady_state_roots(f**2, alpha)
        rho_exact = roots[0]  # lower branch (cold start)

        rel_error = abs(rho_num - rho_exact) / rho_exact
        passed = rel_error < 1e-3
        if not passed:
            all_passed = False

        results.append({
            'alpha': alpha, 'f': f, 'rho_num': rho_num,
            'rho_exact': rho_exact, 'rel_error': rel_error,
            'passed': passed, 'desc': desc, 'roots': roots,
        })

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] α={alpha}, f={f} ({desc})")
        print(f"         ρ_num = {rho_num:.6f}, ρ_exact = {rho_exact:.6f}, "
              f"rel_err = {rel_error:.2e}")

    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}\n")
    return results, all_passed


# ══════════════════════════════════════════════════════════════════════
# Test 2: Bright Soliton Profile
# ══════════════════════════════════════════════════════════════════════

def test_soliton_profile():
    """Verify soliton matches the analytical sech profile."""
    print("=" * 60)
    print("TEST 2: Bright Soliton Profile")
    print("=" * 60)

    # Parameters: CW pump, anomalous GVD, robust soliton regime
    alpha = 4.0
    d2 = 0.01  # anomalous dispersion magnitude
    beta2 = -d2  # our convention: beta[1] = beta_2, negative = anomalous
    f = np.sqrt(6.0)  # CW pump amplitude (f^2 = 6)

    Ntheta = 1024
    dtheta = 2 * np.pi / Ntheta
    theta = np.arange(-Ntheta // 2, Ntheta // 2) * dtheta

    # Find the CW background (lower branch)
    # Define the fitting function
    def sech2_func(theta, B_fit, w_fit, theta_0, bg_level):
        return bg_level + B_fit**2 / np.cosh((theta - theta_0) / w_fit)**2

    # Run two cases: one standard (alpha=4), one high-alpha (alpha=7)
    # The LLE soliton approaches the NLS sech profile as alpha increases.
    cases = [
        {'alpha': 4.0, 'f': float(np.sqrt(6.0)), 'desc': 'Standard'},
        {'alpha': 7.0, 'f': 3.0, 'desc': 'High-alpha (Stable)'},
    ]

    all_passed = True
    best_result = None

    for case in cases:
        alpha = case['alpha']
        f = case['f']
        desc = case['desc']
        
        # Recalculate parameters
        roots = cw_steady_state_roots(f**2, alpha)
        rho_bg = roots[0]
        # In multi-root region, ensure we're on lower branch
        if len(roots) > 1:
             rho_bg = roots[0]
        
        A_bg = cw_steady_state_field(rho_bg, alpha, f)

        Delta = alpha - 2 * rho_bg
        B_analytical = np.sqrt(2 * Delta)
        w_analytical = np.sqrt(d2 / (2 * Delta))

        print(f"\n  Case: {desc}")
        print(f"  Parameters: α={alpha}, d₂={d2}, f={f:.4f}")
        print(f"  CW background: ρ_bg = {rho_bg:.6f}, A_bg = {A_bg:.6f}")
        print(f"  Analytical: Δ = {Delta:.4f}, B = {B_analytical:.4f}, "
              f"w = {w_analytical:.6f}")

        # Initial condition
        sech_profile = 1.2 * B_analytical * np.reciprocal(np.cosh(theta / w_analytical))
        A0 = A_bg + sech_profile * (-1j)

        F = np.full(Ntheta, f, dtype=complex)
        
        # Define beta
        beta2 = -d2
        beta = np.array([0.0, beta2])
        
        # Sufficient time for convergence
        tphot_end = 200
        
        result = LLE(
            Ntheta=Ntheta,
            alpha=alpha,
            beta=beta,
            F=F,
            tphot_end=tphot_end,
            nsaves=50,
            A0=A0,
            min_field=0,
            print_status=True,
        )

        # Get the final profile
        I_final = np.abs(result.A_t[-1])**2
        I_bg_num = np.min(I_final)

        # Better fitting using curve_fit
        # Initial guess
        p0 = [B_analytical, w_analytical, 0.0, rho_bg]
        try:
            peak_idx = np.argmax(I_final)
            peak_theta = theta[peak_idx]
            # Re-center efficiently
            I_shifted = np.roll(I_final, Ntheta//2 - peak_idx)
            
            # Simple bounds to prevent unphysical fits
            # B > 0, w > 0, theta_0 approx 0, bg > 0
            bounds = ([0, 0, -np.pi, 0], [np.inf, np.inf, np.pi, np.inf])
            
            popt, pcov = curve_fit(sech2_func, theta, I_shifted, p0=p0, bounds=bounds, maxfev=10000)
            B_fit = popt[0]
            w_fit = popt[1]
        except Exception as e:
            print(f"    Fitting failed: {e}")
            B_fit = 0.0
            w_fit = 1.0
            peak_theta = 0.0

        # Check agreement
        B_ratio = B_fit / B_analytical
        w_ratio = w_fit / w_analytical

        # Pass criteria
        # High alpha (alpha=7) should be closer to NLS limit (tol~0.15)
        # Standard alpha=4 deviates significantly (tol~0.40)
        tol = 0.15 if alpha > 6 else 0.40
        B_passed = abs(1 - B_ratio) < tol
        w_passed = abs(1 - w_ratio) < tol
        
        case_passed = B_passed and w_passed
        if not case_passed:
            all_passed = False

        print(f"    B_fit = {B_fit:.4f} (analytical: {B_analytical:.4f}, "
              f"ratio: {B_ratio:.3f}) {'PASS' if B_passed else 'FAIL'}")
        print(f"    w_fit = {w_fit:.6f} (analytical: {w_analytical:.6f}, "
              f"ratio: {w_ratio:.3f}) {'PASS' if w_passed else 'FAIL'}")

        # Store the best result for plotting (or the last one)
        # Let's store the High-alpha case as it looks nicer, or consistent with previous plots?
        # Actually, let's store the Standard case (alpha=4) to match the previous plot style,
        # but with improved fitting.
        if alpha == 4.0:
             best_result = {
                'theta': theta, 'I_final': I_final, 'I_bg_num': I_bg_num,
                'B_analytical': B_analytical, 'w_analytical': w_analytical,
                'B_fit': B_fit, 'w_fit': w_fit, 'rho_bg': rho_bg,
                'A_bg': A_bg, 'alpha': alpha, 'd2': d2, 'f': f,
                'peak_theta': peak_theta, 'passed': case_passed,
            }

    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}\n")

    return best_result or {}


# ══════════════════════════════════════════════════════════════════════
# Test 3: Modulational Instability Growth Rate
# ══════════════════════════════════════════════════════════════════════

def test_mi_growth_rate():
    """Verify MI growth rates match the analytical linear stability analysis."""
    print("=" * 60)
    print("TEST 3: Modulational Instability Growth Rate")
    print("=" * 60)

    # Parameters: set up so MI occurs
    alpha = 0.0
    d2 = 0.001  # Smaller dispersion to put unstable modes at higher k
    beta2 = -d2

    # Choose pump to give rho0 = 1.5 (threshold is rho0=1)
    rho0_target = 1.5
    f_squared = rho0_target * (1 + (alpha - rho0_target)**2)
    f = np.sqrt(f_squared)

    # Verify CW solution
    roots = cw_steady_state_roots(f_squared, alpha)
    rho0 = roots[0] if len(roots) == 1 else roots[-1]

    print(f"  Parameters: α={alpha}, d₂={d2}, f={f:.4f}")
    print(f"  CW steady state: ρ₀ = {rho0:.6f}")

    # The most unstable mode l_max^2 = 2(2rho - alpha)/d2
    l_max_sq = max(0, 2 * (2 * rho0 - alpha) / d2)
    l_max = np.sqrt(l_max_sq)
    # The growth rate for l_max is gamma = -1 + rho0
    gamma_max = -1 + rho0
    print(f"  Most unstable mode: l_max ≈ {l_max:.1f}")
    print(f"  Max growth rate: γ_max = {gamma_max:.4f}")

    Ntheta = 512
    dtheta = 2 * np.pi / Ntheta
    
    # Use explicit k vector (LLE uses kfft which wraps, we need physical k)
    k = np.concatenate([np.arange(0, Ntheta // 2),
                        np.arange(-Ntheta // 2, 0)])

    # CW initial condition + single-mode perturbation at l_max
    # Seeding broad noise is tricky because random phase projections.
    # Let's seed white noise with larger amplitude to reduce initial transient.
    A_bg = cw_steady_state_field(rho0, alpha, f)
    perturbation = 1e-4 * (np.random.randn(Ntheta) + 1j * np.random.randn(Ntheta))
    A0 = np.full(Ntheta, A_bg, dtype=complex) + perturbation

    F = np.full(Ntheta, f, dtype=complex)
    beta = np.array([0.0, beta2])

    # Run for short time to measure linear growth
    # Growth time constant is 1/gamma ~ 2.0.
    # Run for 15 lifetimes to see clear growth
    tphot_end = 15.0
    nsaves = 200

    result = LLE(
        Ntheta=Ntheta,
        alpha=alpha,
        beta=beta,
        F=F,
        tphot_end=tphot_end,
        nsaves=nsaves,
        A0=A0,
        min_field=0,
        print_status=False,
    )

    # Measure growth rates from the spectral evolution
    A_bg_k = fft(np.full(Ntheta, A_bg, dtype=complex))

    mode_amplitudes = np.zeros((nsaves, Ntheta))
    for i in range(nsaves):
        delta_Ak = result.A_k[i] - A_bg_k
        mode_amplitudes[i] = np.abs(delta_Ak)

    t_array = result.t

    # Fit window: wait for transient to die, fit before saturation
    t_fit_mask = (t_array > 3.0) & (t_array < 10.0)
    t_fit = t_array[t_fit_mask]

    measured_growth = np.full(Ntheta, np.nan)
    
    for l_idx in range(Ntheta):
        amp = mode_amplitudes[t_fit_mask, l_idx]
        if np.mean(amp) > 1e-7:  # above numerical noise
            # Fit log(amplitude) vs time
            log_amp = np.log(amp + 1e-20)
            A_matrix = np.vstack([t_fit, np.ones_like(t_fit)]).T
            slope, _ = np.linalg.lstsq(A_matrix, log_amp, rcond=None)[0]
            measured_growth[l_idx] = slope

    # Analytical growth rates
    analytical_growth = mi_growth_rate(rho0, alpha, d2, k)

    # Compare for the MI band
    mi_band = analytical_growth > 0
    l_mi = np.abs(k[mi_band])
    gamma_analytical = analytical_growth[mi_band]
    gamma_numerical = measured_growth[mi_band]

    # Report results
    valid = np.isfinite(gamma_numerical)
    n_valid = np.sum(valid)
    
    if n_valid > 5:
        # Check relative error near the peak
        peak_mask = gamma_analytical > 0.5 * np.max(gamma_analytical)
        if np.sum(peak_mask) > 0:
            rel_errors = np.abs(gamma_numerical[peak_mask] - gamma_analytical[peak_mask]) / \
                         np.abs(gamma_analytical[peak_mask])
            mean_rel_error = np.mean(rel_errors)
            passed = mean_rel_error < 0.10
        else:
            mean_rel_error = np.inf
            passed = False
    else:
        mean_rel_error = np.inf
        passed = False

    print(f"\n  MI band modes measured: {n_valid} / {np.sum(mi_band)}")
    if n_valid > 0:
        print(f"  Mean relative error (peak modes): {mean_rel_error:.4f}")
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}\n")

    return {
        'k': k, 'analytical_growth': analytical_growth,
        'measured_growth': measured_growth, 'rho0': rho0,
        'alpha': alpha, 'd2': d2, 'l_max': l_max,
        'gamma_max': gamma_max, 'passed': passed,
    }


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════

def plot_results(cw_results, sol_results, mi_results):
    """Generate the 3-panel benchmark figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.2})

    # ── Panel (a): CW Steady State ──────────────────────────────
    ax = axes[0]

    # Plot the analytical bistability curve for one alpha value
    alpha_demo = 4.0
    f_range = np.linspace(0.01, 6, 500)
    rho_curve = []
    for fi in f_range:
        roots = cw_steady_state_roots(fi**2, alpha_demo)
        for r in roots:
            rho_curve.append((fi**2, r))

    rho_curve = np.array(rho_curve)
    ax.plot(rho_curve[:, 0], rho_curve[:, 1], 'b-', linewidth=1.5,
            label=f'Analytical (α={alpha_demo})', zorder=1)

    # Plot the numerical data points
    for r in cw_results:
        marker = 'o' if r['passed'] else 'x'
        color = 'red' if r['alpha'] == alpha_demo else 'green'
        ax.plot(r['f']**2, r['rho_num'], marker, color=color, markersize=10,
                markeredgewidth=2, zorder=3)

    # Also show the analytical curve for alpha=2
    alpha_demo2 = 2.0
    rho_curve2 = []
    for fi in f_range:
        roots = cw_steady_state_roots(fi**2, alpha_demo2)
        for r_root in roots:
            rho_curve2.append((fi**2, r_root))
    rho_curve2 = np.array(rho_curve2)
    ax.plot(rho_curve2[:, 0], rho_curve2[:, 1], 'g--', linewidth=1.5,
            label=f'Analytical (α={alpha_demo2})', zorder=1)

    ax.set_xlabel(r'$|F|^2$', fontsize=13)
    ax.set_ylabel(r'$\rho = |A_0|^2$', fontsize=13)
    ax.set_title('(a) CW Steady State', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 8)

    # ── Panel (b): Soliton Profile ──────────────────────────────
    ax = axes[1]

    theta = sol_results['theta']
    I_final = sol_results['I_final']
    peak_theta = sol_results['peak_theta']

    # Re-center
    theta_plot = theta - peak_theta
    theta_plot = np.angle(np.exp(1j * theta_plot))
    sort_idx = np.argsort(theta_plot)

    # Analytical sech^2
    Delta = sol_results['alpha'] - 2 * sol_results['rho_bg']
    B_a = sol_results['B_analytical']
    w_a = sol_results['w_analytical']
    rho_bg = sol_results['rho_bg']

    theta_fine = np.linspace(-0.15, 0.15, 1000)
    I_analytical = rho_bg + B_a**2 / np.cosh(theta_fine / w_a)**2

    ax.plot(theta_plot[sort_idx], I_final[sort_idx], 'b-', linewidth=1.5,
            label='Numerical')
    ax.plot(theta_fine, I_analytical, 'r--', linewidth=2,
            label=f'Analytical sech² (B={B_a:.2f}, w={w_a:.4f})')
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(0, max(np.max(I_final) * 1.1, B_a**2 + rho_bg + 1))
    ax.set_xlabel(r'$\theta$ (rad)', fontsize=13)
    ax.set_ylabel(r'$|A|^2$', fontsize=13)
    ax.set_title('(b) Soliton Profile', fontsize=14)
    ax.legend(fontsize=10)

    # Add fit info
    ax.text(0.03, 0.95,
            f'B_num/B_ana = {sol_results["B_fit"]/B_a:.3f}\n'
            f'w_num/w_ana = {sol_results["w_fit"]/w_a:.3f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ── Panel (c): MI Growth Rate ──────────────────────────────
    ax = axes[2]

    k_vals = mi_results['k']
    gamma_a = mi_results['analytical_growth']
    gamma_n = mi_results['measured_growth']

    # Only plot positive k (symmetry)
    pos_mask = k_vals >= 0
    k_pos = k_vals[pos_mask]
    gamma_a_pos = gamma_a[pos_mask]
    gamma_n_pos = gamma_n[pos_mask]

    ax.plot(k_pos, gamma_a_pos, 'r-', linewidth=2, label='Analytical')
    valid = np.isfinite(gamma_n_pos)
    ax.plot(k_pos[valid], gamma_n_pos[valid], 'bo', markersize=4,
            label='Numerical', alpha=0.7)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Mode number $l$', fontsize=13)
    ax.set_ylabel(r'Growth rate Re($\lambda$)', fontsize=13)
    ax.set_title('(c) MI Growth Rate', fontsize=14)
    ax.legend(fontsize=10)

    # Zoom to the MI band
    l_max = mi_results['l_max']
    ax.set_xlim(0, min(l_max * 2, Ntheta // 2))

    fig.tight_layout()
    fig.savefig('benchmark_analytical.png', dpi=200, bbox_inches='tight')
    print(f"Saved benchmark_analytical.png")

    return fig


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═" * 60)
    print("  LLE Analytical Benchmarks")
    print("═" * 60 + "\n")

    # Run all tests
    cw_results, cw_passed = test_cw_steady_state()
    sol_results = test_soliton_profile()
    mi_results = test_mi_growth_rate()

    # Summary
    print("═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    print(f"  Test 1 (CW Steady State):   {'PASS ✓' if cw_passed else 'FAIL ✗'}")
    print(f"  Test 2 (Soliton Profile):    {'PASS ✓' if sol_results['passed'] else 'FAIL ✗'}")
    print(f"  Test 3 (MI Growth Rate):     {'PASS ✓' if mi_results['passed'] else 'FAIL ✗'}")
    all_passed = cw_passed and sol_results['passed'] and mi_results['passed']
    print(f"\n  All tests: {'PASS ✓' if all_passed else 'FAIL ✗'}")
    print("═" * 60 + "\n")

    # Generate plots
    Ntheta = 512  # for plot_results reference
    plot_results(cw_results, sol_results, mi_results)

    plt.show()
