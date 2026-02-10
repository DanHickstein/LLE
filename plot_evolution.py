"""
2D evolution plots for the LLE soliton simulation.

Shows the temporal and spectral evolution of the intracavity field over
time using imshow (waterfall) plots. Runs the simulation and generates
side-by-side 2D plots.

Usage:
    python plot_evolution.py
"""

import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from lle import LLE

# ══════════════════════════════════════════════════════════════════════
# Physical constants (same as run_soliton_pulsed.py)
# ══════════════════════════════════════════════════════════════════════

c = 2.9979e8
lc = 1550e-9
hbar = 1e-34
n0 = 2
fr = 33e9
n2 = 24e-20
mode_area = (1e-6)**2

tRT = 1 / fr
circumference = tRT * (c / n0)
fc = c / lc
wc = 2 * np.pi * fc
g0 = n2 * c * hbar * wc**2 / (n0**2 * circumference * mode_area)

# ══════════════════════════════════════════════════════════════════════
# Simulation setup
# ══════════════════════════════════════════════════════════════════════

Ntheta = 2 ** int(np.ceil(np.log2(16000)))
dtheta = 2 * np.pi / Ntheta
theta = np.arange(-Ntheta // 2, Ntheta // 2) * dtheta
k = np.concatenate([np.arange(0, Ntheta // 2), np.arange(-Ntheta // 2, 0)])
kfreq = fc + fr * k
kwave = c / kfreq

# LLE parameters
alpha = 2.5
F2 = 3.30
dPM = 7.5 * np.pi

F = np.cos(np.pi / 2 * np.sin(theta / 2)**2) * np.exp(1j * dPM * np.cos(theta))
from numpy.fft import fft, ifft
F = fftshift(ifft(np.abs(fft(F))))
F = F / np.max(np.abs(F)) * np.sqrt(F2)

pump_power = 5e-3
dwo = np.sqrt(pump_power / np.mean(np.abs(F)**2) * 8 * g0 / (hbar * wc))

# Additional phase modulation
F = F * np.exp(1j * np.cos(theta) * 10 * np.pi)

# Dispersion
DWb = 1120e-9
DWr = 2 * DWb
DWbk = (c / DWb - c / lc) / fr
DWrk = (c / DWr - c / lc) / fr
Nc = (DWbk + DWrk) / 2
Ns = DWbk - Nc

beta = np.zeros(7)
beta[0] = -7.3e-4
beta[1] = -8e-5
beta[3] = 12 * beta[1] / (Nc**2 - Ns**2)
beta[2] = -1 * beta[3] * Nc / 2
beta[4] = 1e-13
beta[5] = 6e-16
beta[6] = -2.2e-18

min_field = np.sqrt(2 * g0 / dwo * 0.5)

# ══════════════════════════════════════════════════════════════════════
# Run simulation with many save points for smooth 2D plots
# ══════════════════════════════════════════════════════════════════════

tphot_end = 200
nsaves = 500  # more saves for smoother 2D plots

print(f"Running LLE simulation ({nsaves} snapshots)...\n")

result = LLE(
    Ntheta=Ntheta,
    alpha=alpha,
    beta=beta,
    F=F,
    tphot_end=tphot_end,
    nsaves=nsaves,
    min_field=min_field,
    h_init=5e-3,
    tol=5e-6,
    print_status=True,
)

# ══════════════════════════════════════════════════════════════════════
# Prepare data for plotting
# ══════════════════════════════════════════════════════════════════════

# Time axis (in fs)
Tfs = theta / np.pi * tRT * 1e15

# Temporal intensity evolution: |A(theta, t)|^2
I_evo = result.intensity  # shape: (nsaves, Ntheta)

# Spectral intensity evolution (in dB)
spec_evo = np.abs(result.A_k) ** 2
spec_evo_dB = 10 * np.log10(spec_evo + 1e-300)
# Normalize each row to its own peak
spec_evo_dB -= spec_evo_dB.max(axis=1, keepdims=True)

# Wavelength axis (need to sort for proper display)
wl_nm = kwave * 1e9
sort_idx = np.argsort(wl_nm)
wl_sorted = wl_nm[sort_idx]
spec_sorted = spec_evo_dB[:, sort_idx]

# Crop wavelength range for display
wl_mask = (wl_sorted >= 1050) & (wl_sorted <= 2450)
wl_display = wl_sorted[wl_mask]
spec_display = spec_sorted[:, wl_mask]

# Time array for y-axis
t_phot = result.t

# ══════════════════════════════════════════════════════════════════════
# 2D Evolution Plots
# ══════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Left: Temporal evolution ---
# Crop to ±1500 fs for display
t_mask = (Tfs >= -1500) & (Tfs <= 1500)
Tfs_display = Tfs[t_mask]
I_display = I_evo[:, t_mask]

im1 = ax1.imshow(
    I_display,
    aspect='auto',
    origin='lower',
    extent=[Tfs_display[0], Tfs_display[-1], t_phot[0], t_phot[-1]],
    cmap='inferno',
    interpolation='bilinear',
)
ax1.set_xlabel('Time (fs)', fontsize=14)
ax1.set_ylabel(r'$t$ (photon lifetimes)', fontsize=14)
ax1.set_title('Temporal Evolution', fontsize=15)
cb1 = fig.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)
cb1.set_label(r'$|A|^2$', fontsize=13)

# --- Right: Spectral evolution ---
im2 = ax2.imshow(
    spec_display,
    aspect='auto',
    origin='lower',
    extent=[wl_display[0], wl_display[-1], t_phot[0], t_phot[-1]],
    cmap='inferno',
    vmin=-80, vmax=0,
    interpolation='bilinear',
)
ax2.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
ax2.set_ylabel(r'$t$ (photon lifetimes)', fontsize=14)
ax2.set_title('Spectral Evolution', fontsize=15)
cb2 = fig.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
cb2.set_label('Power (dB)', fontsize=13)

fig.suptitle(
    f'LLE Soliton Formation ($\\alpha$={alpha}, $F^2$={F2}, '
    f'{tphot_end} photon lifetimes)',
    fontsize=16, y=0.98,
)

fig.tight_layout()
fig.savefig('soliton_evolution_2d.png', dpi=200, bbox_inches='tight')
print(f"\nSaved soliton_evolution_2d.png")

plt.show()
