"""
Pulsed soliton microcomb simulation — Python port of DCLLE01_soliton_pulsed.m.

Simulates a silicon nitride microresonator driven by a phase-modulated
electro-optic (EO) comb pump, generating a dissipative Kerr soliton.

Physical parameters:
    - Material: Si3N4 (n2 = 24e-20 m^2/W, n0 = 2)
    - FSR: 33 GHz
    - Pump: 1550 nm, 5 mW
    - Phase modulation depth: 7.5*pi (+ additional 10*pi)
    - Detuning alpha = 2.5, F^2 = 3.30
    - Higher-order dispersion up to 7th order

Usage:
    python run_soliton_pulsed.py
"""

import numpy as np
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
import scipy.io

from lle import LLE

# ══════════════════════════════════════════════════════════════════════
# Physical constants and resonator parameters
# ══════════════════════════════════════════════════════════════════════

c = 2.9979e8          # speed of light (m/s)
lc = 1550e-9          # pump wavelength (m)
hbar = 1e-34          # reduced Planck constant (J·s)

# Resonator properties
n0 = 2                # refractive index (Si3N4)
fr = 33e9             # free spectral range / rep rate (Hz)
n2 = 24e-20           # nonlinear index (m^2/W)
mode_area = (1e-6)**2 # effective mode area (m^2)

# Derived quantities
tRT = 1 / fr                         # round-trip time (s)
circumference = tRT * (c / n0)       # resonator circumference (m)
fc = c / lc                          # pump frequency (Hz)
wc = 2 * np.pi * fc                  # pump angular frequency (rad/s)

# Single-photon nonlinear coupling rate
g0 = n2 * c * hbar * wc**2 / (n0**2 * circumference * mode_area)

print(f"=== Resonator Parameters ===")
print(f"  g0 = {g0:.5f}")
print(f"  FSR = {fr*1e-9:.0f} GHz")
print(f"  Circumference = {circumference*1e3:.2f} mm")

# ══════════════════════════════════════════════════════════════════════
# Simulation grid
# ══════════════════════════════════════════════════════════════════════

Ntheta = 2 ** int(np.ceil(np.log2(16000)))  # = 16384
dtheta = 2 * np.pi / Ntheta
theta = np.arange(-Ntheta // 2, Ntheta // 2) * dtheta
k = np.concatenate([np.arange(0, Ntheta // 2),
                    np.arange(-Ntheta // 2, 0)])  # FFT-ordered modes
kfreq = fc + fr * k
kwave = c / kfreq

print(f"  Ntheta = {Ntheta}")

# ══════════════════════════════════════════════════════════════════════
# LLE parameters (normalized)
# ══════════════════════════════════════════════════════════════════════

alpha = 2.5   # normalized detuning
F2 = 3.30     # normalized pump power |F|^2

# -- Pump waveform: phase-modulated EO comb --
dPM = 7.5 * np.pi  # phase modulation depth
F = np.cos(np.pi / 2 * np.sin(theta / 2)**2) * np.exp(1j * dPM * np.cos(theta))
Fch = F.copy()

# Symmetrize the pump spectrum (take abs in freq domain, then IFFT)
F = fftshift(ifft(np.abs(fft(F))))
F = F / np.max(np.abs(F)) * np.sqrt(F2)

# -- Derive physical quantities from the pump --
pump_power = 5e-3  # watts
dwo = np.sqrt(pump_power / np.mean(np.abs(F)**2) * 8 * g0 / (hbar * wc))
Q = wc / dwo
PP_no_pulse = np.max(np.abs(F)**2) * dwo**2 * hbar * wc / (8 * g0)

print(f"\n=== Pump Parameters ===")
print(f"  pump_power = {pump_power*1e3:.1f} mW")
print(f"  dwo = {dwo:.4e} rad/s")
print(f"  Q = {Q:.0f}")
print(f"  PP_no_pulse = {PP_no_pulse:.5f} W")
print(f"  Photon lifetime = {1/dwo*1e9:.3f} ns")
print(f"  Linewidth = {dwo/(2*np.pi*1e6):.1f} MHz")

# -- Dispersion coefficients --
# Set up dispersive wave targets
DWb = 1120e-9
DWr = 2 * DWb
DWbk = (c / DWb - c / lc) / fr
DWrk = (c / DWr - c / lc) / fr
Nc = (DWbk + DWrk) / 2
Ns = DWbk - Nc

beta = np.zeros(7)
beta[0] = -7.3e-4          # 1st order (frequency offset)
beta[1] = -8e-5            # 2nd order (anomalous GVD)
beta[3] = 12 * beta[1] / (Nc**2 - Ns**2)   # 4th order
beta[2] = -1 * beta[3] * Nc / 2             # 3rd order
beta[4] = 1e-13            # 5th order
beta[5] = 6e-16            # 6th order
beta[6] = -2.2e-18         # 7th order

print(f"\n=== Dispersion Coefficients ===")
for j, b in enumerate(beta):
    print(f"  beta[{j+1}] = {b:.4e}")

print(f"\n  Modal dispersion: {-beta[1]*dwo/(2*2*np.pi):.1f} Hz/mode")
print(f"  Mod freq diff: {-dwo*beta[0]/(2*2*np.pi):.1f} Hz")

# -- Apply additional phase modulation to pump (as in original code line 251) --
F = F * np.exp(1j * np.cos(theta) * 10 * np.pi)

# -- Quantum noise floor --
min_field = np.sqrt(2 * g0 / dwo * 0.5)  # half a photon per mode

# ══════════════════════════════════════════════════════════════════════
# Run the LLE simulation
# ══════════════════════════════════════════════════════════════════════

tphot_end = 200  # photon lifetimes (matching Octave run)

print(f"\n{'='*60}")
print(f"Running LLE simulation for {tphot_end} photon lifetimes...")
print(f"{'='*60}\n")

result = LLE(
    Ntheta=Ntheta,
    alpha=alpha,
    beta=beta,
    F=F,
    tphot_end=tphot_end,
    nsaves=200,
    min_field=min_field,
    h_init=5e-3,
    tol=5e-6,
    print_status=True,
)

# ══════════════════════════════════════════════════════════════════════
# Results summary
# ══════════════════════════════════════════════════════════════════════

I_final = result.intensity[-1]
Aft_final = result.A_k[-1]
Tfs = theta / np.pi * tRT * 1e15  # time axis in femtoseconds

print(f"\n=== Python Results ===")
print(f"  max(I) = {np.max(I_final):.6f}")
print(f"  mean(I) = {np.mean(I_final):.6f}")

# FWHM
above_half = I_final > np.max(I_final) / 2
if np.any(above_half):
    indices = np.where(above_half)[0]
    FWHM = Tfs[indices[-1]] - Tfs[indices[0]]
    print(f"  FWHM = {FWHM:.1f} fs")
else:
    FWHM = None
    print(f"  No clear pulse for FWHM measurement")

# ══════════════════════════════════════════════════════════════════════
# Benchmark against Octave results
# ══════════════════════════════════════════════════════════════════════

try:
    octave = scipy.io.loadmat('soliton_pulsed_results.mat')
    I_octave = octave['I'].squeeze()
    Aft_octave = octave['Aft'].squeeze()
    Tfs_octave = octave['Tfs'].squeeze()
    theta_octave = octave['theta'].squeeze()

    print(f"\n=== Octave Reference ===")
    print(f"  max(I) = {np.max(I_octave):.6f}")
    print(f"  mean(I) = {np.mean(I_octave):.6f}")

    # Octave FWHM
    above_half_oct = I_octave > np.max(I_octave) / 2
    if np.any(above_half_oct):
        idx_oct = np.where(above_half_oct)[0]
        FWHM_oct = Tfs_octave[idx_oct[-1]] - Tfs_octave[idx_oct[0]]
        print(f"  FWHM = {FWHM_oct:.1f} fs")
    else:
        FWHM_oct = None

    print(f"\n=== Comparison ===")
    print(f"  max(I) ratio (Python/Octave): {np.max(I_final)/np.max(I_octave):.3f}")
    print(f"  mean(I) ratio: {np.mean(I_final)/np.mean(I_octave):.3f}")
    if FWHM is not None and FWHM_oct is not None:
        print(f"  FWHM ratio: {FWHM/FWHM_oct:.3f}")

    has_octave = True
except FileNotFoundError:
    print("\n  (No Octave reference file found — skipping comparison)")
    has_octave = False

# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════

plt.rcParams.update({'font.size': 13, 'axes.linewidth': 1.5, 'lines.linewidth': 1.5})
wl_nm = kwave * 1e9
sort_idx = np.argsort(wl_nm)
wl_lim = [1050, 2450]
wl_tick = np.arange(1050, 2451, 400)

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# --- Panel 1: Intensity vs theta ---
ax = axes[0]
ax.plot(theta / np.pi, I_final, 'b-', label='Python')
if has_octave:
    ax.plot(theta_octave / np.pi, I_octave, 'r--', alpha=0.7, label='Octave')
ax.plot(theta / np.pi, np.abs(F)**2 / np.max(np.abs(F)**2) * np.max(I_final),
        'g-', alpha=0.4, linewidth=1, label='Pump (scaled)')
ax.set_xlim(-1, 1)
ax.set_ylim(0, max(np.max(I_final) * 1.2, 3))
ax.set_xlabel(r'$\theta / \pi$')
ax.set_ylabel('I (arb)')
ax.set_title(f'Soliton state at t = {tphot_end} photon lifetimes')
ax.legend(loc='upper right')

# --- Panel 2: Spectrum ---
ax = axes[1]
spec_py = 10 * np.log10(np.abs(Aft_final)**2 + 1e-300)
spec_py -= np.max(spec_py)
ax.plot(wl_nm[sort_idx], spec_py[sort_idx], 'b-', linewidth=0.8, label='Python')
if has_octave:
    spec_oct = 10 * np.log10(np.abs(Aft_octave)**2 + 1e-300)
    spec_oct -= np.max(spec_oct)
    ax.plot(wl_nm[sort_idx], spec_oct[sort_idx], 'r--', alpha=0.7, linewidth=0.8,
            label='Octave')
ax.set_xlim(wl_lim)
ax.set_xticks(wl_tick)
ax.set_ylim(-100, 0)
ax.set_xlabel(r'$\lambda$ (nm)')
ax.set_ylabel('Power (dB)')
ax.legend(loc='upper right')

# --- Panel 3: Temporal profile ---
ax = axes[2]
ax.plot(Tfs, I_final, 'b-', label='Python')
if has_octave:
    ax.plot(Tfs_octave, I_octave, 'r--', alpha=0.7, label='Octave')
ax.set_xlim(-1500, 1500)
ax.set_ylim(0, max(np.max(I_final) * 1.2, 3))
ax.set_xlabel('t (fs)')
ax.set_ylabel('I (arb)')
if FWHM is not None:
    ax.text(0.95, 0.85, f'FWHM = {FWHM:.0f} fs', transform=ax.transAxes,
            ha='right', fontsize=13, color='b')
if has_octave and FWHM_oct is not None:
    ax.text(0.95, 0.75, f'Octave FWHM = {FWHM_oct:.0f} fs', transform=ax.transAxes,
            ha='right', fontsize=13, color='r')
ax.legend(loc='upper left')

fig.tight_layout()
fig.savefig('python_vs_octave_comparison.png', dpi=200)
print(f"\nSaved python_vs_octave_comparison.png")

plt.show()
