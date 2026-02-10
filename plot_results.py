"""
Plot results from the Octave LLE soliton simulation.

Loads soliton_pulsed_results.mat and generates three figures matching
the original MATLAB script's output.

Usage:
    python plot_results.py
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load results
data = scipy.io.loadmat('soliton_pulsed_results.mat')

# Extract variables (squeeze to remove extra dimensions from MATLAB)
A = data['A'].squeeze()
Aft = data['Aft'].squeeze()
I = data['I'].squeeze()
theta = data['theta'].squeeze()
k = data['k'].squeeze()
kwave = data['kwave'].squeeze()
kfreq = data['kfreq'].squeeze()
F = data['F'].squeeze()
Fch = data['Fch'].squeeze()
tRT = data['tRT'].item()
Tfs = data['Tfs'].squeeze()
beta = data['beta'].squeeze()
alpha = data['alpha'].item()
F2 = data['F2'].item()
dwo = data['dwo'].item()
Q = data['Q'].item()
pump_power = data['pump_power'].item()
g0 = data['g0'].item()
tphot = data['tphot'].item()
iter_count = data['iter'].item()
wl_lim = data['wl_lim'].squeeze()
wl_tick = data['wl_tick'].squeeze()
D2 = data['D2'].squeeze()
Dgvd = data['Dgvd'].squeeze()
dalpha = data['dalpha'].squeeze()

print(f"Loaded results: {iter_count} iterations, tphot = {tphot:.1f}")
print(f"  max(I) = {np.max(I):.4f}, mean(I) = {np.mean(I):.6f}")
print(f"  Q = {Q:.0f}, dwo = {dwo:.4e}")
print(f"  pump_power = {pump_power*1e3:.1f} mW")

# Common settings
plt.rcParams.update({
    'font.size': 14,
    'axes.linewidth': 2,
    'lines.linewidth': 2,
})

# ── Figure 1: Dispersion and pump spectrum ──────────────────────────
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8))

# Top: D2 dispersion
wl_nm = kwave * 1e9
sort_idx = np.argsort(wl_nm)
ax1a.plot(wl_nm[sort_idx], D2[sort_idx], 'b-')
ax1a.set_xlim(wl_lim)
ax1a.set_xticks(wl_tick)
ax1a.set_ylim(-320, 200)
ax1a.set_xlabel(r'$\lambda$ (nm)')
ax1a.set_ylabel(r'$D_2$ (kHz/mode)')

# Overlay Dgvd on twin axis
ax1a_r = ax1a.twinx()
ax1a_r.plot(wl_nm[sort_idx], Dgvd[sort_idx], 'r-')
ax1a_r.set_ylim(-640, 400)
ax1a_r.set_ylabel(r'$D_\mathrm{GVD}$ (ps/nm·km)', color='r')
ax1a_r.tick_params(axis='y', labelcolor='r')

# Bottom: Input pump spectrum
specF = np.abs(np.fft.fft(F) / np.max(np.abs(np.fft.fft(F))))**2
specF_dB = 10 * np.log10(specF + 1e-300)
ax1b.plot(wl_nm[sort_idx], specF_dB[sort_idx])
ax1b.set_xlim(1525, 1575)
ax1b.set_ylim(-60, 5)
ax1b.set_xlabel(r'$\lambda$ (nm)')
ax1b.set_ylabel('P (dB)')

fig1.tight_layout()
fig1.savefig('octave_input_and_dispersion.png', dpi=200)
print("Saved octave_input_and_dispersion.png")

# ── Figure 2: Final soliton output (3 panels) ──────────────────────
fig2, (ax2a, ax2b, ax2c) = plt.subplots(3, 1, figsize=(10, 12))

# Panel 1: Intensity vs theta/pi
ax2a.plot(theta / np.pi, I, 'b-', label='Output')
ax2a.plot(theta / np.pi, np.abs(F)**2 / np.max(np.abs(F)**2) * np.max(I),
          'r-', alpha=0.7, label='Pump (scaled)')
ax2a.set_xlim(-1, 1)
ax2a.set_ylim(0, max(np.max(I) * 1.1, 3))
ax2a.set_xlabel(r'$\theta / \pi$')
ax2a.set_ylabel('I (arb)')
ax2a.set_title(f'Final state: iter {iter_count}, '
               rf'$t_{{\mathrm{{phot}}}}$ = {tphot:.1f}')
ax2a.legend(loc='upper right')

# Panel 2: Spectrum
spec = 10 * np.log10(np.abs(Aft)**2 + 1e-300)
spec = spec - np.max(spec)
ax2b.plot(wl_nm[sort_idx], spec[sort_idx], 'b-', linewidth=1)
ax2b.set_xlim(wl_lim)
ax2b.set_xticks(wl_tick)
ax2b.set_ylim(-100, 0)
ax2b.set_xlabel(r'$\lambda$ (nm)')
ax2b.set_ylabel('Power (dB)')

# Panel 3: Intensity vs time (fs)
ax2c.plot(Tfs, I, 'b-', label='Output')
ax2c.plot(Tfs, np.abs(F)**2 / np.max(np.abs(F)**2) * np.max(I),
          'r-', alpha=0.7, label='Pump (scaled)')
ax2c.set_xlim(-1500, 1500)
ax2c.set_ylim(0, max(np.max(I) * 1.1, 3))
ax2c.set_xlabel('t (fs)')
ax2c.set_ylabel('I (arb)')
ax2c.legend(loc='upper right')

# FWHM measurement
above_half = I > np.max(I) / 2
if np.any(above_half):
    indices = np.where(above_half)[0]
    t_first = Tfs[indices[0]]
    t_last = Tfs[indices[-1]]
    FWHM = t_last - t_first
    ax2c.text(500, np.max(I) * 0.8, f'FWHM {FWHM:.0f} fs', fontsize=14)
    print(f"Pulse FWHM = {FWHM:.1f} fs")
else:
    print("No clear pulse found for FWHM measurement")

fig2.tight_layout()
fig2.savefig('octave_final_output.png', dpi=200)
print("Saved octave_final_output.png")

# ── Figure 3: Delta-alpha ──────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.fill_between(wl_nm[sort_idx], dalpha.imag[sort_idx], alpha=0.6)
ax3.plot(wl_nm[sort_idx], dalpha.imag[sort_idx], linewidth=1)
ax3.set_xlim(wl_lim)
ax3.set_ylim(-100, 100)
ax3.set_xticks(wl_tick)
ax3.set_xlabel(r'$\lambda$ (nm)')
ax3.set_ylabel(r'$\Delta\alpha(\lambda)$')

fig3.tight_layout()
fig3.savefig('octave_dalpha.png', dpi=200)
print("Saved octave_dalpha.png")

plt.show()
print("\nAll done!")
