graphics_toolkit("gnuplot");
clear;
close all;

c = 2.9979e8;
lc = 1550 * 1e-9;
wl_lim = [1050 2450];
wl_tick = 1050 : 400 : 2450;

tphot_end = 200;
Nt = 2e5;
h_phot = 5e-3;
h = h_phot / 2;
tol = 5e-6;

no = 2;
fr = 33e9;
mode_area = (1e-6) ^ 2;
n2 = 24e-20;
hbar = 1e-34;

tRT = 1 / fr;
diameter = tRT * (c / no) / pi;
r = diameter / 2;
fc = c / lc;
wc = 2 * pi * fc;

circumference = tRT * (c / no);
g0 = n2 * c * hbar * wc.^ 2 / (no ^ 2 * circumference * mode_area);
disp(['g0 = ' num2str(g0)]);

Ntheta = 2 ^ nextpow2(16000);
dtheta = 2 * pi / Ntheta;
theta = (-Ntheta / 2 : Ntheta / 2 - 1) * dtheta;
k = [0:Ntheta / 2 - 1 - Ntheta / 2:-1];
kfreq = fc + fr * k;
kwave = c./ kfreq;

alpha = 2.5;
F2 = 3.30;
disp(['F2 = ' num2str(F2)]);

dPM = 7.5 * pi;
F = cos(pi / 2 * sin(theta / 2).^ 2).*exp(1i * dPM * cos(theta));
Fch = F;
F = fftshift(ifft(abs(fft(F))));
F = F / max(abs(F)) * sqrt(F2);

pump_power = 5e-3;
disp(['pump_power = ' num2str(pump_power)]);

dwo = sqrt(pump_power / mean(abs(F).^ 2) * 8 * g0 / (hbar * wc));
disp(['dwo = ' num2str(dwo)]);
Q = wc / dwo;
disp(['Q = ' num2str(Q)]);
PP_no_pulse = max(abs(F).^ 2) * dwo ^ 2 * hbar * wc / (8 * g0);
disp(['PP_no_pulse = ' num2str(PP_no_pulse)]);

DWb = 1120 * 1e-9;
DWr = 2 * DWb;
DWbk = (c./ (DWb)-c./ lc) / fr;
DWrk = (c./ (DWr)-c./ lc) / fr;
Nc = (DWbk + DWrk) / 2;
Ns = DWbk - Nc;

beta(1) = -7.3e-4;
beta(2) = -8e-5;
beta(4) = 12 * beta(2) / (Nc ^ 2 - Ns ^ 2);
beta(3) = -1 * beta(4) * Nc / 2;
beta(5) = 1e-13;
beta(6) = 6e-16;
beta(7) = -2.2e-18;

dalpha = 0;
for
  j = 1 : length(beta) dalpha = dalpha + 1i * beta(j) / factorial(j) * k.^ j;
end

    disp(['modal dispersion: ' num2str(-beta(2) * dwo /
                                       (2 * 2 * pi)) ' Hz/mode']);
mod_freq_diff = -dwo * beta(1) / (2 * 2 * pi);
disp(['mod_freq_diff = ' num2str(mod_freq_diff)]);
disp(['photon lifetime is ' num2str(1 / dwo * 1e9) ' nanoseconds']);
disp(['resonance linewidth is ' num2str(dwo / (2 * pi * 1e6)) ' MHz']);

D_disp = -beta * dwo / (4 * pi);
D2 = D_disp(2 : end);
temp = 0;
for
  jj = 1 : length(D2) temp =
               temp + D2(jj) ^ jj.*k'.^(jj-1)/factorial(jj-1); end D2 =
                   temp * 1e-3;

Dgvd = D2 * 1e3 *no / c * 1 / fr ^ 2 * 2 *pi *c./ kwave'.^2*1e12/(1e-3*1e9);

                                       F = F.*exp(1i * cos(theta) * 10 * pi);

D = -(1 + 1i * alpha);
for
  j = 1 : length(beta) D = D + 1i * beta(j) / factorial(j) * k.^ j;
end

    A = zeros(size(theta));
A_in = A;

dtphot_plot = 1;
iter = 0;
piter = 1;
I = abs(A).^ 2;
Tfs = theta'/pi*tRT*1e15; min_field = sqrt(2 * g0 / dwo * 1 / 2);

tphot = 0;
tphot_plot_next = dtphot_plot;
pump_adj = 0;
Do = D;

disp('Starting LLE iteration...');
disp(['Target: tphot_end = ' num2str(tphot_end)]);
fflush(stdout);

tic;

while
  tphot < tphot_end iter = iter + 1;
error_val = 1;
h = 2 * h;
Aft = fft(A);

while
  error_val > 2 *tol AI = ifft(exp(D * h / 2).*Aft);
k1 = ifft(exp(D * h / 2).*fft(h * (1i * abs(A).^ 2. * A + F)));

Ak2 = AI + k1 / 2;
k2 = h * (1i * abs(Ak2).^ 2. * Ak2 + F);

Ak3 = AI + k2 / 2;
k3 = h * (1i * abs(Ak3).^ 2. * Ak3 + F);

Ak4 = ifft(exp(D * h / 2).*fft(AI + k3));
k4 = h * (1i * abs(Ak4).^ 2. * Ak4 + F);

A1 = ifft(exp(D * h / 2).*fft(AI + k1 / 6 + k2 / 3 + k3 / 3)) + k4 / 6;

h = h / 2;
A2 = A;
        for
          jstep = 1 : 2 AI = ifft(exp(D * h / 2).*Aft);
        k1 = ifft(exp(D * h / 2).*fft(h * (1i * abs(A2).^ 2. * A2 + F)));

        Ak2 = AI + k1 / 2;
        k2 = h * (1i * abs(Ak2).^ 2. * Ak2 + F);

        Ak3 = AI + k2 / 2;
        k3 = h * (1i * abs(Ak3).^ 2. * Ak3 + F);

        Ak4 = ifft(exp(D * h / 2).*fft(AI + k3));
        k4 = h * (1i * abs(Ak4).^ 2. * Ak4 + F);

        A2 = ifft(exp(D * h / 2).*fft(AI + k1 / 6 + k2 / 3 + k3 / 3)) + k4 / 6;
        if jstep
          > 1 break;
        end Aft = fft(A2);
        end

            error_val = sqrt(sum(abs(A2 - A1).^ 2) / sum(abs(A2).^ 2));
        end

            h_phot = 4 * h;
        tphot = tphot + h_phot;
        Aft = fft(4 / 3 * A2 - 1 / 3 * A1);
        Aft = Aft + min_field * exp(1i * 2 * pi * rand(size(A))).*
                                    (abs(Aft) < min_field);
        A = ifft(Aft);
        I = abs(A).^ 2;
        h = 2 ^ (-1 / 3) * h * (error_val > tol) + 2 ^
            (1 / 3) * h * (error_val <= tol);

        if tphot
          > tphot_plot_next Alog(piter, :) = A;
        Aftlog(piter, :) = Aft;
        tphotlog(piter) = tphot;

        if mod (piter, 100)
          == 0 ||
              piter <=
                  5 fprintf(
                      '  tphot = %.1f / %d  (iter %d, max(I) = %.4f, error = %.2e)\n',
                      tphot, tphot_end, iter, max(I), error_val);
        fflush(stdout);
        end

            piter = piter + 1;
        tphot_plot_next = tphot_plot_next + dtphot_plot;
        end end

            elapsed = toc;
        fprintf('\nSimulation complete!\n');
        fprintf('Total iterations: %d\n', iter);
        fprintf('Elapsed time: %.1f seconds\n', elapsed);
        fprintf('Final max(I) = %.6f\n', max(I));
        fprintf('Final mean(I) = %.6f\n', mean(I));
        fflush(stdout);

        save('-v7', 'soliton_pulsed_results.mat', 'A', 'Aft', 'I', 'theta', 'k',
             'kwave', 'kfreq', 'F', 'Fch', 'tRT', 'Tfs', 'beta', 'alpha', 'F2',
             'dwo', 'Q', 'pump_power', 'g0', 'tphot', 'iter', 'Alog', 'Aftlog',
             'tphotlog', 'wl_lim', 'wl_tick', 'D2', 'Dgvd', 'dalpha');
        disp('Results saved to soliton_pulsed_results.mat');

        fprintf('\nAll done! Use plot_results.py to visualize.\n');
