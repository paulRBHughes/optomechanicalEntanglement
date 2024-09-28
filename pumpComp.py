import numpy as np
import matplotlib.pyplot as plt

import utils


"""
scheme.py
Purpose: to determine the correlation variance minimum and time below 1 for multiple cooling SS
        Here I will vary the relative loss rates for our defined pump strengths and populations
Date: 3 Sept 2024
Paul RB Hughes
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

g = 1
enh = 2
nbc = 0
nbm = 10000
scale = nbm/(nbm + 1)
target = 1e-8
tf = 10
zetas = [0.9999]
off = 1.5
wid = 0.5

# gees = np.arange(1, 5, 0.1)
# tfs = 7.5 * np.reciprocal(gees)


def pulse(amplitude, offset, width, t):
    g = amplitude * np.exp(-0.5 * np.square((t - offset) / width))# / np.square(width)
    return g


def square_pulse(amp, t_up, t_down, t):
    # starts up
    return amp * (np.heaviside(t - t_up, 1) - np.heaviside(t - t_down, 1))


fig, ax = plt.subplots(2)
times = np.arange(0, tf, 0.01)
field = pulse(g*enh, off, wid, times)
squat = square_pulse(g * enh, 0, 1.5, times)
ax[1].hlines(g, 0, tf, linewidth=1.5, linestyle="--")
ax[1].plot(times, field, linewidth=2, linestyle="-")
ax[1].plot(times, squat, linewidth=2, linestyle="-.")


for z in zetas:
    # ic = np.array([-0.5 * z * scale * nbm, -1, 0])  # assumes large bath for mech
    ic = np.array([-z * scale * 0.5, -scale, 0])
    t, state = utils.rel_simulation(z, g, 0, 0, ic, target, tf)
    corr = utils.rel_corr_var(state)
    ax[0].semilogy(t, corr * (nbm + 1), linewidth=2, linestyle="--")

    t, state = utils.pump_simulation(z, pulse, 0, 0, ic, target, tf, g*enh, off, wid)
    corr = utils.rel_corr_var(state)
    ax[0].semilogy(t, corr * (nbm + 1), linewidth=2, linestyle="-", label=z)

    t, state = utils.pump_simulation(z, square_pulse, 0, 0, ic, target, tf, g*enh, 0, 1.5)
    corr = utils.rel_corr_var(state)
    ax[0].semilogy(t, corr * (nbm + 1), linewidth=2, linestyle="-.")

    t, state = utils.rel_simulation(z, 0.1, 0, 0, ic, target, tf)
    corr = utils.rel_corr_var(state)
    ax[0].semilogy(t, corr * (nbm + 1), linewidth=2, linestyle=":")

ax[0].hlines(1, 0, tf, linewidth=1.5, linestyle=":")
ax[0].set_xlabel(r"$\tilde{t}$")
ax[0].set_ylabel(r"$\Delta_{12}^2$")
ax[1].set_xlabel(r"$\tilde{t}$")
ax[1].set_ylabel(r"$g(t)$")

ax[0].legend(title="$\zeta$")
plt.tight_layout()
plt.show()
# plt.savefig("entanglementStrength.pdf", format='pdf', dpi=1200, bbox_inches='tight')

