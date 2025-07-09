import numpy as np
import matplotlib.pyplot as plt
import coolingutils

"""
Purpose: to investigate the role of detuning from the sideband on optomechanical cooling
This code produces figure 1 in the manuscript
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

savefigs = False
path = "plots/detuningEffects.pdf"

# Set parameters
zeta = 0.99
gr = 2
nb1 = 0
nb2 = 40
initial = np.array([nb1, nb2, 0, 0])  # This is the IC of a thermal state at equilibrium
target = 1e-9
tf = 200
detunings = np.flip([0, 1, 5, 20])  # These are \Delta_+ in the paper, code will plot these
syts = np.flip(['-', '--', '-.', ':'])  # Linestyles for plotting
colors = ['lightcoral', 'firebrick', 'darkred', 'red']  # colours for plotting


fig, ax = plt.subplots(2)
for i, detuning in enumerate(detunings):
    # run the set of equations that include the phase and a detuning contribution
    t, state = coolingutils.detuned_simulation(zeta, gr, nb1, nb2, detuning, initial, target, tf)
    ax[0].plot(t, state[2]/np.pi, linewidth=2, linestyle=syts[i], color=colors[i])  # Plotting theta
    mech_pop = state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2]))
    ax[1].loglog(t, mech_pop, linewidth=2, color=colors[i], linestyle=syts[i], label=detuning)  # and the TOTAL (not thermal) population of mechanical mode
ax[0].set_xlabel(r"$\tilde{t}$")
ax[0].set_ylabel(r"$\theta/\pi$")
ax[0].set_xlim([0, 10])
ax[0].text(2, -0.1, "(a)", fontfamily="Computer Modern Roman", fontsize="large", horizontalalignment='center',
        verticalalignment='center')
ax[1].text(1, 2, "(b)", fontfamily="Computer Modern Roman", fontsize="large", horizontalalignment='center',
        verticalalignment='center')
ax[1].set_xlim([0.1, ax[1].get_xlim()[1]])
ax[1].set_xlabel(r"$\tilde{t}$")
ax[1].set_ylabel(r"$n_{m}$")
ax[1].legend(title=r"$\Delta_+/\Gamma_+$", fontsize=14)

ax[0].tick_params(axis='y', direction='in', top=True, right=True, which='both')
ax[1].tick_params(axis='y', direction='in', top=True, right=True, which='both')
ax[0].tick_params(axis='x', direction='in', top=True, left=True, which='major')
ax[1].tick_params(axis='x', direction='in', top=True, right=True, which='major')


plt.tight_layout()
if savefigs:
    plt.savefig(path, format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
