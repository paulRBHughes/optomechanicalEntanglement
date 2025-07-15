import numpy as np
import matplotlib.pyplot as plt
import coolingutils

"""
Purpose: simulate and plot the beamsplitter Hamiltonian, producing Fig. 3 of the manuscript.
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

savefig = False
path = "plots/mechanicalCooling.pdf"

# Set parameters here
zeta = 0.8
gr = 3.5
nb1 = 0.
nb2 = 40
initial = np.array([nb1, nb2, 0])
target = 1e-12  # target accuracy of state variables
tf = 5  # final time
thetalim = 0.5 * np.arctan(-2 * gr)  # This is the steady-state limit of theta
steadylim = (nb1 + nb2) * 0.5 - zeta * (nb2 - nb1) * 0.5  # this is the steady-state limit of the thermal populations for a large pump


t, state = coolingutils.simulation(zeta, gr, nb1, nb2, initial, target, tf)
print(state[:, -1])  # I have it showing the final state, because it has been of interest
# fig, ax = plt.subplots()
# ax.plot(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_c$")
# ax.plot(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_m$")
# ax.hlines(steady, 0, tf, linewidth=1.5, linestyles=":", label="$n_{ss}$")
# ax.set_xlabel(r"$\tilde{t}$")
# ax.set_ylabel("$n$")
# ax.legend()

fig, ax = plt.subplots(3)
ax[0].plot(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_c$", color='b')
ax[0].plot(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_m$", color='r')
ax[0].hlines(steadylim, 0, np.log(tf), linewidth=1.5, linestyles=":")
ax[0].set_ylabel("$n$")
ax[0].legend()


ax[1].plot(t, state[0], linewidth=2, label=r"$n_c^{th}$", color='b')
ax[1].plot(t, state[1], linewidth=2, label=r"$n_m^{th}$", color='r')
ax[1].hlines(steadylim, 0, tf, linewidth=1.5, linestyles=":")
ax[1].set_ylabel(r"$n_{th}$")
ax[1].legend()
ax[2].plot(t, state[2]/np.pi, linewidth=2, label=zeta)
ax[2].set_xlabel(r"$\tilde{t}$")
ax[2].set_ylabel(r"$\theta/\pi$")

plt.tight_layout()
if savefig:
    plt.savefig(path, format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
