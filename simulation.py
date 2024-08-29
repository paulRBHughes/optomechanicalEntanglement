import numpy as np
import matplotlib.pyplot as plt
import coolingutils

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })


zeta = 0.6
gamma = 5
nb1 = 0.
nb2 = 1
initial = np.array([nb1, nb2, 0])
target = 1e-8
tf = 5
# thetass = 0.5 * np.arctan(-2 * gamma)
steady = (nb1 + nb2) * 0.5 - zeta * (nb2 - nb1) * 0.5


t, state = coolingutils.simulation(zeta, gamma, nb1, nb2, initial, target, tf)

fig, ax = plt.subplots(3)
ax[0].plot(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_c$")
ax[0].plot(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_m$")
ax[0].hlines(steady, 0, tf, linewidth=1.5, linestyles=":", label="$n_{ss}$")
ax[0].set_xlabel(r"$\tilde{t}$")
ax[0].set_ylabel("$n$")
ax[0].legend()


ax[1].plot(t, state[0], linewidth=2, label=r"$n_c^{th}$")
ax[1].plot(t, state[1], linewidth=2, label=r"$n_m^{th}$")
ax[1].hlines(steady, 0, tf, linewidth=1.5, linestyles=":", label="$n^{th}_{ss}$")
ax[1].set_xlabel(r"$\tilde{t}$")
ax[1].set_ylabel(r"$n_{th}$")
ax[1].legend()

ax[2].plot(t, state[2] % (2 * np.pi), linewidth=2, label=zeta)
ax[2].set_xlabel(r"$\tilde{t}$")
ax[2].set_ylabel(r"$\theta$")

plt.tight_layout()
# plt.savefig("mechanicalCooling.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
