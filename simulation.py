import numpy as np
import matplotlib.pyplot as plt
import coolingutils

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })


zeta = 0.9999
gamma = 0
nb1 = 0.
nb2 = 50
initial = np.array([.00249999, .00250001, -np.pi*0.25])
target = 1e-12
tf = 1
# thetass = 0.5 * np.arctan(-2 * gamma)
steady = (nb1 + nb2) * 0.5 - zeta * (nb2 - nb1) * 0.5


t, state = coolingutils.simulation(zeta, gamma, nb1, nb2, initial, target, tf)
print(state[:, -1])
# fig, ax = plt.subplots()
# ax.plot(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_c$")
# ax.plot(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_m$")
# ax.hlines(steady, 0, tf, linewidth=1.5, linestyles=":", label="$n_{ss}$")
# ax.set_xlabel(r"$\tilde{t}$")
# ax.set_ylabel("$n$")
# ax.legend()

fig, ax = plt.subplots(2)
ax[0].semilogx(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_c$")
ax[0].semilogx(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_m$")
ax[0].hlines(steady, 0, np.log(tf), linewidth=1.5, linestyles=":", label="$n_{ss}$")
ax[0].set_xlabel(r"$\tilde{t}$")
ax[0].set_ylabel("$n$")
ax[0].legend()

#
# ax[1].plot(t, state[0], linewidth=2, label=r"$n_c^{th}$")
# ax[1].plot(t, state[1], linewidth=2, label=r"$n_m^{th}$")
# ax[1].hlines(steady, 0, tf, linewidth=1.5, linestyles=":", label="$n^{th}_{ss}$")
# ax[1].set_xlabel(r"$\tilde{t}$")
# ax[1].set_ylabel(r"$n_{th}$")
# ax[1].legend()

ax[1].semilogx(t, ((state[2] + np.pi) % (2 * np.pi)) - np.pi, linewidth=2, label=zeta)
ax[1].set_xlabel(r"$\tilde{t}$")
ax[1].set_ylabel(r"$\theta$")

plt.tight_layout()
# plt.savefig("mechanicalCooling.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
