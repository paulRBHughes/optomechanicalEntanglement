import numpy as np
import matplotlib.pyplot as plt
import coolingutils

"""
This is a cooling simulation - looking at the theta factor and populations
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })


zeta = 0.999
# g = 0.5
nb1 = 0.
nb2 = 5.
initial = np.array([nb1, nb2, 0])
target = 1e-9
tf = 100
# steadylim = (nb1 + nb2) * 0.5 - zeta * (nb2 - nb1) * 0.5
# B = kappa/Omega_m. B -> 0 is SBR
B = 0.1

# print(state[:, -1])
# fig, ax = plt.subplots()
# ax.plot(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_c$")
# ax.plot(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2, label=r"$n_m$")
# ax.hlines(steady, 0, tf, linewidth=1.5, linestyles=":", label="$n_{ss}$")
# ax.set_xlabel(r"$\tilde{t}$")
# ax.set_ylabel("$n$")
# ax.legend()
gees = [0.5, 1, 2]
stysy = [':', '-', '--']
fig, ax = plt.subplots(2)
for i, g in enumerate(gees):
    t, state = coolingutils.simulation(zeta, g, B, nb1, nb2, initial, target, tf)
    ax[0].semilogy(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=2, label=g, color='b', linestyle=stysy[i])
    ax[0].semilogy(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2, color='r', linestyle=stysy[i])
    # ax[0].hlines(steadylim, 0, np.log(tf), linewidth=1.5, linestyles=":")
    ax[0].set_xlabel(r"$\tilde{t}$")
    ax[0].set_ylabel("$n$")
    ax[0].legend()
    print(f'{g}, {state[2][-1]}')

    #
    # ax[1].plot(t, state[0], linewidth=2, label=r"$n_c^{th}$")
    # ax[1].plot(t, state[1], linewidth=2, label=r"$n_m^{th}$")
    # # ax[1].hlines(steadylim, 0, tf, linewidth=1.5, linestyles=":")
    # ax[1].set_xlabel(r"$\tilde{t}$")
    # ax[1].set_ylabel(r"$n_{th}$")
    # ax[1].legend()
    recet = ((state[2]/np.pi + 1) % 2) - 1
    ax[1].plot(t, state[2]/np.pi, linewidth=2, label=gees[i])
    ax[1].set_xlabel(r"$\tilde{t}$")
    ax[1].set_ylabel(r"$\theta/\pi$")

ax[1].legend()

plt.tight_layout()
# plt.savefig("mechanicalCooling.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
