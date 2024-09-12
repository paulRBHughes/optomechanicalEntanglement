import numpy as np
import matplotlib.pyplot as plt
import coolingutils

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })


zeta = 0.99
gamma = 3
nb1 = 0
nb2 = 5
initial = np.array([nb1, nb2, 0, 0])
target = 1e-8
tf = 250
# thetass = 0.5 * np.arctan(-2 * gamma)
# steady = (nb1 + nb2) * 0.5 - zeta * (nb2 - nb1) * 0.5
detunings = [0, 2, 10]



fig, ax = plt.subplots(2)
for detuning in detunings:
    t, state = coolingutils.detuned_simulation(zeta, gamma, nb1, nb2, detuning, initial, target, tf)
    ax[0].plot(t, ((state[2] + np.pi) % (2 * np.pi)) - np.pi, linewidth=2)
    ax[0].set_xlabel(r"$\tilde{t}$")
    ax[0].set_ylabel(r"$\theta$")

    # ax[1].plot(t, state[3], linewidth=2, label=zeta)
    # ax[1].set_xlabel(r"$\tilde{t}$")
    # ax[1].set_ylabel(r"$\sigma + \delta_p t$")
    #
    # ax[2].plot(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=2)
    # ax[2].set_xlabel(r"$\tilde{t}$")
    # ax[2].set_ylabel(r"$n_{c}$")

    ax[1].loglog(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2)
    ax[1].set_xlabel(r"$\tilde{t}$")
    ax[1].set_ylabel(r"$n_m$")


plt.tight_layout()
# plt.savefig("mechanicalCooling.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
