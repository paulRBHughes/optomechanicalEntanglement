import numpy as np
import matplotlib.pyplot as plt
import coolingutils

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

# T = 300
# omega1 = 300e12
# omega2 = 300e9
deltOmega = 0
zeta = 0.98
gamma = 0.1
nb1 = 0
nb2 = 2
initial = np.array([nb1, nb2, 0, np.pi/2])
target = 1e-8
tf = 5

t, state = coolingutils.simulation(deltOmega, zeta, gamma, nb1, nb2, initial, target, tf)

fig, ax = plt.subplots(2)
ax[0].plot(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=2)
ax[0].set_xlabel(r"$\tilde{t}$")
ax[0].set_ylabel("$n_{1}$")
# ax[0].hlines(1, 0, tf, linewidth=1.5, linestyles=":")
ax[1].plot(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2, label=zeta)
ax[1].set_xlabel(r"$\tilde{t}$")
ax[1].set_ylabel(r"$n_2$")
# ax[1, 0].plot(t, state[2]%np.pi, linewidth=2, label=zeta)
# ax[1, 0].set_xlabel(r"$\tilde{t}$")
# ax[1, 0].set_ylabel(r"$\theta$")
# ax[1, 1].plot(t, state[3]%np.pi, linewidth=2, label=zeta)
# ax[1, 1].set_xlabel(r"$\tilde{t}$")
# ax[1, 1].set_ylabel(r"$\phi$")
plt.tight_layout()
plt.show()

