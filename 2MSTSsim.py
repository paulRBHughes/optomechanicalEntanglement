import numpy as np
import matplotlib.pyplot as plt
import utils

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

zeta = 0.6
g0 = 0.75
n1b = 0.15
n2b = 0.25
ic = np.array([n1b, n2b, 0])
target = 1e-12
tf = 10

fig, ax = plt.subplots(2)
t, state = utils.simulation(zeta, g0, n1b, n2b, ic, target, tf)
corr = utils.correlation_var(state)
ax[0].plot(t, corr, linewidth=2)
ax[0].set_xlabel(r"$\tilde{t}$")
ax[0].set_ylabel("$\Delta_{12}^2$")
ax[0].hlines(1, 0, tf, linewidth=1.5, linestyles=":")
ax[1].plot(t, 0.5 * (state[1] + state[0]), linewidth=2)
ax[1].set_xlabel(r"$\tilde{t}$")
ax[1].set_ylabel(r"$\bar{n}$")
plt.tight_layout()
plt.show()