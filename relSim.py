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
av_bath = 0.2
diff_bath = 0.1
ic = np.array([av_bath, diff_bath, 0])
target = 1e-8
tf = 10

fig, ax = plt.subplots(2)
t, state = utils.rel_simulation(zeta, g0, av_bath, diff_bath, ic, target, tf)
print(np.size(t))
corr = utils.rel_corr_var(state)
ax[0].plot(t, corr, linewidth=2)
ax[0].set_xlabel(r"$\tilde{t}$")
ax[0].set_ylabel("$\Delta_{12}^2$")
ax[0].hlines(1, 0, tf, linewidth=1.5, linestyles=":")
ax[1].plot(t, state[0], linewidth=2)
ax[1].set_xlabel(r"$\tilde{t}$")
ax[1].set_ylabel(r"$\bar{n}$")
plt.tight_layout()
plt.show()
