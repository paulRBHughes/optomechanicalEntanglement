import numpy as np
import matplotlib.pyplot as plt
import utils

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

# zeta = 0.3
g0 = 0.2
av_bath = 0.5
diff_bath = 1
# dbaths = [0, 0.1, 0.2, 0.3]
zets = [-1, -0.5, 0, 0.5, 1]
target = 1e-11
tf = 5

fig, ax = plt.subplots(2)
for zeta in zets:
    ic = np.array([0.01, 0.01, 0])
    t, state = utils.rel_simulation(zeta, g0, av_bath, diff_bath, ic, target, tf)
    # print(np.size(t))
    corr = utils.rel_corr_var(state)
    ax[0].plot(t, corr, linewidth=2)
    ax[0].set_xlabel(r"$\tilde{t}$")
    ax[0].set_ylabel("$\Delta_{12}^2$")
    ax[0].hlines(1, 0, tf, linewidth=1.5, linestyles=":")
    ax[1].plot(t, state[1] - diff_bath, linewidth=2, label=zeta)
    ax[1].set_xlabel(r"$\tilde{t}$")
    ax[1].set_ylabel(r"$\Delta n - \Delta n_b$")
ax[1].legend(title="$\zeta$")
plt.tight_layout()
plt.show()
