import numpy as np
import matplotlib.pyplot as plt
import utils

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

# zeta = 0.3
g0 = 10
av_bath = 0
diff_bath = 2 * av_bath
# dbaths = [0, 0.1, 0.2, 0.3]
zets = [0.8, 0.9, 0.99, 0.9999]
target = 1e-11
tf = 1

fig, ax = plt.subplots(2)
for zeta in zets:
    ic = np.array([-0.5 * zeta * 75/(76), -1, 0])
    t, state = utils.rel_simulation(zeta, g0, av_bath, diff_bath, ic, target, tf)
    # print(np.size(t))
    corr = utils.rel_corr_var(state)
    # ax.plot(t, np.log(corr), linewidth=2, label=zeta)
    # ax.set_xlabel(r"$\tilde{t}$")
    # ax.set_ylabel(r"$\log(\tilde{\Delta}_{12}^2)$")
    ax[0].plot(t, np.log(corr), linewidth=2)
    ax[0].set_xlabel(r"$\tilde{t}$")
    ax[0].set_ylabel("$\log(\Delta_{12}^2)$")
    ax[0].hlines(0 - np.log(76), 0, tf, linewidth=1.5, linestyles=":")
    ax[1].plot(t, state[1], linewidth=2, label=zeta)
    ax[1].set_xlabel(r"$\tilde{t}$")
    ax[1].set_ylabel(r"$\Delta n_{th}^0$")
ax[0].hlines(0 - np.log(76), 0, tf, linewidth=1.5, linestyles=":", label="entanglement")
ax[1].legend(title="$\zeta$")
plt.tight_layout()
plt.show()
