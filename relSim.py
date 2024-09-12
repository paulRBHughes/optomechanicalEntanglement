import numpy as np
import matplotlib.pyplot as plt
import utils

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

# zeta = 0.3
g0s = [0.5, 3]
stys = ["-", "--"]
av_bath = 0
diff_bath = 2 * av_bath
# dbaths = [0, 0.1, 0.2, 0.3]
zets = [0.99, 0.9999]
target = 1e-11
tfs = [10, 3]

fig, ax = plt.subplots()
for zeta in zets:
    ic = np.array([-0.5 * zeta * 75/(76), -1, 0])
    for i, g0 in enumerate(g0s):
        t, state = utils.rel_simulation(zeta, g0, av_bath, diff_bath, ic, target, tfs[i])
        # print(np.size(t))
        corr = utils.rel_corr_var(state)
        ax.plot(np.log(t[1:]), np.log(corr[1:] * 76), linewidth=2, label=zeta, linestyle=stys[i])
    # ax[0].plot(t, np.log(corr), linewidth=2)
    # ax[0].set_xlabel(r"$\tilde{t}$")
    # ax[0].set_ylabel("$\log(\Delta_{12}^2)$")
    # ax[0].hlines(0 - np.log(76), 0, tf, linewidth=1.5, linestyles=":")
    # ax[1].plot(t, state[1], linewidth=2, label=zeta)
    # ax[1].set_xlabel(r"$\tilde{t}$")
    # ax[1].set_ylabel(r"$\Delta n_{th}^0$")
ax.set_xlabel(r"$\ln\tilde{t}$")
ax.set_ylabel(r"$\ln(\Delta_{12}^2)$")
ax.hlines(0 - np.log(1), np.log(t[1]), np.log(tfs[0]), linewidth=1.5, linestyles=":", label="entanglement")
ax.legend(title="$\zeta$")
plt.tight_layout()
# plt.show()
plt.savefig("corrVar.pdf", format='pdf', dpi=1200, bbox_inches='tight')

