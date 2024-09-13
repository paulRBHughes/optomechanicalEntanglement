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
# fullfig, fax = plt.subplots(3)
for zeta in zets:
    ic = np.array([-0.5 * zeta * 75/(76), -1, 0])
    for i, g0 in enumerate(g0s):
        t, state = utils.rel_simulation(zeta, g0, av_bath, diff_bath, ic, target, tfs[i])
        corr = utils.rel_corr_var(state)
        # now I want to turn the pump off when I've minimized the correlation variance
        mindex = np.argmin(corr)
        off_state = state[:, mindex]
        t1 = t[mindex]
        t2, state2 = utils.rel_simulation(zeta, 0, av_bath, diff_bath, off_state, target, 1000 - t1)
        corr2 = utils.rel_corr_var(state2)
        t0 = t[:mindex]
        t2_scale = t1*np.ones(np.size(t2)) + t2
        fullt = np.concatenate((t0, t2_scale))
        fullcorr = np.concatenate((corr[:mindex], corr2))
        # av = np.concatenate((state[0, :mindex], state2[0, :]))
        # diff = np.concatenate((state[1, :mindex], state2[1, :]))
        # u = np.concatenate((state[2, :mindex], state2[2, :]))
        # print(np.size(t))

        ax.plot(np.log(fullt[1:]), np.log(fullcorr[1:] * 76), linewidth=2, label=zeta, linestyle=stys[i])
        # fax[0].plot(np.log(fullt[1:]), np.log(av[1:] + 1), linewidth=2, label=zeta)
        # fax[1].plot(np.log(fullt[1:]), np.log(diff[1:] + 1), linewidth=2, label=zeta)
        # fax[2].plot(np.log(fullt[1:]), u[1:], linewidth=2, label=zeta)

    # ax[0].plot(t, np.log(corr), linewidth=2)
    # ax[0].set_xlabel(r"$\tilde{t}$")
    # ax[0].set_ylabel("$\log(\Delta_{12}^2)$")
    # ax[0].hlines(0 - np.log(76), 0, tf, linewidth=1.5, linestyles=":")
    # ax[1].plot(t, state[1], linewidth=2, label=zeta)
    # ax[1].set_xlabel(r"$\tilde{t}$")
    # ax[1].set_ylabel(r"$\Delta n_{th}^0$")
ax.set_xlabel(r"$\ln\tilde{t}$")
ax.set_ylabel(r"$\ln(\Delta_{12}^2)$")
ax.hlines(0 - np.log(1), np.log(t[1]), np.log(t2[-1]), linewidth=1.5, linestyles=":", label="entanglement")
ax.legend(title="$\zeta$")
plt.tight_layout()
plt.show()
# plt.savefig("corrVar.pdf", format='pdf', dpi=1200, bbox_inches='tight')

