import numpy as np
import matplotlib.pyplot as plt
import coolingutils
import utils
from multiprocessing import Pool

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

"""
optimalPump.py
Purpose: Determine the pump that creates the longest lasting level under a certain correlation variance.
Date: 23 Sept 2024
Paul RB Hughes
"""

# nbc = 0
nbm = 75
scale = 1/(nbm + 1)
target = 1e-12
# tf = 0.08
gees = np.logspace(-1.5, 1.8, 1000)
tfs = np.flip(np.logspace(-.8, 2, 1000))
colors = ['lightskyblue', 'dodgerblue', 'navy']


def timeunder(z, corrlevel):
    # feed cls largest to smallest
    # feed gees small to large
    gopt = np.zeros(np.size(corrlevel))
    taumax = np.zeros(np.size(corrlevel))
    for i, g in enumerate(gees):  # for every g level we want to investigate
        tf = tfs[i]  # there is a tf we have previously determined that captures the end of entanglement and not much longer
        ic = np.array([-0.5 * z * scale * nbm, -scale * nbm, 0])  # we set the IC according to the tilde eqns
        t, state = utils.rel_simulation(z, g, 0, 0, ic, target, tf)  # and simulate the squeeze
        corr = utils.rel_corr_var(state)  # get the correlation variance
        mc = np.min(corr)  # find its minimum
        # now I want to find time it is entangled below the targets i've set
        for j, cl in enumerate(corrlevel):  # so for each target level
            if mc > (scale * cl):  # if the minimum correlation doesn't break the target
                break  # end the loop and use a stronger g
            entangledindex = np.nonzero(corr < scale * cl)  # if it does then find the indexes where the correlation does
            tau = t[entangledindex[0][-1]] - t[entangledindex[0][0]]  # so we can see how long it stays under
            if tau < taumax[j]:  # if this time is (strictly) shorter than the current optimal
                continue  # we know it will not be the optimal
            taumax[j] = tau
            gopt[j] = g  # otherwise, we have a new optimal pump and need to see its tau for longer cls
        # print(f'g={g} done')

    return gopt


fig, ax = plt.subplots()
levels = np.flip(np.logspace(-1, 0, 35))
zetas = [0.99, 0.999, 1]
for i, zeta in enumerate(zetas):
    gopt = timeunder(zeta, levels)
    ax.loglog(levels, gopt, linewidth=2, label=zeta, color=colors[i])
# np.savetxt(f"optPumps{zeta}", opts)

ax.set_xlabel(r"$(\Delta_{12})_{target}^2$")
ax.set_ylabel(r"$g_{opt}$")
ax.legend(title="$\zeta$")
plt.tight_layout()
plt.show()
# plt.savefig(f"optPumpMulti1.pdf", format='pdf', dpi=1200, bbox_inches='tight')
