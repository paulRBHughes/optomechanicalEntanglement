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
gees = np.logspace(-1.5, 1.8, 100)
tfs = np.flip(np.logspace(-.8, 2, 100))


def timeunder(z, corrlevel, optimal=True):
    tau = np.zeros(np.size(gees))
    for i, g in enumerate(gees):
        tf = tfs[i]
        ic = np.array([-0.5 * z * scale * nbm, -scale * nbm, 0])  # assumes large bath for mech
        t, state = utils.rel_simulation(z, g, 0, 0, ic, target, tf)
        corr = utils.rel_corr_var(state)
        mc = np.min(corr)
        # now I want to find time it is entangled
        if mc > (scale * corrlevel):
            continue
        entangledindex = np.nonzero(corr < scale * corrlevel)
        tau[i] = t[entangledindex[0][-1]] - t[entangledindex[0][0]]
    gopt = gees[np.argmax(tau)]
    print(gopt)
    if optimal:
        return gopt
    return tau


cl = 2e-1  # I want to see how long the pump can keep the correlation variance under this value for
zetas = [1]
fig, ax = plt.subplots()
styts = ["-", '--', '-.', ':']
for i, zeta in enumerate(zetas):
    ax.semilogx(gees, timeunder(zeta, cl, False), label=zeta, linewidth=2, linestyle=styts[i])

ax.set_xlabel(r"$g$")
ax.set_ylabel(rf"$\tau({cl})$")
ax.legend(title="$\zeta$")
plt.tight_layout()
plt.show()
# plt.savefig(f"timeunder{cl}.pdf", format='pdf', dpi=1200, bbox_inches='tight')
