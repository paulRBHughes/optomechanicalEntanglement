import numpy as np
import matplotlib.pyplot as plt
import coolingutils
import utils
import matplotlib.lines as mlines
from multiprocessing import Pool

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

"""
optimalPump.py
Purpose: Determine the pump that creates the longest lasting level under a certain correlation variance.
Date: 23 Sept 2024
Paul RB Hughes
"""

# nbc = 0
nbm = 150
scale = 1/(nbm + 1)
target = 1e-11
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


cls = [0.8]  # I want to see how long the pump can keep the correlation variance under this value for
zetas = [1, 0.999, 0.99]
fig, ax = plt.subplots()
styts = ["-", '--', '-.', ':']
colors = ['navy', 'dodgerblue', 'lightskyblue']
for j, cl in enumerate(cls):
    for i, zeta in enumerate(zetas):
        ax.semilogx(gees, timeunder(zeta, cl, False), linewidth=2, label=zeta, linestyle=styts[i], color=colors[j])

ax.set_xlabel(r"$g$")
ax.set_ylabel(r"$\tilde{\tau}(0.8)$")
ax.legend(title="$\zeta$")
plt.tight_layout()
ax.set_xlim([0.2, ax.get_xlim()[1]])
# solid = mlines.Line2D([], [], color=colors[0], linestyle=styts[0], label=zetas[0])
# dashed = mlines.Line2D([], [], color=colors[0], linestyle=styts[1], label=zetas[1])
# dashdot = mlines.Line2D([], [], color=colors[2], linestyle=styts[2], label=zetas[2])
# first_legend = ax.legend(handles=[solid, dashed], title="$\zeta$")
# ax.add_artist(first_legend)
# lsb = mlines.Line2D([], [], color=colors[0], linestyle=styts[0], label=cls[0])
# db = mlines.Line2D([], [], color=colors[1], linestyle=styts[0], label=cls[1])
# navy = mlines.Line2D([], [], color=colors[2], linestyle=styts[0], label=cls[2])
# ax.legend(handles=[lsb, db, navy], title="$(\Delta_{12}^2)_{target}$", loc='lower left')
plt.show()
# plt.savefig(f"tildetau0.8.pdf", format='pdf', dpi=1200, bbox_inches='tight')
