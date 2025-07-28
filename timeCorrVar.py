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
optimalPumpVaryBath.py
Purpose: Determine the pump that creates the longest lasting level under a certain correlation variance.
          creates fig 8 of manuscript
Date: 23 Sept 2024
Modified Often
Paul RB Hughes
"""

# nbc = 0  # again, assuming this is very small
nbm = 75
scale = 1/(nbm + 1)
target = 1e-12  # simulation precision
gees = np.logspace(-1, 2, 500)
tf = 10
cv_target = 0.8
corrmax = (nbm + 1)
show_opts = True
savefigs = False
path = "plots/"


def timeunder(z, corrlevel, optimal=True):

    tau = np.zeros(np.size(gees))
    for i, g in enumerate(gees):
        ic = np.array([-0.5 * z * scale * nbm, -scale * nbm, 0])  # assumes large bath for mech
        t, state = utils.rel_simulation(z, g, 0, 0, ic, corrmax, target, tf)
        corr = utils.rel_corr_var(state)
        mc = np.min(corr)
        # now I want to find time it is entangled
        if mc > corrlevel * scale:
            continue
        entangledindex = np.nonzero(corr < corrlevel * scale)
        tau[i] = t[entangledindex[0][-1]] - t[entangledindex[0][0]]
    gopt = gees[np.argmax(tau)]
    if show_opts:
        print(f"g_opt = {gopt}, taumax = {np.max(tau)}")
    if optimal:
        return gopt
    return tau


zetas = [1, 0.999, 0.99]
fig, ax = plt.subplots()
styts = ["-", '--', '-.', ':']
colors = ['navy', 'dodgerblue', 'lightskyblue']
for i, zeta in enumerate(zetas):
    ax.semilogx(gees, timeunder(zeta, cv_target, False), linewidth=2, label=zeta, linestyle=styts[i], color=colors[0])

ax.set_xlabel(r"$g_b$")
ax.set_ylabel(rf"$\tilde\tau({cv_target})$")
ax.legend(title="$\zeta$")
plt.tight_layout()
if savefigs:
    plt.savefig(path + f"tildetau{cv_target}.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
