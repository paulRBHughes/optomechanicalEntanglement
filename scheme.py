import numpy as np
import matplotlib.pyplot as plt
import coolingutils
import utils

"""
scheme.py
Purpose: to determine the correlation variance minimum and time below 1 for multiple cooling SS
        Here I will vary the relative loss rates for our defined pump strengths and populations
Date: 3 Sept 2024
Paul RB Hughes
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

# gammas = np.array([0.5, 1, 2])
zetas = np.arange(0.92, 0.9999, 0.0001)
nbc = 0
nbm = 75
scale = 1/(nbm + 1)
hbaromegaonk = 0.0000959847469  # 2MHz
hbaromegaonkopt = 0.2663576  # 1THz
# G, Z = np.meshgrid(gammas, zetas)
target = 1e-11
tf = 0.08

# Large pumping limit, subcritical squeeze
# to help sim along, we use the unthermalized equations and approximate (pretty well tbh).
gees = [1, 4, 10]
tfs = [5, 3, 1]


def largecooled(i):
    g = gees[i]
    tf = tfs[i]
    mincorr = np.zeros(np.size(zetas))
    enttime = np.zeros(np.size(zetas))
    for i, z in enumerate(zetas):
        ic = np.array([-0.5 * z * scale * nbm, -1, 0])  # assumes large bath for mech
        t, state = utils.rel_simulation(z, g, 0, 0, ic, target, tf)
        corr = utils.rel_corr_var(state)
        mc = np.min(corr)
        mincorr[i] = mc
        # now I want to find time it is entangled
        if mc > scale:
            continue
        entangledindex = np.nonzero(corr < scale)
        enttime[i] = t[entangledindex[0][-1]] - t[entangledindex[0][0]]
    # print(mincorr)
    # print(enttime)
    return mincorr, enttime

fig, ax = plt.subplots()
for i, g in enumerate(gees):
    mc, et = largecooled(i)
    np.savetxt(f"Tcorr{g}", et)
    # ax[0].plot(zetas, mc, label=g, linewidth=2)
    # ax[0].set_xlabel(r"$\zeta$")
    # ax[0].set_ylabel(r"$\tilde{\Delta}_{12, \mathrm{min}}^2$")
    # ax[1].plot(zetas, et, label=g, linewidth=2)
    # ax[1].set_xlabel(r"$\zeta$")
    # ax[1].set_ylabel("$T$")
    ax.plot(zetas, et, label=g, linewidth=2)
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\tau$")
# ax[1].legend(title="$g$")
ax.legend(title="$g_{blue}$")
plt.tight_layout()
# plt.show()
plt.savefig("entanglementTime.pdf", format='pdf', dpi=1200, bbox_inches='tight')

