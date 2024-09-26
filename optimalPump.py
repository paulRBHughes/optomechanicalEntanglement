import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
import matplotlib.lines as mlines

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
gees = np.logspace(-1.5, 1.8, 10000)
tfs = np.flip(np.logspace(-.8, 2, 10000))
colors = ['lightskyblue', 'dodgerblue', 'navy']


def optimal(z, corrlevel):
    # feed cls largest to smallest
    # feed gees small to large
    gopt = np.zeros(np.size(corrlevel))
    # tausarray = np.zeros([np.size(gees), np.size(corrlevel)])
    taumax = np.zeros(np.size(corrlevel))
    with Pool(processes=15) as pool:
        tausarray = pool.map(timeunder, range(np.size(gees)))
    for j in range(np.size(corrlevel)):  # I hate this...
        corrtimes = np.zeros(np.size(gees))
        for n in range(np.size(gees)):
            corrtimes[n] = tausarray[n][j]
        taumax[j] = np.max(corrtimes)
        gopt[j] = gees[np.argmax(corrtimes)]

    return gopt, taumax


gfig, gax = plt.subplots()
tfig, tax = plt.subplots()
levels = np.flip(np.logspace(-1, 0, 100))
zetas = [0.99, 0.999, 1]
for i, zeta in enumerate(zetas):


    def timeunder(k):
        g = gees[k]
        tf = tfs[k]  # there is a tf we have previously determined that captures the end of entanglement and not much longer
        ic = np.array([-0.5 * zeta * scale * nbm, -scale * nbm, 0])  # we set the IC according to the tilde eqns
        t, state = utils.rel_simulation(zeta, g, 0, 0, ic, target, tf)  # and simulate the squeeze
        corr = utils.rel_corr_var(state)  # get the correlation variance
        mc = np.min(corr)  # find its minimum
        taus = np.zeros(np.size(levels))
        # now I want to find time it is entangled below the targets i've set
        for j, cl in enumerate(levels):  # so for each target level
            if mc > (scale * cl):  # if the minimum correlation doesn't break the target
                break  # end the loop and use a stronger g
            entangledindex = np.nonzero(corr < scale * cl)  # if it does then find the indexes where the correlation does
            tau = t[entangledindex[0][-1]] - t[entangledindex[0][0]]  # so we can see how long it stays under
            taus[j] = tau
        print(f'g={g} done...')
        return taus


    opts = optimal(zeta, levels)
    gax.loglog(levels, opts[0], linewidth=2, label=zeta, color=colors[i], linestyle='-')
    gmin = (1 + (1 - zeta)*nbm)*np.reciprocal(levels) - 1
    gax.loglog(levels, gmin, linewidth=2, color=colors[i], linestyle='--')
    tax.loglog(levels, opts[1], linewidth=2, label=zeta, color=colors[i], linestyle='-')
# np.savetxt(f"optPumps{zeta}", opts)

gax.set_xlabel(r"$\Delta_{target}$")
gax.set_ylabel(r"$g_{blue}$")

solid = mlines.Line2D([], [], color=colors[2], linestyle='-', label='$g_{opt}$')
dashed = mlines.Line2D([], [], color=colors[2], linestyle='--', label='$g_{min}$')
first_legend = gax.legend(handles=[solid, dashed])
gax.add_artist(first_legend)

gax.legend(title="$\zeta$")

tax.set_xlabel(r"$\Delta_{target}$")
tax.set_ylabel(r"$\tau_{max}$")
tax.legend(title="$\zeta$")

gax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
tax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
gax.tick_params(axis='x', direction='in', top=True, left=True, which='major')
tax.tick_params(axis='x', direction='in', top=True, right=True, which='major')

plt.tight_layout()
# plt.show()
gfig.savefig(f"optimalPump.pdf", format='pdf', dpi=1200, bbox_inches='tight')
tfig.savefig(f"optimalTime.pdf", format='pdf', dpi=1200, bbox_inches='tight')
