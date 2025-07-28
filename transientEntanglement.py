import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import coolingutils
import utils
from multiprocessing import Pool


"""
transientEntanglement.py
Purpose: to determine the correlation variance minimum and time below 1 for multiple cooling SS
        Here I will vary the relative loss rates for our defined pump strengths and populations
Date: 3 Sept 2024
Modified often
Paul RB Hughes
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

savefig = False
path = "plots/transientEntanglement.pdf"
# nbc = 0  # Optical mode bath population (zero is typical, or close enough as makes no matter), so ignored here
nbm = 75  # mechanical mode bath population
scale = 1/(nbm + 1)  # this comes up a lot when using the bathless equations. Must change if nonzero nbc
target = 1e-12
gees = [1, 5, 20]
tf = 10  # This is more than enough since the simulation stops when we get above corrmax
corrmax = 2 * (nbm + 1)
zetas = [0.99, 1]
sts = ['--', '-']  # linestyles for the zetas on the plot
colors = ['lightskyblue', 'dodgerblue', 'navy']  # colors for the different gees

fig, ax = plt.subplots()

for j, g in enumerate(gees):
    for i, z in enumerate(zetas):
        ic = np.array([-0.5 * z * scale * nbm, -scale * nbm, 0])  # assumes ncb ~ 0
        t, state = utils.rel_simulation(z, g, 0, 0, ic, corrmax, target, tf)
        corr = utils.rel_corr_var(state) * (nbm + 1)  # Need to scale this and the max up
        ax.loglog(t, corr, linewidth=2, linestyle=sts[i], color=colors[j])


ax.set_xlabel(r"$\tilde{t}$")
threshold = ax.hlines(1, 0, tf, linewidth=1.5, linestyles=":", color='black')  # correlation threshold
ax.set_ylim([ax.get_ylim()[0], 2])
ax.set_ylabel(r"$\Delta_{12}^2$")
solid = mlines.Line2D([], [], color=colors[2], linestyle=sts[0], label=zetas[0])
dashed = mlines.Line2D([], [], color=colors[2], linestyle=sts[1], label=zetas[1])
first_legend = ax.legend(handles=[solid, dashed], title="$\zeta$", loc='lower right')
ax.add_artist(first_legend)
lsb = mlines.Line2D([], [], color=colors[0], linestyle=sts[1], label=gees[0])
db = mlines.Line2D([], [], color=colors[1], linestyle=sts[1], label=gees[1])
navy = mlines.Line2D([], [], color=colors[2], linestyle=sts[1], label=gees[2])
ax.legend(handles=[lsb, db, navy], title="$g$", loc='lower left')
ax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
ax.tick_params(axis='x', direction='in', top=True, left=True, which='major')
plt.tight_layout()
if savefig:
    plt.savefig(path, format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

