import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import coolingutils
import utils
from multiprocessing import Pool


"""
scheme.py
Purpose: to determine the correlation variance minimum and time below 1 for multiple cooling SS
        Here I will vary the relative loss rates for our defined pump strengths and populations
        Creates fig 5 Right now
Date: 3 Sept 2024
Modified 24 Sept 2024
Paul RB Hughes
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

# tfs = [12, 3, 2]
# zetas = np.arange(0.98, 0.9999, 0.0001)
nbc = 0
nbm = 75
scale = 1/(nbm + 1)
# G, Z = np.meshgrid(gammas, zetas)
target = 1e-11
tf = 0.08
# gees = np.arange(0.5, 5.1, 0.1)
# gees = np.logspace(-1.5, 1.8, 100)
gees = [1]
# tfs = np.flip(np.logspace(-0.8, 2, 100))  # this is a really bad, hacky solution. Fine for now
tfs = [2]
zetas = [0.99, 1]
sts = ['-', '--', '-.']
colors = ['lightskyblue', 'dodgerblue', 'navy']
geelines = np.empty(np.size(gees))
zeelines = np.empty(np.size(zetas))
alphas = [0., 0.4, 1]

B = 0.01

fig, ax = plt.subplots()

for j, g in enumerate(gees):
    tf = tfs[j]
    for i, z in enumerate(zetas):
        ic = np.array([0, 0, 0])  # assumes large bath for mech
        t, state = utils.simulation(z, g, B, 0, 3, ic, target, tf)
        corr = utils.rel_corr_var(state) * (nbm + 1)
        ax.loglog(t, corr, linewidth=2, linestyle=sts[i], color=colors[j])


ax.set_xlabel(r"$\tilde{t}$")
threshold = ax.hlines(1, 0, tfs[0], linewidth=1.5, linestyles=":", color='black')
ax.set_ylim([ax.get_ylim()[0], 2])
ax.set_ylabel(r"$\Delta_{12}^2$")
solid = mlines.Line2D([], [], color=colors[2], linestyle=sts[0], label=zetas[0])
dashed = mlines.Line2D([], [], color=colors[2], linestyle=sts[1], label=zetas[1])
# dashdot = mlines.Line2D([], [], color=colors[2], linestyle=sts[2], label=zetas[2])
first_legend = ax.legend(handles=[solid, dashed], title="$\zeta$")
ax.add_artist(first_legend)
lsb = mlines.Line2D([], [], color=colors[0], linestyle=sts[0], label=gees[0])
# db = mlines.Line2D([], [], color=colors[1], linestyle=sts[0], label=gees[1])
# navy = mlines.Line2D([], [], color=colors[2], linestyle=sts[0], label=gees[2])
# ax.legend(handles=[lsb, db, navy], title="$g_{blue}$", loc='center left')
ax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
ax.tick_params(axis='x', direction='in', top=True, left=True, which='major')
plt.tight_layout()
plt.show()
# plt.savefig("transientEntanglement.pdf", format='pdf', dpi=1200, bbox_inches='tight')

