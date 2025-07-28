import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

"""
Purpose: to show how different zeta changes the mode populations in the steady-state for different pumps. 
            Produces fig. 3 of the manuscript.
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

savefig = False
path = "plots/drivevss.pdf"

# Choose params
gr = np.arange(0.001, 10.001, 0.001)
cav_pops = 497.83972 * np.square(gr)
nbc = 0
nbm = 40
Zs = [0.9, 0.99, 0.999, 0.9999]  # Zetas
styles = [":", "-.", '--', '-']
reds = ['lightcoral', 'firebrick', 'darkred', 'red']
blues = ['lightskyblue', 'dodgerblue', 'navy', 'blue']

fig, ax = plt.subplots()
theta = 0.5 * np.arctan(-2 * gr)  # Steady-state theta
for i, Z in enumerate(Zs):
    cooled = ((1 - Z) * nbm * np.square(np.cos(theta)) + (1 + Z) * nbc * np.square(np.sin(theta)))/(1 - Z * np.cos(2 * theta))
    opt = ((1 + Z) * nbc * np.square(np.cos(theta)) + (1 - Z) * nbm * np.square(np.sin(theta)))/(1 + Z * np.cos(2 * theta))
    cooltot = cooled * np.square(np.cos(theta)) + opt * np.square(np.sin(theta))  # The actual observable
    optot = cooled * np.square(np.sin(theta)) + opt * np.square(np.cos(theta))
    ax.loglog(gr, cooltot, linestyle=styles[i], color=reds[i], label=Z)
    ax.loglog(gr, optot, linestyle=styles[i], color=blues[i])


ax.set_xlabel(r"$g_r$")
ax.set_ylabel("$n^{ss}$")
ax.legend(title="$\zeta$", loc='center left')
plt.tight_layout()

if savefig:
    plt.savefig(path, format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
