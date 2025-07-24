"""
Project: Optomechanical Cooling with the BTS
Purpose: This script produces the heatmap of Fig. 4 in the manuscript, according the steady-state behaviour of the BTS
Paul RB Hughes
Last Modified June 2025
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

# Control
savefig = False
pathway = "optomechanical/heatmap.pdf"

# Set parameters here
gr_array = np.arange(0, 3.01, 0.01)
k = np.arange(0, 6, 0.2)  # to make the zetas
zetas = 1 - 1/(10**k)
saveplots = False
path = "plots/heatmap.pdf"

# set parameters
gr = np.arange(0, 3.01, 0.01)
k = np.arange(0, 6, 0.2)  # this corresponds to the y-axis of the heatmap. zero maps to zero
zetas = 1 - 1/(10**k)  # zetas calculated from such
nbc = 0
nbm = 8334.15  # According to the oscillator at T=4K
hbaromegaonk = 0.000119981  # 10MHz
hbaromegaonkopt = 0.2663576  # 1THz
G, K = np.meshgrid(gr, k)
G, Z = np.meshgrid(gr, zetas)

# plotting
fig, ax = plt.subplots()
theta = 0.5 * np.arctan(-2 * G)  # steady-state thetas
cooled = ((1 - Z) * nbm * np.square(np.cos(theta)) + (1 + Z) * nbc * np.square(np.sin(theta)))/(1 - Z * np.cos(2 * theta))
opt = ((1 + Z) * nbc * np.square(np.cos(theta)) + (1 - Z) * nbm * np.square(np.sin(theta)))/(1 + Z * np.cos(2 * theta))
cooltot = cooled * np.square(np.cos(theta)) + opt * np.square(np.sin(theta))  # total population is the observable
optot = cooled * np.square(np.sin(theta)) + opt * np.square(np.cos(theta))
temp = hbaromegaonk / np.log(1 + 1/cooltot)

steadycor = ax.contourf(G, K, np.log10(temp), cmap='copper', origin="lower")
ax.set_xlabel("$g_r$")
ax.set_ylabel("$-\log_{10}(1 - \zeta)$")
plt.tight_layout()

fig.colorbar(steadycor, ax=ax, label=r"$\log_{10}(T_{eff})$")

if saveplots:
    plt.savefig(path, format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

