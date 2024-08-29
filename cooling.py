import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

gammas = np.arange(0, 5, 0.01)
zetas = np.arange(0, 0.99, 0.004)
nbc = 0
nbm = 800000
hbaromegaonk = 0.0000959847469  # 2MHz
hbaromegaonkopt = 0.2663576  # 1THz
G, Z = np.meshgrid(gammas, zetas)

fig, ax = plt.subplots()
theta = 0.5 * np.arctan(-2 * G)
cooled = ((1 - Z) * nbm * np.square(np.cos(theta)) + (1 + Z) * nbc * np.square(np.sin(theta)))/(1 - Z * np.cos(2 * theta))
opt = ((1 + Z) * nbc * np.square(np.cos(theta)) + (1 - Z) * nbm * np.square(np.sin(theta)))/(1 + Z * np.cos(2 * theta))
cooltot = cooled * np.square(np.cos(theta)) + opt * np.square(np.sin(theta))
optot = cooled * np.square(np.sin(theta)) + opt * np.square(np.cos(theta))
temp = hbaromegaonk / np.log(1 + 1/cooltot)
otemp = hbaromegaonkopt / np.log(1 + 1/optot)

# print(theta)

steadycor = ax.contourf(G, Z, otemp, cmap='copper', origin="lower")
ax.set_xlabel("$|g|$")
ax.set_ylabel("$\zeta$")
# ax.set_title("Steady-State")
plt.tight_layout()

fig.colorbar(steadycor, ax=ax, label=r"$T_c$")

# plt.show()
plt.savefig("heatmapopt.pdf", format='pdf', dpi=1200, bbox_inches='tight')

