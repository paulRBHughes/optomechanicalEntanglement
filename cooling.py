import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

gammas = np.arange(0, 3.01, 0.01)
k = np.arange(0, 6, 0.2)
zetas = 1 - 1/(10**k)
nbc = 0
nbm = 8334.15
hbaromegaonk = 0.000119981  # 10MHz
hbaromegaonkopt = 0.2663576  # 1THz
G, K = np.meshgrid(gammas, k)
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

steadycor = ax.contourf(G, K, np.log10(temp), cmap='copper', origin="lower")
ax.set_xlabel("$g_r$")
ax.set_ylabel("$-\log_{10}(1 - \zeta)$")
# ax.set_title("Steady-State")
plt.tight_layout()

fig.colorbar(steadycor, ax=ax, label=r"$\log_{10}(T_{eff})$")

# plt.show()
plt.savefig("heatmap.pdf", format='pdf', dpi=1200, bbox_inches='tight')

