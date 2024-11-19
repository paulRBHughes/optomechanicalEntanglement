import numpy as np
import matplotlib.pyplot as plt
"""
Creates heatmap of population
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })
B = 0.01
Omegasq = 1/(16*B**2 + 1)

gammas = np.arange(0, 10.01, 0.01)
k = np.arange(0, 6, 0.2)
zetas = 1 - 1/(10**k)
nbc = 0
nbm = 800000
hbaromegaonk = 0.0000959847469  # 2MHz
hbaromegaonkopt = 0.2663576  # 1THz
G, K = np.meshgrid(gammas, k)
G, Z = np.meshgrid(gammas, zetas)

theta = 0.5 * np.arctan(2*G*(1+Z)/(2*(1+Z)+np.square(G)*Omegasq))
q = np.square(G)*Omegasq/(1+Z)
r = q*B**2 * 0.0625
cooled = (((1 - Z) * nbm * np.square(np.cos(theta)) + (1 + Z) * nbc * np.square(np.sin(theta)) + r*np.square(np.cos(theta)))
          /(1 - Z * np.cos(2 * theta) + q * np.square(np.cos(theta))))
opt = (((1 + Z) * nbc * np.square(np.cos(theta)) + (1 - Z) * nbm * np.square(np.sin(theta)) + r*np.square(np.sin(theta)))
       /(1 + Z * np.cos(2 * theta) + q * np.square(np.sin(theta))))

fig, ax = plt.subplots()
# theta = 0.5 * np.arctan(-2 * G)
# cooled = ((1 - Z) * nbm * np.square(np.cos(theta)) + (1 + Z) * nbc * np.square(np.sin(theta)))/(1 - Z * np.cos(2 * theta))
# opt = ((1 + Z) * nbc * np.square(np.cos(theta)) + (1 - Z) * nbm * np.square(np.sin(theta)))/(1 + Z * np.cos(2 * theta))
cooltot = cooled * np.square(np.cos(theta)) + opt * np.square(np.sin(theta))
optot = cooled * np.square(np.sin(theta)) + opt * np.square(np.cos(theta))
temp = hbaromegaonk / np.log(1 + 1/cooltot)
otemp = hbaromegaonkopt / np.log(1 + 1/optot)

# print(theta)

steadycor = ax.contourf(G, K, np.log10(temp), cmap='copper', origin="lower")
ax.set_xlabel("$|g|$")
ax.set_ylabel("$\log_{10}(1 - \zeta)$")
# ax.set_title("Steady-State")
plt.tight_layout()

fig.colorbar(steadycor, ax=ax, label=r"$\log_{10}(T_m)$")

plt.show()
# plt.savefig("heatmap.pdf", format='pdf', dpi=1200, bbox_inches='tight')

