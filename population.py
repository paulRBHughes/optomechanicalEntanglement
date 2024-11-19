import numpy as np
import matplotlib.pyplot as plt

"""
Current fig 3 in manuscript - population vs g plot
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

gees = np.logspace(-4, 5, 10000)
# cav_pops = 497.83972 * np.square(gees)
# k = np.arange(0, 6, 0.2)
# zetas = 1 - 1/(10**k)
nbc = 0
nbm = 40
# hbaromegaonk = 0.0000959847469  # 2MHz
# hbaromegaonkopt = 0.2663576  # 1THz
# G, K = np.meshgrid(gammas, k)
# G, Z = np.meshgrid(gammas, zetas)
Zs = [0.9, 0.99, 0.999, 0.9999]
styles = [":", "-.", '--', '-']
reds = ['lightcoral', 'firebrick', 'darkred', 'red']
blues = ['lightskyblue', 'dodgerblue', 'navy', 'blue']

# detuning_factor = 10
B = 0.05
Omegasq = 1/(16*B**2 + 1)

fig, ax = plt.subplots()
# theta = 0.5 * np.arctan(-2 * gammas)
for i, Z in enumerate(Zs):
    theta = 0.5 * np.arctan(2*gees*(1+Z)/(2*(1+Z)+np.square(gees)*Omegasq))
    q = np.square(gees)*Omegasq/(1+Z)
    r = q*B**2 * 0.0625
    cooled = (((1 - Z) * nbm * np.square(np.cos(theta)) + (1 + Z) * nbc * np.square(np.sin(theta)) + r*np.square(np.cos(theta)))
              /(1 - Z * np.cos(2 * theta) + q * np.square(np.cos(theta))))
    opt = (((1 + Z) * nbc * np.square(np.cos(theta)) + (1 - Z) * nbm * np.square(np.sin(theta)) + r*np.square(np.sin(theta)))
           /(1 + Z * np.cos(2 * theta) + q * np.square(np.sin(theta))))
    cooltot = cooled * np.square(np.cos(theta)) + opt * np.square(np.sin(theta))
    optot = cooled * np.square(np.sin(theta)) + opt * np.square(np.cos(theta))
    ax.loglog(gees, cooltot, linestyle=styles[i], color=reds[i], label=Z)
    # ax.loglog(gees, optot, linestyle=styles[i], color=blues[i])
# temp = hbaromegaonk / np.log(1 + 1/cooltot)
# otemp = hbaromegaonkopt / np.log(1 + 1/optot)

# print(theta)


# ax.set_xlabel(r"$\bar{n}_c$")
ax.set_xlabel(r'$g$')
ax.set_ylabel("$n^{ss}$")
# ax.set_xlim(1, 2e5)
ax.legend(title="$\zeta$", loc='center left')
# ax.set_title("Steady-State")
plt.tight_layout()

plt.show()
# plt.savefig("Stokesdrivevss.pdf", format='pdf', dpi=1200, bbox_inches='tight')

