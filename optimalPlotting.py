import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

zeta = 0.999
N = np.load(f'ZvargoptsNBS.npy')
C = np.load(f'ZvargoptsLEV.npy')
T = np.load(f'Zvartopts.npy')
G = np.load(f'Zvargopts.npy')

logG = np.log10(G)
logT = np.log10(T)

fig, ax = plt.subplots()
GCF = ax.contourf(C, N, logG, cmap='viridis', origin="lower")
fig.colorbar(GCF, ax=ax, label=r"$\log_{10} g_{opt}$")
ax.set_xlabel(r"$\Delta^2_{t}$")
ax.set_ylabel(r"$\zeta$")
plt.tight_layout()
fig.savefig(f"optimalPumpVaryZeta.pdf", format='pdf', dpi=1200)

fig, ax = plt.subplots()
TCF = ax.contourf(C, N, logT, cmap='viridis', origin="lower")
fig.colorbar(GCF, ax=ax, label=r"$\log_{10} \tilde\tau_{opt}$")
ax.set_xlabel(r"$\Delta^2_{t}$")
ax.set_ylabel(r"$\zeta$")
plt.tight_layout()
fig.savefig(f"optimalTimeVaryZeta.pdf", format='pdf', dpi=1200)
