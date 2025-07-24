import numpy as np
import matplotlib.pyplot as plt
from os import listdir

"""
This was for plotting the results of the sims in slopes. Deprecated.
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "18",
        "figure.autolayout": False
    })

saveplots = False

levels = np.flip(np.logspace(-4.5, -0.3, 10000))
# zetas = np.arange(0.99, 0.99991, 0.0001)
# zetas = np.append(np.arange(0.36, 0.99, 0.01), [0.99])
zetas = np.arange(0.99, 0.99991, 0.0001)
C, Z = np.meshgrid(levels, zetas)

path = 'ratios/zetatoone/nInf/'
GInf = np.zeros(np.shape(Z))
TInf = np.zeros(np.shape(Z))
for i, zeta in enumerate(zetas):
    gopt = np.load(path+f"Gopts{zeta}.npy")
    GInf[:][i] = gopt[:]
    taus = np.load(path + f"Topts{zeta}.npy")
    TInf[:][i] = taus[:]

path = 'ratios/zetatoone/nNine/'
GOne = np.zeros(np.shape(Z))
TOne = np.zeros(np.shape(Z))
for i, zeta in enumerate(zetas):
    gopt = np.load(path + f"Gopts{zeta}.npy")
    GOne[:][i] = gopt[:]
    taus = np.load(path + f"Topts{zeta}.npy")
    TOne[:][i] = taus[:]

path = 'ratios/zetatoone/nTwentyFour/'
GZero = np.zeros(np.shape(Z))
TZero = np.zeros(np.shape(Z))
for i, zeta in enumerate(zetas):
    gopt = np.load(path + f"Gopts{zeta}.npy")
    GZero[:][i] = gopt[:]
    taus = np.load(path + f"Topts{zeta}.npy")
    TZero[:][i] = taus[:]

# zetas = np.zeros(int(np.size(listdir(path))/2))
# i = 0
# for file in listdir(path):
#     if "Gopts" in file:
#         zetas[i] = file[5:-4]
#         i += 1
# zetas = np.sort(zetas)
# zetas = zetas[4:]
# zetas = np.arange(0.99, 0.999, 0.001)

# levels = np.flip(np.logspace(-4.7, 0, 10000))
barlevels = np.linspace(-2, 2.9, 9)
# tbarlevles = np.linspace(-3, 3, 7)
# C, Z = np.meshgrid(levels, zetas)
# G = np.zeros(np.shape(Z))
# T = np.zeros(np.shape(Z))
# styles = ['-', '--', ':', '-.']
fig, ax = plt.subplots(3)
figD1, axD1 = plt.subplots()
# figD2, axD2 = plt.subplots()
figD1T, axD1T = plt.subplots()
# figD2T, axD2T = plt.subplots()
# for i, zeta in enumerate(zetas):
#     gopt = np.load(path+f"Gopts{zeta}.npy")
#     G[:][i] = gopt[:]
#     taus = np.load(path + f"Topts{zeta}.npy")
#     T[:][i] = taus[:]
    # ax[0].loglog(levels, gopt)
    # ax[1].loglog(levels, taus)
gcf = ax[0].contourf(np.log10(C), Z, np.log10(TZero))
ax[0].text(-1.3, 0.6, f'$n_m^b = 0$', bbox=dict(alpha=0.8, ec='k', fc='white', boxstyle='square'), size='16')
ax[1].contourf(C, Z, np.log10(TOne))
ax[1].text(-1.3, 0.6, f'$n_m^b = 1$', bbox=dict(alpha=0.8, ec='k', fc='white', boxstyle='square'), size='16')
ax[2].contourf(C, Z, np.log10(TInf))
ax[2].text(-1.3, 0.6, f'$n_m^b = \infty$', bbox=dict(alpha=0.8, ec='k', fc='white', boxstyle='square'), size='16')
bar = fig.colorbar(gcf, ax=ax[:]).set_label(label=r"$\log_{10} g_{opt}$")
ax[1].set_ylabel("$\zeta$")
ax[2].set_xlabel(r"$\log_{10} \tilde \Delta_t^2$")
plt.subplots_adjust(wspace=0.01,hspace=0.01)

diffbarlevel = np.linspace(0, 100, 7)

tcf = axD1.contourf(np.log10(C), Z, (GZero - GInf)/GInf, diffbarlevel)
bar = figD1.colorbar(tcf, ax=axD1).set_label(label=r"$(g_{opt}(n_b^m=0) - g_{opt}(n_b^m=\infty))/g_{opt}(n_b^m=\infty)$")
axD1.set_xlabel(r"$\log_{10} \tilde \Delta_t^2$")
axD1.set_ylabel(r"$\zeta$")
axD1.set_xlim(-3.5, -0.3)
axD1.set_ylim(0.99, 0.999)

# hcf = axD2.contourf(np.log10(C), Z, (GOne - GInf)/GInf, levels=diffbarlevel)
# tbar = figD2.colorbar(hcf, ax=axD2).set_label(label=r"$(g_{opt}(n_b^m=1) - g_{opt}(n_b^m=\infty))/g_{opt}(n_b^m=\infty)$")
# axD2.set_xlabel(r"$\log_{10} \tilde \Delta_t^2$")
# axD2.set_ylabel(r"$\zeta$")
# # axD2.set_ylim(0.4, 0.9)
# # ax[1].set_xlim(-4.6, -0.3)
# # ax[1].set_xlim(ax[1].get_xlim()[0], -0.3)

tcft = axD1T.contourf(np.log10(C), Z, (TZero - TInf)/TInf)
bart = figD1T.colorbar(tcft, ax=axD1T).set_label(label=r"$(\tilde\tau_{max}(n_b^m=0) - \tilde\tau_{max}(n_b^m=\infty))/\tilde\tau_{max}(n_b^m=\infty)$")
axD1T.set_xlabel(r"$\log_{10} \tilde \Delta_t^2$")
axD1T.set_ylabel(r"$\zeta$")
#
# hcft = axD2T.contourf(np.log10(C), Z, (TOne - TInf)/TInf)
# tbart = figD2T.colorbar(hcft, ax=axD2T).set_label(label=r"$(\tilde\tau_{max}(n_b^m=1) - \tilde\tau_{max}(n_b^m=\infty))/\tilde\tau_{max}(n_b^m=\infty)$")
# axD2T.set_xlabel(r"$\log_{10} \tilde \Delta_t^2$")
# axD2T.set_ylabel(r"$\zeta$")

if saveplots:
    # fig.savefig("ratios/raws.pdf", format='pdf', dpi=1200, bbox_inches='tight')
    figD1.savefig("ratios/geescope.pdf", format='pdf', dpi=1200, bbox_inches='tight')
    # figD2.savefig("ratios/diffs2ratio.pdf", format='pdf', dpi=1200, bbox_inches='tight')

    # figD1T.savefig("ratios/diffs1ratiotime.pdf", format='pdf', dpi=1200, bbox_inches='tight')
    # figD2T.savefig("ratios/diffs2ratiotime.pdf", format='pdf', dpi=1200, bbox_inches='tight')

plt.show()
