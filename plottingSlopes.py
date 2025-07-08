import numpy as np
import matplotlib.pyplot as plt

path = 'ratios/'
# zetas = np.array([0.6, 0.8, 0.9, 0.95])
zetas = np.arange(0.5, 0.93, 0.01)
levels = np.flip(np.logspace(-2.3, 0, 10000))
C, Z = np.meshgrid(levels, zetas)
G = np.zeros(np.shape(Z))
T = np.zeros(np.shape(Z))
# styles = ['-', '--', ':', '-.']
fig, ax = plt.subplots()
for i, zeta in enumerate(zetas):
    gopt = np.load(path+f"Gopts{zeta}.npy")
    G[:][i] = gopt[:]
    taus = np.load(path + f"Topts{zeta}.npy")
    T[:][i] = taus[:]
    # levels = np.load(path + f"targets{zeta}.npy")
    # ax[0].loglog(levels, gopt)
    # ax[1].loglog(levels, taus)
ax.contourf(np.log10(C), Z, 10*np.log10(G))

plt.show()

