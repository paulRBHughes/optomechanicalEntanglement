import numpy as np
import matplotlib.pyplot as plt
import coolingutils
from multiprocessing import Pool


plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

gammas = np.arange(1, 5, 0.01)
zetas = np.arange(0, 0.99, 0.004)
nbc = 0
nbm = 800000
initial = np.array([nbc, nbm, 0])
target = 1e-8
tf = 5
# hbaromegaonk = 0.0000959847469  # 2MHz
# hbaromegaonkopt = 0.2663576  # 1THz
# G, Z = np.meshgrid(gammas, zetas)
tmin = np.zeros([np.size(gammas), np.size(zetas)])
minn = np.zeros([np.size(gammas), np.size(zetas)])


def sims(zeta, gamma, nbc, nbm, initial, target, tf):
    t, state = coolingutils.simulation(zeta, gamma, nbc, nbm, initial, target, tf)
    pop = state[1, 1:] * np.square(np.cos(state[2, 1:])) + state[0, 1:] * np.square(np.sin(state[2, 1:]))
    return np.array([np.min(pop), t[np.argmin(pop)+1]])



for j, zeta in enumerate(zetas):


    def wrapper(gammas):
        return sims(zeta, gammas, nbc, nbm, initial, target, tf)


    with Pool(processes=15) as pool:
        a = pool.map(wrapper, gammas)
        minn[:, j] = a[0]
        tmin[:, j] = a[1]

np.savetxt("minpop", minn)
np.savetxt("tmin", tmin)


