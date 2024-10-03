import numpy as np
import matplotlib.pyplot as plt
import coolingutils
import scipy.signal as signal

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })


zeta = 0.99
gamma = 2
nb1 = 0
nb2 = 5
initial = np.array([nb1, nb2, 0, 0])
target = 1e-9
tf = 200
# thetass = 0.5 * np.arctan(-2 * gamma)
# steady = (nb1 + nb2) * 0.5 - zeta * (nb2 - nb1) * 0.5
detunings = np.flip([0, 1, 5, 20])
# detunings = [0.5]
syts = np.flip(['-', '--', '-.', ':'])
colors = ['lightcoral', 'firebrick', 'darkred', 'red']


fig, ax = plt.subplots(2)
for i, detuning in enumerate(detunings):
    t, state = coolingutils.detuned_simulation(zeta, gamma, nb1, nb2, detuning, initial, target, tf)
    # transform = fft.fft(state[2])
    cleanup = ((state[2]/np.pi + 1) % 2) - 1
    ax[0].plot(t, state[2]/np.pi, linewidth=2, linestyle=syts[i], color=colors[i])

    # ax[1].plot(t, state[3], linewidth=2, label=zeta)
    # ax[1].set_xlabel(r"$\tilde{t}$")
    # ax[1].set_ylabel(r"$\sigma + \delta_p t$")
    # #
    # ax.loglog(t, state[0]*np.square(np.cos(state[2])) + state[1]*np.square(np.sin(state[2])), linewidth=1, color='b', linestyle=syts[i], label=detuning)
    ax[1].loglog(t, state[1]*np.square(np.cos(state[2])) + state[0]*np.square(np.sin(state[2])), linewidth=2, color=colors[i], linestyle=syts[i], label=detuning)
ax[0].set_xlabel(r"$\tilde{t}$")
ax[0].set_ylabel(r"$\theta/\pi$")
ax[0].set_xlim([0, 10])
ax[0].text(1, -0.1, "(a)", fontfamily="Computer Modern Roman", fontsize="large", horizontalalignment='center',
        verticalalignment='center')
ax[1].text(1, 0.2, "(b)", fontfamily="Computer Modern Roman", fontsize="large", horizontalalignment='center',
        verticalalignment='center')
ax[1].set_xlim([0.1, ax[1].get_xlim()[1]])
ax[1].set_xlabel(r"$\tilde{t}$")
ax[1].set_ylabel(r"$n_{m}$")
# ax.set_xlabel(r"$\tilde{t}$")
# ax.set_ylabel(r"$n$")
ax[1].legend(title=r"$\delta_r$")

ax[0].tick_params(axis='y', direction='in', top=True, right=True, which='both')
ax[1].tick_params(axis='y', direction='in', top=True, right=True, which='both')
ax[0].tick_params(axis='x', direction='in', top=True, left=True, which='major')
ax[1].tick_params(axis='x', direction='in', top=True, right=True, which='major')


plt.tight_layout()
plt.savefig("detuningEffects.pdf", format='pdf', dpi=1200, bbox_inches='tight')
# plt.show()
