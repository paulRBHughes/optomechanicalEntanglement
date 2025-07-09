"""
Project: optomechanical cooling
Purpose: Model ODE equations and make plots to cool mechanical mode
July 2024
Paul RB Hughes
"""

# Import
import numpy as np
from numba import njit



def eom(zeta, gr, nb1, nb2, state):
    # This function holds the coupled ODE equations for the system exactly on sideband.
    n1 = state[0]
    n2 = state[1]
    theta = state[2]
    deltan = n2 - n1
    deltanb = nb2 - nb1
    twonth = n1 + n2
    twonb = nb1 + nb2
    dn1 = ((1 + zeta) * nb1 * np.square(np.cos(theta)) + (1 - zeta) * nb2 * np.square(np.sin(theta)) -
           (1 + zeta * np.cos(2 * theta)) * n1)
    dn2 = ((1 - zeta) * nb2 * np.square(np.cos(theta)) + (1 + zeta) * nb1 * np.square(np.sin(theta)) -
           (1 - zeta * np.cos(2 * theta)) * n2)
    dtheta = gr * 0.5 + (zeta * (twonb - twonth) - deltanb) * np.sin(2 * theta) / deltan
    return np.array([dn1, dn2, dtheta])


def rk4slope(zeta, g, nb1, nb2, state, dt):
    # This function uses the RK4 method for the above eom
    k1 = eom(zeta, g, nb1, nb2, state)
    k2 = eom(zeta, g, nb1, nb2, state + 0.5 * k1 * dt)
    k3 = eom(zeta, g, nb1, nb2, state + 0.5 * k2 * dt)
    k4 = eom(zeta, g, nb1, nb2, state + k3 * dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def simulation(zeta, g, nb1, nb2, initial, target, tf):
    # now, we run through the system with our rk4
    # target will be the target precision of a thermal population
    dt = 0.01  # this is basically arbitrary
    t = np.zeros(100000000)  # also basically arbitrary, but needs to be large enough to handle the simulation
    state = np.zeros([3, 100000000])
    state[:, 0] = initial[:]
    i = 0
    while t[i] < tf:  # standard dynamic timestep trick
        test = 30 * dt * target
        # two steps of dt
        k = rk4slope(zeta, g, nb1, nb2, state[:, i], dt)
        check1a = state[:, i] + k * dt
        k = rk4slope(zeta, g, nb1, nb2, check1a, dt)
        check1 = check1a + k * dt
        # one step of 2dt
        k = rk4slope(zeta, g, nb1, nb2, state[:, i], 2 * dt)
        check2 = state[:, i] + 2 * k * dt
        # checking error estimate
        check = np.max(np.absolute(check1[:2] - check2[:2]))  # only care about populations, not theta
        if check < 0.5 * test:
            rho = 2  # limit the maximum the timestep can increase for safety
        else:
            rho = test/check
        if rho > 1:
            t[i + 1] = t[i] + dt
            state[:, i + 1] = check1a
            i += 1
        dt = dt * rho**0.25  # otherwise go for the timestep adjustment

    t = t[:i-1]
    state = state[:, :i - 1]
    return t, state


def detuned_eom(zeta, g, nb1, nb2, detune, state, t):
    # this function does the same as above but with the possibility of a laser detuning from resonance
    # so state must be a 4 parameters. Rather than looking at the phase directly, we will look at the part that does not
    #   evolve linearly with time, which I'm calling sigma here. i.e. phi(t) = sigma(t) - Delta_+ * t
    epsilon = 1e-21  # this is for preventing divergence. Stability requires that when tan(2theta) = 0, sin(phi = 0)
    n1 = state[0]
    n2 = state[1]
    theta = state[2]
    sigma = state[3]
    deltan = n2 - n1
    deltanb = nb2 - nb1
    twonth = n1 + n2
    twonb = nb1 + nb2
    dn1 = ((1 + zeta) * nb1 * np.square(np.cos(theta)) + (1 - zeta) * nb2 * np.square(np.sin(theta)) -
           (1 + zeta * np.cos(2 * theta)) * n1)
    dn2 = ((1 - zeta) * nb2 * np.square(np.cos(theta)) + (1 + zeta) * nb1 * np.square(np.sin(theta)) -
           (1 - zeta * np.cos(2 * theta)) * n2)
    dtheta = g * 0.5 * np.cos(sigma - detune * t) + (zeta * (twonb - twonth) - deltanb) * np.sin(2 * theta) / deltan
    dsigma = 0
    if np.sin(sigma - detune * t) > epsilon:  # catch the problematic cases
        dsigma = 0.5 * g * np.sin(sigma - detune * t) / (np.tan(2 * theta))  # and update the detuning param.
    return np.array([dn1, dn2, dtheta, dsigma])


def detuned_rk4slope(zeta, gamma, nb1, nb2, detuning, state, t, dt):
    k1 = detuned_eom(zeta, gamma, nb1, nb2, detuning, state, t)
    k2 = detuned_eom(zeta, gamma, nb1, nb2, detuning, state + 0.5 * k1 * dt, t + 0.5 * dt)
    k3 = detuned_eom(zeta, gamma, nb1, nb2, detuning, state + 0.5 * k2 * dt, t + 0.5 * dt)
    k4 = detuned_eom(zeta, gamma, nb1, nb2, detuning, state + k3 * dt, t + dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def detuned_simulation(zeta, gamma, nb1, nb2, detuning, initial, target, tf):
    # target will be the target precision of a thermal population
    dt = 0.01  # arb
    t = np.zeros(100000000)  # arb
    state = np.zeros([4, 100000000])
    state[:, 0] = initial[:]
    i = 0
    while t[i] < tf:  # standard dynamic timestep trick
        test = 30 * dt * target
        # two steps of dt
        k = detuned_rk4slope(zeta, gamma, nb1, nb2, detuning, state[:, i], t[i], dt)
        check1a = state[:, i] + k * dt
        k = detuned_rk4slope(zeta, gamma, nb1, nb2, detuning, check1a, t[i] + dt, dt)
        check1 = check1a + k * dt
        # one step of 2dt
        k = detuned_rk4slope(zeta, gamma, nb1, nb2, detuning, state[:, i], t[i], 2*dt)
        check2 = state[:, i] + 2 * k * dt
        # checking error estimate
        check = np.max(np.absolute(check1[:2] - check2[:2]))  # only care about populations
        if check < 0.5 * test:
            rho = 2
        else:
            rho = test/check
        if rho > 1:
            t[i + 1] = t[i] + dt
            state[:, i + 1] = check1a
            # state[2:, i + 1] = state[2:, i + 1] % 2*np.pi - np.pi
            i += 1
        dt = dt * rho**0.25

    t = t[:i-1]
    state = state[:, :i - 1]
    return t, state


def bath(omega, T):
    # Not in use any more
    C = 75.3862e-12  # h/k
    x = np.exp(C*omega/T)
    return 1/(x - 1)

