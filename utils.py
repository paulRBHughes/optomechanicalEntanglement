"""
Project: 2 Mode Thermal Squeezing in Thermal Bath
Purpose: Model ODE equations and make plots to see how things like seperability are effected by thermal baths
May 2024
Paul RB Hughes
"""

# Import
import numpy as np
from numba import njit

@njit
def thermal_eom(zeta, mode_bath, other_bath, nth, u):
    # Function to calculate the time derivative of the (thermal) population nth
    # zeta is the ratio of gamma- to gamma+, other variables define the state
    # BECAUSE gamma- is not symmetric, when calculating n2 use -zeta!!!!!!!!!!!!!!!!
    dn = (1+zeta) * np.cosh(u)**2 * (mode_bath - nth) + (1-zeta) * np.sinh(u)**2 * (other_bath + nth + 1)
    return dn


def eom(zeta, g, batha, bathb, state):
    # caluculate the time derivative of the squeeze parameter amplitude
    # a must correspond to the mode that has positive gamma-, i.e. n1 unless I change something
    na = state[0]
    nb = state[1]
    u = state[2]
    dn1 = thermal_eom(zeta, batha, bathb, na, u)
    dn2 = thermal_eom(-zeta, bathb, batha, nb, u)
    du = 0.5*g - np.cosh(u) * np.sinh(u) * (batha + bathb + 1 + zeta*(batha - na + nb - bathb))/(na + nb + 1)
    return np.array([dn1, dn2, du])


def rk4slope(zeta, g, batha, bathb, state, dt):
    k1 = eom(zeta, g, batha, bathb, state)
    k2 = eom(zeta, g, batha, bathb, state + 0.5 * k1 * dt)
    k3 = eom(zeta, g, batha, bathb, state + 0.5 * k2 * dt)
    k4 = eom(zeta, g, batha, bathb, state + k3 * dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def simulation(zeta, g, bath1, bath2, initial, target, tf):
    # target will be the target precision of a thermal population
    dt = 0.01  # arb
    t = np.zeros(100000000)  # arb
    state = np.zeros([3, 100000000])
    state[:, 0] = initial[:]
    i = 0
    while t[i] < tf:  # standard dynamic timestep trick
        test = 30 * dt * target
        # two steps of dt
        k = rk4slope(zeta, g, bath1, bath2, state[:, i], dt)
        check1a = state[:, i] + k * dt
        k = rk4slope(zeta, g, bath1, bath2, check1a, dt)
        check1 = check1a + k * dt
        # one step of 2dt
        k = rk4slope(zeta, g, bath1, bath2, state[:, i], 2*dt)
        check2 = state[:, i] + 2 * k * dt
        # checking error estimate
        check = np.max(np.absolute(check1[:2] - check2[:2]))  # only care about populations
        if check < 0.5 * test:
            rho = 2
        else:
            rho = test/check  # taking faith in old code
        if rho > 1:
            t[i + 1] = t[i] + dt
            state[:, i + 1] = check1a
            i += 1
        dt = dt * rho**0.25

    t = t[:i-1]
    state = state[:, :i - 1]
    return t, state


def correlation_var(state):
    return (1 + state[0,:] + state[1,:]) * np.exp(-2 * state[2,:])




