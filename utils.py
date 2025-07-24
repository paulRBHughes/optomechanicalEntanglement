"""
Project: 2 Mode Thermal Squeezing in Thermal Bath
Purpose: Model ODE equations and make plots to see how things like seperability are effected by thermal baths
May 2024
Paul RB Hughes
Last Modified: June 2025
"""

# Import
import numpy as np
from functools import partial


def thermal_eom(zeta, mode_bath, other_bath, nth, u):
    # Function to calculate the time derivative of the (thermal) population nth
    # zeta is the ratio of gamma- to gamma+, other variables define the state
    # this is general to both modes, but requires an input of -zeta for calculating the second mode
    dn = (1+zeta) * 0.5 * (1 + np.cosh(2*u)) * (mode_bath - nth) + (1-zeta) * 0.5 * (np.cosh(2*u) - 1) * (other_bath + nth + 1)
    return dn


def eom(zeta, g, batha, bathb, state):
    # caluculate the time derivative of the squeezed thermal state parameters
    n1th = state[0]
    n2th = state[1]
    u = state[2]
    dn1th = thermal_eom(zeta, batha, bathb, n1th, u)
    dn2th = thermal_eom(-zeta, bathb, batha, n2th, u)
    du = 0.5*(g - np.sinh(2*u) * (batha + bathb + 1 + zeta*(batha - n1th + n2th - bathb))/(n1th + n2th + 1))
    return np.array([dn1th, dn2th, du])


def rel_eom(zeta, g, av_bath, diff_bath, state):
    # same as above but uses the average and difference numbers
    av = state[0]
    diff = state[1]
    u = state[2]
    dav = 0.5 * ( ((2*av_bath + 1) + zeta*(diff - diff_bath))*np.cosh(2*u) - 2*av - 1)
    ddiff = diff_bath - diff - zeta*(2*av_bath + 1 - np.cosh(2*u) * (2*av + 1))
    du = 0.5 * (g - np.sinh(2*u)*(2*av_bath + 1 + zeta*(diff - diff_bath))/(2*av + 1))
    return np.array([dav, ddiff, du])


def rk4slope(zeta, g, batha, bathb, state, dt):
    # this is the rk4 algorithm applied to these equations
    k1 = eom(zeta, g, batha, bathb, state)
    k2 = eom(zeta, g, batha, bathb, state + 0.5 * k1 * dt)
    k3 = eom(zeta, g, batha, bathb, state + 0.5 * k2 * dt)
    k4 = eom(zeta, g, batha, bathb, state + k3 * dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def rel_rk4slope(zeta, g, avb, db, state, dt):
    k1 = rel_eom(zeta, g, avb, db, state)
    k2 = rel_eom(zeta, g, avb, db, state + 0.5 * k1 * dt)
    k3 = rel_eom(zeta, g, avb, db, state + 0.5 * k2 * dt)
    k4 = rel_eom(zeta, g, avb, db, state + k3 * dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def simulation(zeta, g, bath1, bath2, initial, target, tf):
    # This function runs the time evolution of the 2MSTS using the rk4 function
    # target will be the target precision of all three params. Could modify to target precision in just thermal pop,
    #   but perturbations in u can cause larger changes fast
    # tf is the end time to run the evolution
    dt = 0.01  # initializing as a reasonably small timestep
    t = np.zeros(100000000)  # initializing as a large vector to store times and state. 100M is fine for most runs
    state = np.zeros([3, 100000000])
    state[:, 0] = initial[:]  # Set the IC
    i = 0
    while t[i] < tf:  # standard dynamic timestep trick for the rk4
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
        check = np.max(np.absolute(check1[:2] - check2[:2]))
        if check < 0.5 * test:
            rho = 2  # set a maximum limit on how much larger the step size gets.
        else:
            rho = test/check  # if the step size is too large, then we reduce based on the error and target
        if rho > 1:
            t[i + 1] = t[i] + dt
            state[:, i + 1] = check1a
            i += 1
        dt = dt * rho**0.25  # adjust step size

    t = t[:i-1]  # cut of vestigial t
    state = state[:, :i - 1]  # and state
    return t, state


def rel_simulation(zeta, g, avb, db, initial, corrmax, target, tf):
    # similar to above with the relative EOM, but now we have a corrmax, a limit on the correlation variance before we stop evolution
    # This lets us pump above threshold and not grind the sim to a halt. Still problems if e.g. the population is too large
    #   and CV is supressed by u, but otherwise this allows us to see the behaviour of transients
    dt = 0.01
    t = np.zeros(100000000)
    state = np.zeros([3, 100000000])
    state[:, 0] = initial[:]
    i = 0
    startsbelow = (2*state[0, 0] + 1)*np.exp(-2*state[2, 0]) < corrmax  # check if CV begins below set maximum
    while t[i] < tf:
        # now I want to check if the CV is above my set threshold. If it started above it though, we need to be okay with that
        isabove = (2*state[0, i] + 1)*np.exp(-2*state[2, i]) > corrmax
        if startsbelow:  # if we were below
            if isabove:  # and now are above,
                break    # then we can stop the evolution here
        else:  # if we started above the CV,
            if not isabove:  # but now are below,
                startsbelow = True  # then we set ourselves as below


        test = 30 * dt * target
        # two steps of dt
        k = rel_rk4slope(zeta, g, avb, db, state[:, i], dt)
        check1a = state[:, i] + k * dt
        k = rel_rk4slope(zeta, g, avb, db, check1a, dt)
        check1 = check1a + k * dt
        # one step of 2dt
        k = rel_rk4slope(zeta, g, avb, db, state[:, i], 2*dt)
        check2 = state[:, i] + 2 * k * dt
        # checking error estimate
        check = np.absolute(check1[2] - check2[2])  # now, I'm actually only looking at the error in u
        if check < 0.5 * test:
            rho = 2
        else:
            rho = test/check
        if rho > 1:
            t[i + 1] = t[i] + dt
            state[:, i + 1] = check1a
            i += 1
        dt = dt * rho**0.25


    t = t[:i-1]
    state = state[:, :i - 1]
    return t, state



def pump_rk4slope(zeta, g, avb, db, state, t, dt):
    # Expanded equations for a time varying pump function g.
    k1 = rel_eom(zeta, g(t), avb, db, state)
    k2 = rel_eom(zeta, g(t + 0.5 * dt), avb, db, state + 0.5 * k1 * dt)
    k3 = rel_eom(zeta, g(t + 0.5 * dt), avb, db, state + 0.5 * k2 * dt)
    k4 = rel_eom(zeta, g(t + dt), avb, db, state + k3 * dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def pump_simulation(zeta, g_func, avb, db, initial, target, tf, *args):
    # this function runs the 2MSTS evolution according to the rk4 above, with pumping (not used in paper)
    g = partial(g_func, *args)
    dt = 0.01  # arb
    t = np.zeros(100000000)  # arb
    state = np.zeros([3, 100000000])
    state[:, 0] = initial[:]
    i = 0
    while t[i] < tf:  # standard dynamic timestep trick
        test = 30 * dt * target
        # two steps of dt
        k = pump_rk4slope(zeta, g, avb, db, state[:, i], t[i], dt)
        check1a = state[:, i] + k * dt
        k = pump_rk4slope(zeta, g, avb, db, check1a, t[i] + dt, dt)
        check1 = check1a + k * dt
        # one step of 2dt
        k = pump_rk4slope(zeta, g, avb, db, state[:, i], t[i], 2*dt)
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
    # quick utility to calculate the CV for 2MSTS. State follows the same form as the rest of the code
    return (1 + state[0, :] + state[1, :]) * np.exp(-2 * state[2, :])


def rel_corr_var(state):
    # same as above but using other EOM form
    return (2*state[0, :] + 1)*np.exp(-2*state[2, :])

