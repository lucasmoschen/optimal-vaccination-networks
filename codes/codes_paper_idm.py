#!/usr/bin/env python

import pandas as pd
import numpy as np
import gekko as gk
from scipy.integrate import solve_ivp
from time import time
import json

def model_ode(t, y, u, alpha, beta, gamma, mu, K, p_matrix, population, population_eff):
    """
    Defines a system of ordinary differential equations (ODEs) for a compartmental model in epidemiology.

    Parameters:
    t (float): The current time.
    y (array-like): The current state of the system. Should be a 1D array of length 4*K, 
                    where the first K elements represent the susceptible individuals, 
                    the next K elements represent the exposed individuals, 
                    the next K elements represent the infectious individuals, 
                    and the last K elements represent the recovered individuals.
    u (array-like): The vaccinatio rate for each city.
                    Should be a 1D array of length K.
    alpha (float): Proportion of the night among the 24 hours.
    beta (array-like): The transmission rate for each city. 
                       Should be a 1D array of length K.
    gamma (float): The recovery rate (the inverse of the infectious period).
    mu (float): The birth and natural death rate. 
    K (int): The number of cities or groups in the population.
    p_matrix (array-like): A matrix that represents the transitions between different cities. 
                           Should be a 2D array of shape (K, K).
    population (array-like): The total population in each city. 
                             Should be a 1D array of length K.
    population_eff (array-like): The effective population during the day.
                                 Should be a 1D array of length K.
                                 P_i^{\mathrm{eff}} = \sum_{j=1}^K p_matrix{ji} population_j
                                 
    Returns:
    array-like: The rates of change of the susceptible, exposed, infectious, and recovered individuals. 
                Returns a 1D array of length 4*K.
    """
    S, I, R, _ = y[:K], y[K:2*K], y[2*K:3*K], y[3*K:]
    I_eff = (I * population) @ p_matrix / population_eff
    S_dot = -alpha * beta * S * I - (1-alpha) * S * (p_matrix @ (beta * I_eff))
    I_dot = -S_dot - (gamma + mu) * I
    R_dot = u * S + gamma * I - mu * R
    H_dot = -S_dot
    S_dot += mu - (u + mu) * S
    
    return np.hstack([S_dot, I_dot, R_dot, H_dot])

def optimal_vaccination_strategy(parameters):
    
    # Gekko object
    remote=True
    if 'remote' in parameters:
        remote = parameters['remote']
    m = gk.GEKKO(remote=remote)
    
    # Including the time
    T = parameters['T']
    npd = 20
    if 'npd' in parameters:
        npd = parameters['npd']
    n_points = npd*T + 1
    m.time = np.linspace(0, T, n_points)

    # Number of cities
    populations = parameters['populations']/parameters['populations'].sum()
    K = len(populations)

    # Parameters
    beta = parameters['beta']
    cv = m.Const(parameters['cv'])
    ch = m.Const(parameters['ch'])
    rh = m.Const(parameters['rh'])
    alpha = m.Const(parameters['alpha'])
    gamma = m.Const(parameters['gamma'])
    mu = m.Const(parameters['mu'])
    p_matrix = parameters['p_matrix']
    population_eff = populations @ p_matrix

    # Initial values
    i0 = parameters['i0']
    r0 = parameters['r0']
    s0 = np.ones(K) - i0 - r0
    
    # Constraints
    week_max = parameters['week_max']
    rate_max = parameters['rate_max']

    # Including the variables
    s = m.Array(m.Var, K, lb=0.0)
    i = m.Array(m.Var, K, lb=0.0)
    v = m.Array(m.Var, K, lb=0.0, value=0.0)
    h = m.Array(m.Var, K, lb=0.0, value=0.0)
    counting = m.Array(m.Var, K, lb=0.0, value=0.0)

    w = [m.MV(lb=0.0, ub=rate_max[city], value=rate_max[city]/2, name='u{}'.format(city)) for city in range(K)]
    for city in range(K):
        s[city].value = s0[city]
        i[city].value = i0[city]
        
        w[city].STATUS = 1
        w[city].DCOST = 0

    # Maximizing 
    obj_points = np.zeros(n_points)
    obj_points[-1] = 1.0
    final = m.Param(value=obj_points)
    m.Obj(cv * sum(v*populations) * final + ch * sum(h*populations) * final)

    # ODE equations
    i_eff = [m.Intermediate(sum(p_matrix[:,city]*populations*i)/population_eff[city]) for city in range(K)]
    for city in range(K):
        new_infections = s[city] * (alpha*beta[city]*i[city] + (1-alpha)*sum(beta*p_matrix[city,:]*i_eff))
        m.Equation(s[city].dt() == -new_infections - w[city] + mu - mu*s[city])
        m.Equation(i[city].dt() == new_infections - (gamma + mu)*i[city])
        m.Equation(v[city].dt() == w[city]) #no need to account for deaths
        m.Equation(h[city] == rh*m.integral(i[city]))
        m.Equation(counting[city] == m.integral(new_infections))

    for week in range(T//7):
        weekly_mark = np.zeros(n_points)
        weekly_mark[npd*7*(week+1)] = 1.0
        weekly_constraint = m.Param(value=weekly_mark)
        m.Equation( sum(v * populations) * weekly_constraint <= week_max[week])

    # Solving
    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.MV_TYPE = 1
    m.options.SOLVER = 3  # Using IPOPT solver
    m.options.MAX_ITER = 500
    m.options.DIAGLEVEL = 1
    
    m.options.RTOL=parameters['RTOL']
    m.options.OTOL=parameters['OTOL']

    m.solve(disp=True)
    
    return m, s, i, v, h, w, counting

def optimal_vaccination_strategy_optimized(parameters):
    
    # Gekko object
    remote=True
    if 'remote' in parameters:
        remote = parameters['remote']
    m = gk.GEKKO(remote=remote)
    
    # Including the time
    T = parameters['T']
    npd = 20
    if 'npd' in parameters:
        npd = parameters['npd']
    n_points = npd*T + 1
    m.time = np.linspace(0, T, n_points)

    # Number of cities
    populations = parameters['populations']/parameters['populations'].sum()
    K = len(populations)

    # Parameters
    beta = parameters['beta']
    cv = m.Const(parameters['cv'])
    ch = m.Const(parameters['ch'])
    rh = m.Const(parameters['rh'])
    alpha = m.Const(parameters['alpha'])
    gamma = m.Const(parameters['gamma'])
    mu = m.Const(parameters['mu'])
    p_matrix = parameters['p_matrix']
    population_eff = populations @ p_matrix

    # Initial values
    i0 = parameters['i0']
    r0 = parameters['r0']
    s0 = np.ones(K) - i0 - r0
    
    # Constraints
    week_max = parameters['week_max']
    rate_max = parameters['rate_max']

    # Including the variables
    s = m.Array(m.Var, K, lb=0.0)
    i = m.Array(m.Var, K, lb=0.0)
    w = [m.MV(lb=0.0, ub=rate_max[city], value=0.0, name='uS{}'.format(city)) for city in range(K)]
    #v = m.Var(lb=0.0, value=0.0)
    #h = m.Var(lb=0.0, value=0.0)
    v = m.Intermediate(m.integral(sum(w*populations)))
    h = m.Intermediate(rh*m.integral(sum(i*populations)))
    for city in range(K):
        s[city].value = s0[city]
        i[city].value = i0[city]        
        w[city].STATUS = 1
        w[city].DCOST = 0

    # Maximizing 
    obj_points = np.zeros(n_points)
    obj_points[-1] = 1.0
    final = m.Param(value=obj_points)
    m.Obj((cv * v + ch * h) * final)

    # ODE equations
    i_eff = [m.Intermediate(sum(p_matrix[:,city]*populations*i)/population_eff[city]) for city in range(K)]
    new_infections = [m.Intermediate(alpha*beta[city]*i[city] + (1-alpha)*sum(beta*p_matrix[city,:]*i_eff))
                     for city in range(K)]
    for city in range(K):
        m.Equation(s[city].dt() == -s[city]*new_infections[city] - w[city] + mu - mu*s[city])
        m.Equation(i[city].dt() == s[city]*new_infections[city] - (gamma + mu)*i[city])
    #m.Equation(v.dt() == sum(w*populations))
    #m.Equation(h.dt() == rh*sum(i*populations))

    for week in range(T//7):
        weekly_mark = np.zeros(n_points)
        weekly_mark[npd*7*(week+1)] = 1.0
        weekly_constraint = m.Param(value=weekly_mark)
        m.Equation( v * weekly_constraint <= week_max[week])

    # Solving
    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.MV_TYPE = 1
    m.options.SOLVER = 3  # Using IPOPT solver
    m.options.MAX_ITER = 500
    m.options.DIAGLEVEL = 1
    
    m.options.RTOL=parameters['RTOL']
    m.options.OTOL=parameters['OTOL']

    m.solve(disp=True)
    
    return m, s, i, v, h, w

def optimal_vaccination_strategy_uniform(parameters):
    
    # Gekko object
    remote=True
    if 'remote' in parameters:
        remote = parameters['remote']
    m = gk.GEKKO(remote=remote)
    
    # Including the time
    T = parameters['T']
    npd = 20
    if 'npd' in parameters:
        npd = parameters['npd']
    n_points = npd*T + 1
    m.time = np.linspace(0, T, n_points)

    # Number of cities
    populations = parameters['populations']/parameters['populations'].sum()
    K = len(populations)

    # Parameters
    beta = parameters['beta']
    cv = m.Const(parameters['cv'])
    ch = m.Const(parameters['ch'])
    rh = m.Const(parameters['rh'])
    alpha = m.Const(parameters['alpha'])
    gamma = m.Const(parameters['gamma'])
    mu = m.Const(parameters['mu'])
    p_matrix = parameters['p_matrix']
    population_eff = populations @ p_matrix

    # Initial values
    i0 = parameters['i0']
    r0 = parameters['r0']
    s0 = np.ones(K) - i0 - r0
    
    # Constraints
    week_max = parameters['week_max']
    rate_max = parameters['rate_max']

    # Including the variables
    s = m.Array(m.Var, K, lb=0.0)
    i = m.Array(m.Var, K, lb=0.0)
    u = m.MV(lb=0.0, value=0.0)
    u.STATUS = 1 
    u.DCOST = 0
    v = m.Intermediate(m.integral(u*m.sum(s*populations)))
    h = m.Intermediate(rh*m.integral(sum(i*populations)))
    for city in range(K):
        s[city].value = s0[city]
        i[city].value = i0[city]        

    # Maximizing 
    obj_points = np.zeros(n_points)
    obj_points[-1] = 1.0
    final = m.Param(value=obj_points)
    m.Obj((cv * v + ch * h) * final)

    # ODE equations
    i_eff = [m.Intermediate(sum(p_matrix[:,city]*populations*i)/population_eff[city]) for city in range(K)]
    new_infections = [m.Intermediate(alpha*beta[city]*i[city] + (1-alpha)*sum(beta*p_matrix[city,:]*i_eff))
                     for city in range(K)]
    for city in range(K):
        m.Equation(s[city].dt() == -s[city]*new_infections[city] - u * s[city] + mu - mu*s[city])
        m.Equation(i[city].dt() == s[city]*new_infections[city] - (gamma + mu)*i[city])
        m.Equation(u*s[city] <= rate_max[city])

    for week in range(T//7):
        weekly_mark = np.zeros(n_points)
        weekly_mark[npd*7*(week+1)] = 1.0
        weekly_constraint = m.Param(value=weekly_mark)
        m.Equation( v * weekly_constraint <= week_max[week])

    # Solving
    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.MV_TYPE = 1
    m.options.SOLVER = 3  # Using IPOPT solver
    m.options.MAX_ITER = 500
    m.options.DIAGLEVEL = 1
    
    m.options.RTOL=parameters['RTOL']
    m.options.OTOL=parameters['OTOL']

    m.solve(disp=True)
    
    return m, s, i, v, h, u

def optimal_vaccination_strategy_metropolitan(parameters):
    
    # Gekko object
    remote=True
    if 'remote' in parameters:
        remote = parameters['remote']
    m = gk.GEKKO(remote=remote)
    
    # Including the time
    T = parameters['T']
    npd = 20
    if 'npd' in parameters:
        npd = parameters['npd']
    n_points = npd*T + 1
    m.time = np.linspace(0, T, n_points)

    # Number of cities
    populations = parameters['populations']/parameters['populations'].sum()
    K = len(populations)

    # Parameters
    beta = parameters['beta']
    cv = m.Const(parameters['cv'])
    ch = m.Const(parameters['ch'])
    rh = m.Const(parameters['rh'])
    alpha = m.Const(parameters['alpha'])
    gamma = m.Const(parameters['gamma'])
    mu = m.Const(parameters['mu'])
    p_matrix = parameters['p_matrix']
    population_eff = populations @ p_matrix

    # Initial values
    i0 = parameters['i0']
    r0 = parameters['r0']
    s0 = np.ones(K) - i0 - r0
    
    # Constraints
    week_max = parameters['week_max']
    rate_max = parameters['rate_max']

    # Including the variables
    s = m.Array(m.Var, K, lb=0.0)
    i = m.Array(m.Var, K, lb=0.0)
    v = m.Var(value=0.0, lb=0.0)
    h = m.Intermediate(rh*m.integral(sum(i*populations)))
    w = [m.MV(lb=0.0, ub=rate_max[city], value=rate_max[city]/2, name='u{}'.format(city)) for city in range(K)]
    for city in range(K):
        s[city].value = s0[city]
        i[city].value = i0[city]        
        w[city].STATUS = 1
        w[city].DCOST = 0

    # Maximizing 
    obj_points = np.zeros(n_points)
    obj_points[-1] = 1.0
    final = m.Param(value=obj_points)
    m.Obj((cv * v  + ch * h) * final)

    # ODE equations
    i_eff_0 = m.Intermediate(sum(p_matrix[:,0]*populations*i)/population_eff[0])
    for city in range(K):
        if city == 0:
            new_infections = s[0] * beta[0] * (alpha*i[0] + (1-alpha)*i_eff_0)
        else:
            new_infections = s[city] * (beta[city]*i[city]*(alpha + (1-alpha)*p_matrix[city,city]) + (1-alpha)*beta[0]*p_matrix[city,0]*i_eff_0)
        m.Equation(s[city].dt() == -new_infections - w[city] + mu - mu*s[city])
        m.Equation(i[city].dt() == new_infections - (gamma + mu)*i[city])
    m.Equation(v.dt() == sum(w*populations))

    for week in range(T//7):
        weekly_mark = np.zeros(n_points)
        weekly_mark[npd*7*(week+1)] = 1.0
        weekly_constraint = m.Param(value=weekly_mark)
        m.Equation( v * weekly_constraint <= week_max[week])

    # Solving
    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.MV_TYPE = 1
    m.options.SOLVER = 3  # Using IPOPT solver
    m.options.MAX_ITER = 500
    m.options.DIAGLEVEL = 1
    
    m.options.RTOL=parameters['RTOL']
    m.options.OTOL=parameters['OTOL']

    m.solve(disp=True)
    
    return m, s, i, v, h, w

def constant_vaccination(parameters):
    
    # Gekko object
    remote=True
    if 'remote' in parameters:
        remote = parameters['remote']
    m = gk.GEKKO(remote=remote)
    
    # Including the time
    T = parameters['T']
    npd = 20
    if 'npd' in parameters:
        npd = parameters['npd']
    n_points = npd*T + 1
    m.time = np.linspace(0, T, n_points)

    # Number of cities
    populations = parameters['populations']/parameters['populations'].sum()
    K = len(populations)

    # Parameters
    beta = parameters['beta']
    cv = m.Const(parameters['cv'])
    ch = m.Const(parameters['ch'])
    rh = m.Const(parameters['rh'])
    alpha = m.Const(parameters['alpha'])
    gamma = m.Const(parameters['gamma'])
    mu = m.Const(parameters['mu'])
    p_matrix = parameters['p_matrix']
    population_eff = populations @ p_matrix

    # Initial values
    i0 = parameters['i0']
    r0 = parameters['r0']
    s0 = np.ones(K) - i0 - r0
    
    # Constraints
    week_max = parameters['week_max']
    rate_max = parameters['rate_max']

    # Including the variables
    s = m.Array(m.Var, K, lb=0.0)
    i = m.Array(m.Var, K, lb=0.0)
    v = m.Array(m.Var, K, lb=0.0, value=0.0)
    h = m.Array(m.Var, K, lb=0.0, value=0.0)
    counting = m.Array(m.Var, K, lb=0.0, value=0.0)

    u = m.FV(lb=0.0, value=0.0)
    u.STATUS = 1
    for city in range(K):
        s[city].value = s0[city]
        i[city].value = i0[city]
        
    # Maximizing 
    obj_points = np.zeros(n_points)
    obj_points[-1] = 1.0
    final = m.Param(value=obj_points)
    m.Obj(cv * sum(v*populations) * final + ch * sum(h*populations) * final)

    # ODE equations
    i_eff = [m.Intermediate(sum(p_matrix[:,city]*populations*i)/population_eff[city]) for city in range(K)]
    for city in range(K):
        new_infections = s[city] * (alpha*beta[city]*i[city] + (1-alpha)*sum(beta*p_matrix[city,:]*i_eff))
        m.Equation(s[city].dt() == -new_infections + mu - (u + mu)*s[city])
        m.Equation(i[city].dt() == new_infections - (gamma + mu)*i[city])
        m.Equation(v[city].dt() == u*s[city])
        m.Equation(h[city] == rh*m.integral(i[city]))
        m.Equation(counting[city] == m.integral(new_infections))        
        m.Equation(u*s[city] <= rate_max[city])        
                
    for week in range(T//7):
        weekly_mark = np.zeros(n_points)
        weekly_mark[npd*7*(week+1)] = 1.0
        weekly_constraint = m.Param(value=weekly_mark)
        m.Equation( sum(v * populations) * weekly_constraint <= week_max[week])

    # Solving
    m.options.IMODE = 5
    m.options.NODES = 2
    m.options.SOLVER = 3
    m.options.MAX_ITER = 500 
        
    m.options.RTOL=parameters['RTOL']
    m.options.OTOL=parameters['OTOL']

    m.solve(disp=True)
    
    return m, s, i, v, h, u, counting

def constant_vaccination_optimized(parameters):
    
    # Gekko object
    remote=True
    if 'remote' in parameters:
        remote = parameters['remote']
    m = gk.GEKKO(remote=remote)
    
    # Including the time
    T = parameters['T']
    npd = 20
    if 'npd' in parameters:
        npd = parameters['npd']
    n_points = npd*T + 1
    m.time = np.linspace(0, T, n_points)

    # Number of cities
    populations = parameters['populations']/parameters['populations'].sum()
    K = len(populations)

    # Parameters
    beta = parameters['beta']
    cv = m.Const(parameters['cv'])
    ch = m.Const(parameters['ch'])
    rh = m.Const(parameters['rh'])
    alpha = m.Const(parameters['alpha'])
    gamma = m.Const(parameters['gamma'])
    mu = m.Const(parameters['mu'])
    p_matrix = parameters['p_matrix']
    population_eff = populations @ p_matrix

    # Initial values
    i0 = parameters['i0']
    r0 = parameters['r0']
    s0 = np.ones(K) - i0 - r0
    
    # Constraints
    week_max = parameters['week_max']
    rate_max = parameters['rate_max']

    # Including the variables
    s = m.Array(m.Var, K, lb=0.0)
    i = m.Array(m.Var, K, lb=0.0)
    v = m.Var(lb=0.0, value=0.0)
    h = m.Var(lb=0.0, value=0.0)
    u = m.FV(lb=0.0, value=0.0)
    u.STATUS = 1
    for city in range(K):
        s[city].value = s0[city]
        i[city].value = i0[city]
        
    # Maximizing 
    obj_points = np.zeros(n_points)
    obj_points[-1] = 1.0
    final = m.Param(value=obj_points)
    m.Obj(cv * v * final + ch * h * final)

    # ODE equations
    i_eff = [m.Intermediate(sum(p_matrix[:,city]*populations*i)/population_eff[city]) for city in range(K)]
    for city in range(K):
        new_infections = s[city] * (alpha*beta[city]*i[city] + (1-alpha)*sum(beta*p_matrix[city,:]*i_eff))
        m.Equation(s[city].dt() == -new_infections + mu - (u + mu)*s[city])
        m.Equation(i[city].dt() == new_infections - (gamma + mu)*i[city])
        m.Equation(u*s[city] <= rate_max[city])
    m.Equation(v.dt() == u*sum(s*populations))
    m.Equation(h.dt() == rh*sum(i*populations))
                
    for week in range(T//7):
        weekly_mark = np.zeros(n_points)
        weekly_mark[npd*7*(week+1)] = 1.0
        weekly_constraint = m.Param(value=weekly_mark)
        m.Equation(v * weekly_constraint <= week_max[week])

    # Solving
    m.options.IMODE = 5
    m.options.NODES = 2
    m.options.SOLVER = 3
    m.options.MAX_ITER = 500 
        
    m.options.RTOL=parameters['RTOL']
    m.options.OTOL=parameters['OTOL']

    m.solve(disp=True)
    
    return m, s, i, v, h, u

def optimal_vaccination_strategy_feedback(parameters):
    
    # Gekko object
    remote=True
    if 'remote' in parameters:
        remote = parameters['remote']
    m = gk.GEKKO(remote=remote)
    
    # Including the time
    T = parameters['T']
    npd = 20
    if 'npd' in parameters:
        npd = parameters['npd']
    n_points = npd*T + 1
    m.time = np.linspace(0, T, n_points)

    # Number of cities
    populations = parameters['populations']/parameters['populations'].sum()
    K = len(populations)

    # Parameters
    beta = parameters['beta']
    cv = m.Const(parameters['cv'])
    ch = m.Const(parameters['ch'])
    rh = m.Const(parameters['rh'])
    alpha = m.Const(parameters['alpha'])
    gamma = m.Const(parameters['gamma'])
    mu = m.Const(parameters['mu'])
    p_matrix = parameters['p_matrix']
    population_eff = populations @ p_matrix

    # Initial values
    i0 = parameters['i0']
    r0 = parameters['r0']
    s0 = np.ones(K) - i0 - r0
    
    # Constraints
    week_max = parameters['week_max']
    rate_max = parameters['rate_max']

    # Including the variables
    s = m.Array(m.Var, K, lb=0.0)
    i = m.Array(m.Var, K, lb=0.0)
    v = m.Array(m.Var, K, lb=0.0, value=0.0)
    h = m.Array(m.Var, K, lb=0.0, value=0.0)

    u = [m.MV(lb=0.0, value=0.01, name='u{}'.format(city)) for city in range(K)]
    for city in range(K):
        s[city].value = s0[city]
        i[city].value = i0[city]
        
        u[city].STATUS = 1
        u[city].DCOST = 0

    # Maximizing 
    obj_points = np.zeros(n_points)
    obj_points[-1] = 1.0
    final = m.Param(value=obj_points)
    m.Obj(cv * sum(v*populations) * final + ch * sum(h*populations) * final)

    # ODE equations
    i_eff = [m.Intermediate(sum(p_matrix[:,city]*populations*i)/population_eff[city]) for city in range(K)]
    for city in range(K):
        new_infections = s[city] * (alpha*beta[city]*i[city] + (1-alpha)*sum(beta*p_matrix[city,:]*i_eff))
        m.Equation(s[city].dt() == -new_infections - u[city]*s[city] + mu - mu*s[city])
        m.Equation(i[city].dt() == new_infections - (gamma + mu)*i[city])
        m.Equation(v[city].dt() == u[city]*s[city])
        m.Equation(h[city] == rh*m.integral(i[city]))
        m.Equation(u[city]*(1-v[city]) <= rate_max[city])

    for week in range(T//7):
        weekly_mark = np.zeros(n_points)
        weekly_mark[npd*7*(week+1)] = 1.0
        weekly_constraint = m.Param(value=weekly_mark)
        m.Equation( sum(v * populations) * weekly_constraint <= week_max[week])

    # Solving
    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.MV_TYPE = 1
    m.options.SOLVER = 3  # Using IPOPT solver
    m.options.MAX_ITER = 500
    m.options.DIAGLEVEL = 1
    
    m.options.RTOL=parameters['RTOL']
    m.options.OTOL=parameters['OTOL']

    m.solve(disp=True)
    
    return m, s, i, v, h, u

def optimal_vaccination_strategy_alpha_time(parameters):
    
    # Gekko object
    remote=True
    if 'remote' in parameters:
        remote = parameters['remote']
    m = gk.GEKKO(remote=remote)
    
    # Including the time
    T = parameters['T']
    npd = 20
    if 'npd' in parameters:
        npd = parameters['npd']
    n_points = npd*T + 1
    m.time = np.linspace(0, T, n_points)

    # Number of cities
    populations = parameters['populations']/parameters['populations'].sum()
    K = len(populations)

    # Parameters
    beta = parameters['beta']
    cv = m.Const(parameters['cv'])
    ch = m.Const(parameters['ch'])
    rh = m.Const(parameters['rh'])
    alpha_t = np.array([parameters['alpha'](ti) for ti in m.time])
    alpha = m.Param(value=alpha_t)
    gamma = m.Const(parameters['gamma'])
    mu = m.Const(parameters['mu'])
    p_matrix = parameters['p_matrix']
    population_eff = populations @ p_matrix

    # Initial values
    i0 = parameters['i0']
    r0 = parameters['r0']
    s0 = np.ones(K) - i0 - r0
    
    # Constraints
    week_max = parameters['week_max']
    rate_max = parameters['rate_max']

    # Including the variables
    s = m.Array(m.Var, K, lb=0.0)
    i = m.Array(m.Var, K, lb=0.0)
    w = [m.MV(lb=0.0, ub=rate_max[city], value=0.0, name='uS{}'.format(city)) for city in range(K)]
    v = m.Intermediate(m.integral(sum(w*populations)))
    h = m.Intermediate(rh*m.integral(sum(i*populations)))
    for city in range(K):
        s[city].value = s0[city]
        i[city].value = i0[city]        
        w[city].STATUS = 1
        w[city].DCOST = 0

    # Maximizing 
    obj_points = np.zeros(n_points)
    obj_points[-1] = 1.0
    final = m.Param(value=obj_points)
    m.Obj((cv * v + ch * h) * final)

    # ODE equations
    i_eff = [m.Intermediate(sum(p_matrix[:,city]*populations*i)/population_eff[city]) for city in range(K)]
    new_infections = [m.Intermediate(alpha*beta[city]*i[city] + (1-alpha)*sum(beta*p_matrix[city,:]*i_eff))
                     for city in range(K)]
    for city in range(K):
        m.Equation(s[city].dt() == -s[city]*new_infections[city] - w[city] + mu - mu*s[city])
        m.Equation(i[city].dt() == s[city]*new_infections[city] - (gamma + mu)*i[city])

    for week in range(T//7):
        weekly_mark = np.zeros(n_points)
        weekly_mark[npd*7*(week+1)] = 1.0
        weekly_constraint = m.Param(value=weekly_mark)
        m.Equation( v * weekly_constraint <= week_max[week])

    # Solving
    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.MV_TYPE = 1
    m.options.SOLVER = 3  # Using IPOPT solver
    m.options.MAX_ITER = 500
    m.options.DIAGLEVEL = 1
    
    m.options.RTOL=parameters['RTOL']
    m.options.OTOL=parameters['OTOL']

    m.solve(disp=True)
    
    return m, s, i, v, h, w

def gk2dict(s, i, v, h, w, new_infections, K):
    data = {}
    data['susceptible'] = [list(np.array(s[city])) for city in range(K)]
    data['infected'] = [list(np.array(i[city])) for city in range(K)]
    data['vaccinated'] = [list(np.array(v[city])) for city in range(K)]
    data['hospitalized'] = [list(np.array(h[city])) for city in range(K)]
    data['new_infections'] = [list(np.array(new_infections[city])) for city in range(K)]
    data['w'] = [list(np.array(w[city])) for city in range(K)]
    return data

def exp0():
    # Experiment to track the time considering the faster optimizer.
    # Setting the parameters
    K = 2

    populations = np.array([1e6, 1e5])
    p_matrix = np.array([[1,0],[0.2, 0.8]])
    population_eff = populations @ p_matrix

    beta = np.array([0.25, 0.18])
    alpha = 0.64
    gamma = 1/7
    mu = 3.6e-5

    T = 42  

    rate_max = np.array([0.8, 0.8])/T
    week_max = np.array([1, 2, 3, 4, 5, 6, 7, 8])/13

    # Initial condition
    y0 = np.zeros(4*K)
    y0[K] = 1/populations[0]
    y0[0:K] = 1-y0[K:2*K]

    n_days = 180
    t_eval = np.linspace(0, n_days, 20 * n_days)

    args = (np.zeros(2), alpha, beta, gamma, mu, K, p_matrix, populations, population_eff)

    sol = solve_ivp(fun=lambda t, y: model_ode(t, y, *args),
                    t_span=(0,n_days),
                    y0=y0,
                    method='RK45',
                    t_eval=t_eval,
                    max_step=1e-2
                )
    
    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                'beta': beta, 'alpha': alpha, 'gamma': gamma, 'mu': mu,
                'i0': sol.y[2:4, 20*100],
                'r0': sol.y[4:6, 20*100],
                'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                'week_max': week_max,
                'rate_max': rate_max,
                'RTOL': 1e-6, 'OTOL': 1e-6}

    t0 = time()
    _ = optimal_vaccination_strategy_optimized(parameters)
    print(time() - t0)

    K = 5

    # Fixed parameters
    gamma = 1/7
    mu = 3.6e-5
    beta = np.array([0.25, 0.2, 0.15, 0.15, 0.1])
    alpha = 0.64

    # Population settings
    x1 = 0.2
    populations = 1e5*np.array([50, 10, 10, 1, 1])
    p_matrix = np.array([[1,0,0,0,0],[x1,1-x1,0,0,0],[x1,0,1-x1,0,0],[x1,0,0,1-x1,0],[x1,0,0,0,1-x1]])
    population_eff = populations @ p_matrix

    # Initial condition
    y0 = np.zeros(4*K)
    y0[K] = 1/populations[0]
    y0[0:K] = 1-y0[K:2*K]

    T = 42

    rate_max = np.array([0.8, 0.8, 0.8, 0.8, 0.8])/T
    week_max = np.array([1, 2, 3, 4, 5, 6, 7, 8])/20

    args = (np.zeros(5), alpha, beta, gamma, mu, K, p_matrix, populations, population_eff)

    n_days = 200
    t_eval = np.linspace(0, n_days, 100 * n_days)

    sol = solve_ivp(fun=lambda t, y: model_ode(t, y, *args),
                    t_span=(0,n_days),
                    y0=y0,
                    method='RK45',
                    t_eval=t_eval,
                    max_step=1e-2
                )
    
    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                  'beta': beta, 'alpha': alpha, 'gamma': gamma, 'mu': mu,
                  'npd': 15, 'remote': True,
                  'i0': sol.y[K:2*K, 120*100], 
                  'r0': sol.y[2*K:3*K, 120*100], 
                  'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                  'week_max': week_max,
                  'rate_max': rate_max,
                  'RTOL': 1e-6, 'OTOL': 1e-6}

    t0 = time()
    _  = optimal_vaccination_strategy_optimized(parameters)
    print(time() - t0)

def exp1():
    # Experiment for figure "numerical_simulations_control_fig4.pdf" with 5 cities.
    # number of cities
    K = 5

    # Fixed parameters
    gamma = 1/7
    mu = 3.6e-5
    beta = np.array([0.25, 0.2, 0.15, 0.15, 0.1])
    alpha = 0.64

    # Population settings
    x1 = 0.2
    populations = 1e5*np.array([50, 10, 10, 1, 1])
    p_matrix = np.array([[1,0,0,0,0],[x1,1-x1,0,0,0],[x1,0,1-x1,0,0],[x1,0,0,1-x1,0],[x1,0,0,0,1-x1]])
    population_eff = populations @ p_matrix

    # Initial condition
    y0 = np.zeros(4*K)
    y0[K] = 1/populations[0]
    y0[0:K] = 1-y0[K:2*K]

    T = 42

    rate_max = np.array([0.8, 0.8, 0.8, 0.8, 0.8])/T
    week_max = np.array([1, 2, 3, 4, 5, 6, 7, 8])/20

    args = (np.zeros(5), alpha, beta, gamma, mu, K, p_matrix, populations, population_eff)

    n_days = 200
    t_eval = np.linspace(0, n_days, 100 * n_days)

    sol = solve_ivp(fun=lambda t, y: model_ode(t, y, *args),
                    t_span=(0,n_days),
                    y0=y0,
                    method='RK45',
                    t_eval=t_eval,
                    max_step=1e-2
                )
    
    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                  'beta': beta, 'alpha': alpha, 'gamma': gamma, 'mu': mu,
                  'npd': 15, 'remote': True,
                  'i0': sol.y[K:2*K, 120*100], 
                  'r0': sol.y[2*K:3*K, 120*100], 
                  'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                  'week_max': week_max,
                  'rate_max': rate_max,
                  'RTOL': 1e-8, 'OTOL': 1e-8}

    t0 = time()
    m, susceptible, infected, vaccinated, hospitalized, w, new_infections  = optimal_vaccination_strategy(parameters)
    print(time() - t0)
    data = gk2dict(susceptible, infected, vaccinated, hospitalized, w, new_infections, K)

    with open('data.json', 'w') as file:
        json.dump(data, file)

def exp2():
    # Experiment for figure "vaccination_strategy_comparison.pdf"

    # number of cities
    K = 5
    T = 42

    # Fixed parameters
    gamma = 1/7
    mu = 3.6e-5
    beta = np.array([0.25, 0.2, 0.15, 0.15, 0.1])
    alpha = 0.64

    # Population settings
    x1 = 0.2
    populations = 1e5*np.array([50, 10, 10, 1, 1])
    p_matrix = np.array([[1,0,0,0,0],[x1,1-x1,0,0,0],[x1,0,1-x1,0,0],[x1,0,0,1-x1,0],[x1,0,0,0,1-x1]])

    rate_max = np.array([0.8, 0.8, 0.8, 0.8, 0.8])/T
    week_max = np.array([1, 2, 3, 4, 5, 6, 7, 8])/20
    
    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                  'beta': beta, 'alpha': alpha, 'gamma': gamma, 'mu': mu,
                  'npd': 15, 'remote': True,
                  'i0': np.array([0.03, 0.02, 0.02, 0.01, 0.01]), 
                  'r0': np.array([0.02, 0.01, 0.01, 0.005, 0.005]), 
                  'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                  'week_max': week_max,
                  'rate_max': rate_max,
                  'RTOL': 1e-8, 'OTOL': 1e-8}
    
    data = {}
    beta0_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    
    for index, beta0 in enumerate(beta0_values):

        beta[0] = beta0
        parameters['beta'] = beta
        _, _, _, _, _, _, new_infections  = optimal_vaccination_strategy(parameters)
        _, _, _, _, _, _, new_infections_const  = constant_vaccination(parameters)

        data['exp{}'.format(index)] = (beta[0], 
                                       [list(np.array(new_infections[city])) for city in range(K)], 
                                       [list(np.array(new_infections_const[city])) for city in range(K)])

        print('\nMESSAGE - Finished beta = {}\n'.format(beta[0]))

    with open('data2.json', 'w') as file:
        json.dump(data, file)
        
def exp3():
    # Experiment for figure "numerical_simulations_control_fig6.pdf"
    # number of cities
    K = 5

    # Fixed parameters
    gamma = 1/7
    mu = 3.6e-5
    beta = np.array([0.25, 0.2, 0.15, 0.15, 0.1])
    alpha = 0.64

    # Population settings
    x1 = 0.2
    populations = 1e5*np.array([50, 10, 10, 1, 1])
    p_matrix = np.array([[1,0,0,0,0],[x1,1-x1,0,0,0],[x1,0,1-x1,0,0],[x1,0,0,1-x1,0],[x1,0,0,0,1-x1]])
    population_eff = populations @ p_matrix

    # Initial condition
    y0 = np.zeros(4*K)
    y0[K] = 1/populations[0]
    y0[0:K] = 1-y0[K:2*K]

    T = 42

    rate_max = np.array([0.8, 0.8, 0.8, 0.8, 0.8])/T
    week_max = np.array([1, 2, 3, 4, 5, 6, 7, 8])/20

    args = (np.zeros(5), alpha, beta, gamma, mu, K, p_matrix, populations, population_eff)

    n_days = 200
    t_eval = np.linspace(0, n_days, 100 * n_days)

    sol = solve_ivp(fun=lambda t, y: model_ode(t, y, *args),
                    t_span=(0,n_days),
                    y0=y0,
                    method='RK45',
                    t_eval=t_eval,
                    max_step=1e-2
                )
    
    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                  'beta': beta, 'alpha': alpha, 'gamma': gamma, 'mu': mu,
                  'npd': 15, 'remote': True,
                  'i0': sol.y[K:2*K, 120*100], 
                  'r0': sol.y[2*K:3*K, 120*100], 
                  'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                  'week_max': week_max,
                  'rate_max': rate_max,
                  'RTOL': 1e-8, 'OTOL': 1e-8}

    t0 = time()
    m, susceptible, infected, vaccinated, hospitalized, u  = optimal_vaccination_strategy_feedback(parameters)
    print(time() - t0)
    data = gk2dict(susceptible, infected, vaccinated, hospitalized, u, [[0.0],[0.0],[0.0],[0.0],[0.0]], K)

    with open('data3.json', 'w') as file:
        json.dump(data, file)

def exp4():

    # Population settings
    p_matrix = pd.read_csv('rio_de_janeiro_transition_matrix.csv', index_col=0).to_numpy()
    populations = pd.read_csv('rio_de_janeiro_population.csv', index_col=0).to_numpy().flatten()
    population_eff = p_matrix.T @ populations

    # number of cities
    K = populations.shape[0]

    # Fixed parameters
    gamma = 1/7
    mu = 3.6e-5
    alpha = 0.64

    rng = np.random.RandomState(seed=10)
    #beta = np.sort(0.3*rng.random(size=K))[::-1]
    x = np.sort(rng.pareto(a=10, size=19))[::-1]
    beta = 0.3*(x+0.01)/(x+0.01).max()

    # Initial condition
    y0 = np.zeros(4*K)
    y0[K] = 1/populations[0]
    y0[0:K] = 1-y0[K:2*K]

    T = 56

    rate_max = 0.6*np.ones(K)/T
    week_max = 0.4*np.array([1, 2, 3, 4, 5, 6, 7, 8])/8

    args = (np.zeros(K), alpha, beta, gamma, mu, K, p_matrix, populations, population_eff)

    n_days = 120
    t_eval = np.linspace(0, n_days, 20 * n_days)

    sol = solve_ivp(fun=lambda t, y: model_ode(t, y, *args),
                    t_span=(0,n_days),
                    y0=y0,
                    method='RK45',
                    t_eval=t_eval,
                    max_step=1e-2
                )

    day = np.where(np.sum((sol.y[:K,:].T*populations).T, axis=0)/populations.sum() < 0.95)[0][0]

    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                'beta': beta, 'alpha': alpha, 'gamma': gamma, 'mu': mu,
                'npd': 15, 'remote': True,
                'i0': sol.y[K:2*K, day],
                'r0': sol.y[2*K:3*K, day],
                'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                'week_max': week_max,
                'rate_max': rate_max,
                'RTOL': 1e-6, 'OTOL': 1e-6}

    t0 = time()
    _, susceptible, infected, vaccinated, hospitalized, w  = optimal_vaccination_strategy_optimized(parameters)
    print(time() - t0)
    data1 = {}
    data1['susceptible'] = [list(np.array(susceptible[city])) for city in range(K)]
    data1['infected'] = [list(np.array(infected[city])) for city in range(K)]
    data1['vaccinated'] = list(np.array(vaccinated))
    data1['hospitalized'] = list(np.array(hospitalized))
    data1['w'] = [list(np.array(w[city])) for city in range(K)]
    with open('data4.json', 'w') as file:
        json.dump(data1, file)

    summing = np.diag(p_matrix) + p_matrix[:,0]
    p_matrix_ = np.diag(np.diag(p_matrix)/summing)
    p_matrix_[:,0] = p_matrix[:,0]/summing
    p_matrix_[0,0] = 1.0
    
    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix_,
                  'beta': beta, 'alpha': alpha, 'gamma': gamma, 'mu': mu,
                  'npd': 15, 'remote': True,
                  'i0': sol.y[K:2*K, day],
                  'r0': sol.y[2*K:3*K, day],
                  'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                  'week_max': week_max,
                  'rate_max': rate_max,
                  'RTOL': 1e-6, 'OTOL': 1e-6}

    t0 = time()
    _, susceptible, infected, vaccinated, hospitalized, w  = optimal_vaccination_strategy_metropolitan(parameters)
    print(time() - t0)
    data2 = {}
    data2['susceptible'] = [list(np.array(susceptible[city])) for city in range(K)]
    data2['infected'] = [list(np.array(infected[city])) for city in range(K)]
    data2['vaccinated'] = list(np.array(vaccinated))
    data2['hospitalized'] = list(np.array(hospitalized))
    data2['w'] = [list(np.array(w[city])) for city in range(K)]
    with open('data5.json', 'w') as file:
        json.dump(data2, file)

def exp5():
    
    # Population settings
    p_matrix = pd.read_csv('rio_de_janeiro_transition_matrix.csv', index_col=0).to_numpy()
    populations = pd.read_csv('rio_de_janeiro_population.csv', index_col=0).to_numpy().flatten()
    population_eff = p_matrix.T @ populations

    # number of cities
    K = populations.shape[0]

    # Fixed parameters
    gamma = 1/7
    mu = 3.6e-5
    alpha = 0.64

    rng = np.random.RandomState(seed=10)
    beta = np.sort(0.3*rng.random(size=K))[::-1]

    # Initial condition
    y0 = np.zeros(4*K)
    y0[K] = 1/populations[0]
    y0[0:K] = 1-y0[K:2*K]

    T = 56

    rate_max = 0.6*np.ones(K)/T
    week_max = 0.4*np.array([1, 2, 3, 4, 5, 6, 7, 8])/8

    args = (np.zeros(K), alpha, beta, gamma, mu, K, p_matrix, populations, population_eff)

    n_days = 120
    t_eval = np.linspace(0, n_days, 20 * n_days)

    sol = solve_ivp(fun=lambda t, y: model_ode(t, y, *args),
                    t_span=(0,n_days),
                    y0=y0,
                    method='RK45',
                    t_eval=t_eval,
                    max_step=1e-2
                )

    day = np.where(np.sum((sol.y[:K,:].T*populations).T, axis=0)/populations.sum() < 0.95)[0][0]

    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                'beta': beta, 'alpha': alpha, 'gamma': gamma, 'mu': mu,
                'npd': 10, 'remote': True,
                'i0': sol.y[K:2*K, day],
                'r0': sol.y[2*K:3*K, day],
                'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                'week_max': week_max,
                'rate_max': rate_max,
                'RTOL': 1e-6, 'OTOL': 1e-6}

    t0 = time()
    _, _, _, _, _, u  = constant_vaccination_optimized(parameters)
    print(time() - t0)
    print(u.value)

def exp6():

    # Population settings
    p_matrix = pd.read_csv('rio_de_janeiro_transition_matrix.csv', index_col=0).to_numpy()
    populations = pd.read_csv('rio_de_janeiro_population.csv', index_col=0).to_numpy().flatten()
    population_eff = p_matrix.T @ populations

    # number of cities
    K = populations.shape[0]

    # Fixed parameters
    gamma = 1/7
    mu = 3.6e-5
    alpha = 0.64

    rng = np.random.RandomState(seed=10)
    #beta = np.sort(0.3*rng.random(size=K))[::-1]
    x = np.sort(rng.pareto(a=10, size=19))[::-1]
    beta = 0.3*(x+0.01)/(x+0.01).max()

    # Initial condition
    y0 = np.zeros(4*K)
    y0[K] = 1/populations[0]
    y0[0:K] = 1-y0[K:2*K]

    T = 56

    rate_max = 0.6*np.ones(K)/T
    week_max = 0.4*np.array([1, 2, 3, 4, 5, 6, 7, 8])/8

    args = (np.zeros(K), alpha, beta, gamma, mu, K, p_matrix, populations, population_eff)

    n_days = 120
    t_eval = np.linspace(0, n_days, 20 * n_days)

    sol = solve_ivp(fun=lambda t, y: model_ode(t, y, *args),
                    t_span=(0,n_days),
                    y0=y0,
                    method='RK45',
                    t_eval=t_eval,
                    max_step=1e-2
                )

    day = np.where(np.sum((sol.y[:K,:].T*populations).T, axis=0)/populations.sum() < 0.95)[0][0]

    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                'beta': beta, 'alpha': alpha, 'gamma': gamma, 'mu': mu,
                'npd': 15, 'remote': True,
                'i0': sol.y[K:2*K, day],
                'r0': sol.y[2*K:3*K, day],
                'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                'week_max': week_max,
                'rate_max': rate_max,
                'RTOL': 1e-6, 'OTOL': 1e-6}

    t0 = time()
    _, susceptible, infected, vaccinated, hospitalized, u  = optimal_vaccination_strategy_uniform(parameters)
    print(time() - t0)
    data = {}
    data['susceptible'] = [list(np.array(susceptible[city])) for city in range(K)]
    data['infected'] = [list(np.array(infected[city])) for city in range(K)]
    data['vaccinated'] = list(np.array(vaccinated))
    data['hospitalized'] = list(np.array(hospitalized))
    data['u'] = list(np.array(u))
    with open('data6.json', 'w') as file:
        json.dump(data, file)

def exp7():
    # Experiment for figure "numerical_simulations_control_fig4.pdf" with 5 cities.
    # number of cities
    K = 5

    # Fixed parameters
    gamma = 1/7
    mu = 3.6e-5
    beta = np.array([0.25, 0.2, 0.15, 0.15, 0.1])
    alpha = 0.64

    # Population settings
    x1 = 0.2
    populations = 1e5*np.array([50, 10, 10, 1, 1])
    p_matrix = np.array([[1,0,0,0,0],[x1,1-x1,0,0,0],[x1,0,1-x1,0,0],[x1,0,0,1-x1,0],[x1,0,0,0,1-x1]])
    population_eff = populations @ p_matrix

    # Initial condition
    y0 = np.zeros(4*K)
    y0[K] = 1/populations[0]
    y0[0:K] = 1-y0[K:2*K]

    T = 42

    rate_max = np.array([0.8, 0.8, 0.8, 0.8, 0.8])/T
    week_max = np.array([1, 2, 3, 4, 5, 6, 7, 8])/20

    args = (np.zeros(5), alpha, beta, gamma, mu, K, p_matrix, populations, population_eff)

    n_days = 200
    t_eval = np.linspace(0, n_days, 100 * n_days)

    sol = solve_ivp(fun=lambda t, y: model_ode(t, y, *args),
                    t_span=(0,n_days),
                    y0=y0,
                    method='RK45',
                    t_eval=t_eval,
                    max_step=1e-2
                )
    
    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                  'beta': beta, 'alpha': lambda t: int(not (1/4 <= (t % 1) < 3/4)), 'gamma': gamma, 'mu': mu,
                  'npd': 15, 'remote': True,
                  'i0': sol.y[K:2*K, 120*100], 
                  'r0': sol.y[2*K:3*K, 120*100], 
                  'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                  'week_max': week_max,
                  'rate_max': rate_max,
                  'RTOL': 1e-8, 'OTOL': 1e-8}

    t0 = time()
    m, susceptible, infected, vaccinated, hospitalized, w  = optimal_vaccination_strategy_alpha_time(parameters)
    print(time() - t0)
    data = {}
    data['susceptible'] = [list(np.array(susceptible[city])) for city in range(K)]
    data['infected'] = [list(np.array(infected[city])) for city in range(K)]
    data['vaccinated'] = list(np.array(vaccinated))
    data['hospitalized'] = list(np.array(hospitalized))
    data['w'] = [list(np.array(w[city])) for city in range(K)]

    with open('data7.json', 'w') as file:
        json.dump(data, file)

    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                  'beta': beta, 'alpha': lambda t: 1 - max(0, min(1, 6 * (t % 1) - 1, 1 - 6 * (t % 1) + 4)), 'gamma': gamma, 'mu': mu,
                  'npd': 15, 'remote': True,
                  'i0': sol.y[K:2*K, 120*100], 
                  'r0': sol.y[2*K:3*K, 120*100], 
                  'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                  'week_max': week_max,
                  'rate_max': rate_max,
                  'RTOL': 1e-8, 'OTOL': 1e-8}

    t0 = time()
    m, susceptible, infected, vaccinated, hospitalized, w  = optimal_vaccination_strategy_alpha_time(parameters)
    print(time() - t0)
    data = {}
    data['susceptible'] = [list(np.array(susceptible[city])) for city in range(K)]
    data['infected'] = [list(np.array(infected[city])) for city in range(K)]
    data['vaccinated'] = list(np.array(vaccinated))
    data['hospitalized'] = list(np.array(hospitalized))
    data['w'] = [list(np.array(w[city])) for city in range(K)]

    with open('data8.json', 'w') as file:
        json.dump(data, file)

    parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
                  'beta': beta, 'alpha': lambda t: 0.5 + 0.5 * np.cos(2 * np.pi * t), 'gamma': gamma, 'mu': mu,
                  'npd': 15, 'remote': True,
                  'i0': sol.y[K:2*K, 120*100], 
                  'r0': sol.y[2*K:3*K, 120*100], 
                  'cv': 0.01, 'ch': 1000, 'rh': 0.1,
                  'week_max': week_max,
                  'rate_max': rate_max,
                  'RTOL': 1e-8, 'OTOL': 1e-8}

    t0 = time()
    m, susceptible, infected, vaccinated, hospitalized, w = optimal_vaccination_strategy_alpha_time(parameters)
    print(time() - t0)
    data = {}
    data['susceptible'] = [list(np.array(susceptible[city])) for city in range(K)]
    data['infected'] = [list(np.array(infected[city])) for city in range(K)]
    data['vaccinated'] = list(np.array(vaccinated))
    data['hospitalized'] = list(np.array(hospitalized))
    data['w'] = [list(np.array(w[city])) for city in range(K)]

    with open('data9.json', 'w') as file:
        json.dump(data, file)

if __name__ == '__main__':

    experiment = int(input("What is your experiment?"))
    if experiment == 0:
        exp0()
    elif experiment == 1:
        exp1()
    elif experiment == 2:
        exp2()
    elif experiment == 3:
        exp3()
    elif experiment == 4:
        exp4()
    elif experiment == 5:
        exp5()
    elif experiment == 6:
        exp6()
    elif experiment == 7:
        exp7()
