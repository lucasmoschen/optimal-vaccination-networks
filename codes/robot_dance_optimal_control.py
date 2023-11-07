import numpy as np
import gekko as gk

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scipy.integrate import solve_ivp

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

def model_ode(t, y, u, alpha, beta, tau, gamma, K, p_matrix, population, population_eff):
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
    tau (float): The rate at which exposed individuals become infectious (the inverse of the incubation period).
    gamma (float): The recovery rate (the inverse of the infectious period).
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
    S, E, I, R, _ = y[:K], y[K:2*K], y[2*K:3*K], y[3*K:4*K], y[4*K:]
    I_eff = (I * population) @ p_matrix / population_eff
    S_dot = -alpha * beta * S * I - (1-alpha) * S * (p_matrix @ (beta * I_eff))
    E_dot = -S_dot - (tau + u) * E
    I_dot = tau * E - gamma * I
    R_dot = gamma * I - u*R
    S_dot -= u*S
    H_dot = tau * E
    
    return np.hstack([S_dot, E_dot, I_dot, R_dot, H_dot])

# Setting the parameters
K = 2

populations = np.array([1e6, 1e5])
p_matrix = np.array([[1,0],[0.2, 0.8]])
population_eff = populations @ p_matrix

beta = np.array([0.4, 0.2])
alpha = 0.64
tau = 1/3
gamma = 1/7

T = 56

rate_max = np.array([0.9, 0.8])/T
week_max = np.array([1, 2, 3, 4, 5, 6, 7, 8])/13

# Initial condition
y0 = np.zeros(5*K)
y0[K] = 1/populations[0]
y0[0:K] = 1-y0[K:2*K]

n_days = 180
t_eval = np.linspace(0, n_days, 100 * n_days)

args = (np.zeros(2), alpha, beta, tau, gamma, K, p_matrix, populations, population_eff)

sol = solve_ivp(fun=lambda t, y: model_ode(t, y, *args),
                t_span=(0,n_days),
                y0=y0,
                method='RK45',
                t_eval=t_eval,
                max_step=1e-2
               )

parameters = {'T': T, 'populations': populations, 'p_matrix': p_matrix,
              'beta': beta, 'alpha': alpha, 'tau': tau, 'gamma': gamma, 
              'e0': sol.y[2:4,75*100], 
              'i0': sol.y[4:6,75*100], 
              'r0': sol.y[6:8,75*100],
              'cv': 1, 'ch': 100, 'rh': 0.1,
              'week_max': week_max,
              'rate_max': rate_max,
              'RTOL': 1e-6, 'OTOL': 1e-6}

# Gekko object
m = gk.GEKKO()

# Including the time
T = parameters['T']
npd = 20
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
tau = m.Const(parameters['tau'])
gamma = m.Const(parameters['gamma'])
p = parameters['p_matrix']
population_eff = populations @ p_matrix

# Initial values
e0 = parameters['e0']
i0 = parameters['i0']
r0 = parameters['r0']
s0 = np.ones(K) - e0 - i0 - r0

# Constraints
week_max = parameters['week_max']
rate_max = parameters['rate_max']

# Including the variables
s = m.Array(m.Var, K, lb=0.0)
e = m.Array(m.Var, K, lb=0.0)
i = m.Array(m.Var, K, lb=0.0)
r = m.Array(m.Var, K, lb=0.0)
v = m.Array(m.Var, K, lb=0.0, value=0.0)
h = m.Array(m.Var, K, lb=0.0, value=0.0)

u = [m.MV(lb=0.0, value=0.01, name='u{}'.format(city)) for city in range(K)]
for city in range(K):
    s[city].value = s0[city]
    e[city].value = e0[city]
    i[city].value = i0[city]
    r[city].value = r0[city]
    
    u[city].STATUS = 1
    u[city].DCOST = 0

# Maximizing 
obj_points = np.zeros(n_points)
obj_points[-1] = 1.0
final = m.Param(value=obj_points)
m.Obj(cv * m.sum(v*populations) * final + ch * m.sum(h*populations) * final)

# ODE equations
i_eff = [m.Intermediate(m.sum(p_matrix[:,city]*populations*i)/population_eff[city]) for city in range(K)]
for city in range(K):
    new_infections = s[city] * (alpha*beta[city]*i[city] + (1-alpha)*m.sum(beta*p_matrix[city,:]*i_eff))
    m.Equation(s[city].dt() == -new_infections - s[city]*u[city])
    m.Equation(e[city].dt() == new_infections - (tau+u[city])*e[city])
    m.Equation(i[city].dt() == tau*e[city] - gamma*i[city])
    m.Equation(r[city].dt() == gamma*i[city] - u[city]*r[city])
    m.Equation(v[city].dt() == u[city]*(s[city] + e[city] + r[city]))
    m.Equation(h[city].dt() == rh*i[city])
    
    # The vaccination rate is limited
    m.Equation(u[city]*(s[city] + e[city] + r[city]) <= rate_max[city])
    
    
weekly_constraint = []
weekly_mark = []
for week in range(T//7):
    weekly_mark.append(np.zeros(n_points))
    weekly_mark[-1][npd*7*(week+1)] = 1.0
    weekly_constraint.append(m.Param(value=weekly_mark[-1]))
    m.Equation( m.sum(v * populations) * weekly_constraint[-1] <= week_max[week])

# Solving
m.options.IMODE = 6
m.options.NODES = 6
m.options.MV_TYPE = 1
m.options.SOLVER = 3  # Using IPOPT solver
m.options.MAX_ITER = 500

m.options.RTOL=parameters['RTOL']
m.options.OTOL=parameters['OTOL']

print("Done! Let's do it!!")

m.options.DIAGLEVEL = 3

m.solve(disp=True)