import numpy as np
import gekko as gk
import pickle

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
    populations = parameters['populations']
    K = len(populations)

    # Parameters
    beta = parameters['beta']
    cv = m.Const(parameters['cv'])
    ch = m.Const(parameters['ch'])
    alpha = m.Const(parameters['alpha'])
    gamma = m.Const(parameters['gamma'])
    p_matrix = parameters['p_matrix']
    population_eff = populations @ p_matrix
    if 'q_matrix' in parameters:
        q = parameters['q_matrix']
    else:
        q = np.zeros((K, K))

    # Initial values
    s0 = parameters['s0']
    i0 = parameters['i0']
    
    # Constraints
    week_max = parameters['week_max']
    rate_max = parameters['rate_max']

    # Including the variables
    s = m.Array(m.Var, K, lb=0.0)
    i = m.Array(m.Var, K, lb=0.0)  
    v = m.Var(value=0.0, lb=0.0)
    h = m.Var(value=0.0, lb=0.0)
    
    w = [m.MV(lb=0.0, ub=rate_max[city], name='u{}'.format(city)) for city in range(K)]
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
    for city in range(K):
        new_infections = s[city] * (alpha*beta[city]*i[city] + (1-alpha)*sum(beta*p_matrix[city,:]*i_eff))
        transition_term = sum(q[city, j] * s[j] for j in range(K))
        m.Equation(s[city].dt() == -new_infections - w[city] + transition_term)
        m.Equation(i[city].dt() == new_infections - gamma*i[city])
    m.Equation(v.dt() == sum(w*populations))
    m.Equation(h.dt() == sum(i*populations))

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
    m.options.MAX_ITER = 200
    m.options.DIAGLEVEL = 1
    
    m.options.RTOL=parameters['RTOL']
    m.options.OTOL=parameters['OTOL']

    m.solve(disp=True)

    # Adjoint variables calculation
    s = np.array([list(s[city]) for city in range(K)])
    i = np.array([list(i[city]) for city in range(K)])
    w = np.array([list(w[city]) for city in range(K)])
    psi_s = np.zeros_like(s)
    psi_i = np.zeros_like(i)
    delta_t = 1/npd
    B = np.diag(beta)
    A = alpha.value * B + (1-alpha.value) * p_matrix @ B @ np.diag(1/population_eff) @ p_matrix.T @ np.diag(populations)    
    for index in range(n_points-1, 0, -1):
        derivative = (psi_s[:, index] - psi_i[:, index]) * (A @ i[:, index]) - (q.T @ psi_s[:, index])
        psi_s[:, index-1] = psi_s[:, index] - delta_t*derivative
        
        derivative = A.T @ np.diag(s[:, index]) @ (psi_s[:, index] - psi_i[:, index]) 
        derivative += psi_i[:, index]*gamma.value + ch.value * populations
        psi_i[:, index-1] = psi_i[:, index] - delta_t*derivative
        
    return m, s, i, np.array(v), np.array(h), w, psi_s, psi_i

experiment_number = int(input("Enter the experiment number (1 or 2): "))
if experiment_number == 1:
    print("Starting the simulation for the first experiment")
    T = 28
    gamma = 1/7
    alpha = 0.64
    week_max = np.array([1, 2, 3, 4]) / 30

    # --- Experiment 1: 3 cities ---
    K1 = 3
    beta1 = np.array([0.3, 0.2, 0.1])  # all <= 0.4
    populations1 = 1e5 * np.array([100, 10, 10])
    populations1 = populations1 / populations1.sum()
    p_matrix1 = np.array([[0.9,   0.05, 0.05],
                        [0.45,  0.45, 0.10],
                        [0.45,  0.10, 0.45]])
    rate_max1 = np.full(K1, 0.3) / T
    s0_1 = np.array([0.96, 0.97, 0.95])
    i0_1 = np.array([0.02, 0.02, 0.01])
    params1 = {'T': T,
            'populations': populations1,
            'p_matrix': p_matrix1,
            'beta': beta1,
            'alpha': alpha,
            'gamma': gamma,
            'npd': 15,
            'remote': True,
            's0': s0_1,
            'i0': i0_1,
            'cv': 0.01,
            'ch': 100,
            'week_max': week_max,
            'rate_max': rate_max1,
            'RTOL': 1e-6,
            'OTOL': 1e-6}

    # --- Experiment 2: 5 cities ---
    K2 = 5
    beta2 = np.array([0.35, 0.3, 0.25, 0.2, 0.15])
    populations2 = 1e5 * np.array([50, 30, 10, 5, 5])
    populations2 = populations2 / populations2.sum()
    p_matrix2 = np.array([
        [0.8,  0.05, 0.05, 0.05, 0.05],
        [0.1,  0.7,  0.1,  0.05, 0.05],
        [0.05, 0.1,  0.7,  0.1,  0.05],
        [0.05, 0.05, 0.1,  0.7,  0.1],
        [0.05, 0.05, 0.05, 0.1,  0.75]
    ])
    rate_max2 = np.array([0.35, 0.32, 0.3, 0.28, 0.26]) / T
    s0_2 = np.full(K2, 0.97)  # > 0.95 for all cities
    i0_2 = np.full(K2, 0.01)
    params2 = {'T': T,
            'populations': populations2,
            'p_matrix': p_matrix2,
            'beta': beta2,
            'alpha': alpha,
            'gamma': gamma,
            'npd': 15,
            'remote': True,
            's0': s0_2,
            'i0': i0_2,
            'cv': 0.01,
            'ch': 100,
            'week_max': week_max,
            'rate_max': rate_max2,
            'RTOL': 1e-6,
            'OTOL': 1e-6}

    # --- Experiment 3: 8 cities ---
    K3 = 8
    beta3 = np.array([0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04])
    populations3 = 1e5 * np.array([70, 40, 20, 10, 5, 3, 1, 1])
    populations3 = populations3 / populations3.sum()
    p_matrix3 = np.array([
        [0.7,  0.1,  0.05, 0.05, 0.03, 0.03, 0.02, 0.02],
        [0.1,  0.7,  0.1,  0.02, 0.02, 0.02, 0.02, 0.02],
        [0.05, 0.1,  0.75, 0.02, 0.02, 0.02, 0.02, 0.02],
        [0.05, 0.02, 0.02, 0.75, 0.05, 0.05, 0.03, 0.03],
        [0.03, 0.02, 0.02, 0.05, 0.8,  0.03, 0.02, 0.03],
        [0.03, 0.02, 0.02, 0.05, 0.03, 0.8,  0.03, 0.02],
        [0.02, 0.02, 0.02, 0.03, 0.02, 0.03, 0.85, 0.01],
        [0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.01, 0.85]
    ])
    rate_max3 = np.array([0.5, 0.48, 0.46, 0.44, 0.42, 0.4, 0.38, 0.36]) / T 
    s0_3 = np.full(K3, 0.98)
    i0_3 = np.full(K3, 0.005)
    params3 = {'T': T,
            'populations': populations3,
            'p_matrix': p_matrix3,
            'beta': beta3,
            'alpha': alpha,
            'gamma': gamma,
            'npd': 15,
            'remote': True,
            's0': s0_3,
            'i0': i0_3,
            'cv': 0.01,
            'ch': 100,
            'week_max': week_max,
            'rate_max': rate_max3,
            'RTOL': 1e-6,
            'OTOL': 1e-6}

    experiments = [params1, params2, params3]
    results = []

    for params in experiments:
        m, susceptible, infected, vaccinated, hospitalized, w, psi_s, psi_i = optimal_vaccination_strategy(params)
        results.append((m, w, psi_s))

    with open("simulation_results.pkl", "wb") as f:
        pickle.dump(results, f)

else:
    print("Starting the simulation for the second experiment")
    T = 28
    gamma = 1/7
    alpha = 0.64
    week_max = np.array([1, 2, 3, 4]) / 30

    # --- Experiment 1: 3 cities ---
    K1 = 3
    beta1 = np.array([0.3, 0.2, 0.1])  # all <= 0.4
    populations1 = 1e5 * np.array([100, 10, 10])
    populations1 = populations1 / populations1.sum()
    p_matrix1 = np.array([[0.9,   0.05, 0.05],
                        [0.45,  0.45, 0.10],
                        [0.45,  0.10, 0.45]])
    q3_off = np.array([
    [0.0,  0.03, 0.02],
    [0.02, 0.0,  0.02],
    [0.03, 0.01, 0.0 ]
    ])
    q3 = q3_off.copy()
    for j in range(3):
        q3[j, j] = -np.sum(q3_off[:, j])
    rate_max1 = np.full(K1, 0.3) / T
    s0_1 = np.array([0.96, 0.97, 0.95])
    i0_1 = np.array([0.02, 0.02, 0.01])
    params1 = {'T': T,
            'populations': populations1,
            'p_matrix': p_matrix1,
            'q_matrix': q3,
            'beta': beta1,
            'alpha': alpha,
            'gamma': gamma,
            'npd': 15,
            'remote': True,
            's0': s0_1,
            'i0': i0_1,
            'cv': 0.01,
            'ch': 100,
            'week_max': week_max,
            'rate_max': rate_max1,
            'RTOL': 1e-6,
            'OTOL': 1e-6}

    # --- Experiment 2: 5 cities ---
    K2 = 5
    beta2 = np.array([0.35, 0.3, 0.25, 0.2, 0.15])
    populations2 = 1e5 * np.array([50, 30, 10, 5, 5])
    populations2 = populations2 / populations2.sum()
    p_matrix2 = np.array([
        [0.8,  0.05, 0.05, 0.05, 0.05],
        [0.1,  0.7,  0.1,  0.05, 0.05],
        [0.05, 0.1,  0.7,  0.1,  0.05],
        [0.05, 0.05, 0.1,  0.7,  0.1],
        [0.05, 0.05, 0.05, 0.1,  0.75]
    ])
    q5_off = np.array([
        [0.0,  0.02, 0.01, 0.03, 0.02],
        [0.01, 0.0,  0.02, 0.01, 0.03],
        [0.02, 0.01, 0.0,  0.02, 0.01],
        [0.01, 0.03, 0.01, 0.0,  0.02],
        [0.03, 0.01, 0.02, 0.01, 0.0 ]
    ])
    q5 = q5_off.copy()
    for j in range(5):
        q5[j, j] = -np.sum(q5_off[:, j])
    rate_max2 = np.array([0.35, 0.32, 0.3, 0.28, 0.26]) / T
    s0_2 = np.full(K2, 0.97)  # > 0.95 for all cities
    i0_2 = np.full(K2, 0.01)
    params2 = {'T': T,
            'populations': populations2,
            'p_matrix': p_matrix2,
            'q_matrix': q5,
            'beta': beta2,
            'alpha': alpha,
            'gamma': gamma,
            'npd': 15,
            'remote': True,
            's0': s0_2,
            'i0': i0_2,
            'cv': 0.01,
            'ch': 100,
            'week_max': week_max,
            'rate_max': rate_max2,
            'RTOL': 1e-6,
            'OTOL': 1e-6}
    
    experiments = [params1, params2]
    results = []

    for params in experiments:
        m, susceptible, infected, vaccinated, hospitalized, w, psi_s, psi_i = optimal_vaccination_strategy(params)
        results.append((m, w, psi_s))

    with open("simulation_results2.pkl", "wb") as f:
        pickle.dump(results, f)