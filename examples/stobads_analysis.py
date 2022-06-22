import numpy as np
from pybads.bads.bads import BADS
from pybads.bads.bads_dump import BADSDump
from pybads.function_examples import quadratic_unknown_noisy_fcn, quadratic_noisy_fcn, extra_noisy_quadratic_fcn

np.random.seed(23)

runs = 50

lb = np.array([[-5, -5]])     # Lower bounds
ub = np.array([[5, 5]])       # Upper bounds
plb = np.array([[-2, -2]])      # Plausible lower bounds
pub = np.array([[2, 2]])        # Plausible upper bounds

for run in range(runs):
    x0 = np.random.uniform(low=plb, high=pub, size=(1, 2))
    bads = BADS(extra_noisy_quadratic_fcn, x0, lb, ub, plb, pub, gamma_uncertain_interval=1.96)
    x_min, fval = bads.optimize()
    print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval= {fval} \n\t \
        total time: {round(bads.optim_state['total_time'], 2)} s \n overhead: {round(bads.optim_state['overhead'], 2)}")

    x_global_min = np.array([0., 0.])
    print(f"The true, noiseless minimum is at x = {np.sum(x_min**2)} \n")
    print(f"The true global minimum is at x = [0, 0], where fval = 0\n")
    bads_dump = BADSDump(f"./dumps/bads/bads_extra_noise_run_{run}")
    bads_dump.to_JSON(bads.x, bads.u, bads.fval, bads.fsd, bads.iteration_history, x_global_min)

results = []
for epsilon in np.linspace(0.1, 10, 100):
    success_runs = 0
    for run in range(runs):    
        bads_dump = BADSDump(f"./dumps/bads/bads_extra_noise_run_{run}")
        dump = bads_dump.load_JSON() 
        true_fval = 0
        fval_ub = dump['fval'] + dump['fsd']
        fval_lb = dump['fval'] - dump['fsd']
        is_success_run = (dump['fval'] <= true_fval + epsilon) &\
                        (dump['fval'] >= true_fval - epsilon)\
        
        if is_success_run:
            success_runs += 1
            
    results.append((success_runs / runs, epsilon))

results = list(zip(*results))
from matplotlib import pyplot as plt
plt.plot(results[1], results[0])
plt.show()