import numpy as np
from pybads import BADS
from pybads.bads import BADSDump
from pybads.function_examples import quadratic_unknown_noisy_fcn, extra_noisy_quadratic_fcn, quadratic_hetsk_noisy_fcn, rosebrocks_hetsk_noisy_fcn

D = 2
x0 = np.ones((1, D)) * -3      # Starting point
lb = np.ones((1, D)) * -7      # Lower bounds
ub = np.ones((1, D)) * 7       # Upper bounds
plb = np.ones((1, D)) * -3     # Plausible lower bounds
pub = np.ones((1, D)) * 3     # Plausible upper bounds

title = 'Noise objective function'
print("\n *** Example 3: " + title)
print("\t We test BADS on a noisy quadratic function with unit Gaussian noise.")
bads = BADS(quadratic_unknown_noisy_fcn, x0, lb, ub, plb, pub)

optimize_result = bads.optimize()
x_min = optimize_result['x']
fval = optimize_result['fval'] 
print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval= {fval} \n\t \
    total time: {round(bads.optim_state['total_time'], 2)} s \n overhead: {round(bads.optim_state['overhead'], 2)}")
x_global_min = np.array([0., 0.])
print(f"The true, noiseless minimum is at x = {np.sum(x_min**2)} \n")
print(f"The true global minimum is at x = [0, 0], where fval = 0\n")

bads_dump = BADSDump("stobads_noise")
bads_dump.to_JSON(bads.x, bads.u, bads.fval, bads.fsd, bads.iteration_history,
            x_global_min)

extra_noise = True
if extra_noise:
    title = 'Extra Noise objective function'
    print("\n *** Example 4: " + title)
    print("\t We test BADS on a particularly noisy function.")
    bads = BADS(extra_noisy_quadratic_fcn, x0, lb, ub, plb, pub)
    optimize_result = bads.optimize()
    x_min = optimize_result['x']
    fval = optimize_result['fval']
    print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval= {fval} \n\t \
    total time: {round(bads.optim_state['total_time'], 2)} s \n overhead: {round(bads.optim_state['overhead'], 2)}")
    print(f"The true global minimum is at x = [0, 0], where fval = 0\n")
   
heteroskedastic_noise = True 
if heteroskedastic_noise:
    user_dict = {"uncertainty_handling": True, "specify_target_noise": True}
    title = 'Heteroskedastic Noise objective function'
    print("\n *** Example 5: " + title)
    bads = BADS(quadratic_hetsk_noisy_fcn, x0, lb, ub, plb, pub, options=user_dict)
    optimize_result = bads.optimize()
    x_min = optimize_result['x']
    fval = optimize_result['fval'] 

    print(f"BADS minimum at: \n\n\t x = {x_min.flatten()} \n\t fval= {fval} \n\t \
    total time: {round(bads.optim_state['total_time'], 2)} s \n overhead: {round(bads.optim_state['overhead'], 2)}")
    print(f"The true global minimum is at x = [1, 1], where fval = 0\n")
