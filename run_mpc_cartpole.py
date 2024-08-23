import scienceplots
from mpc import *

plt.style.use(['ieee', 'science'])
filepath = "CartPole/multi_traj/control/dissipative_2/noise_ 0.0100/trial_12/"

result = Result(filepath=filepath)
x0 = torch.tensor([[0, 0, 0.6, 0]]).to(device=device, dtype=dtype)
delta_t = result.gt_model.delta_t
model = CartPole(delta_t=delta_t)
koopman_model = LinearModelDiscrete(A=result.A, B=result.B)
Q = np.eye(model.num_states)
lower_u = -np.inf
upper_u = np.inf
n_horizon = 100
n_robust = 1
steps = 800
setup_mpc = {
    'n_horizon': n_horizon,
    't_step': delta_t,
    'n_robust': n_robust,
    'store_full_solution': True,
}

mpc = MPC(model=model)

set_and_run_mpc(mpc, steps,
                x0, Q,
                lower_u, upper_u,
                setup_mpc)
mpc.save('witout_surrogate', steps, path=filepath)
#

mpc_with_surrogate = MPC(model=model,surrogate_model=koopman_model,observable=result.observable)
set_and_run_mpc(mpc_with_surrogate, steps,
                x0, Q,
                lower_u, upper_u,
                setup_mpc)
mpc_with_surrogate.save('with_surrogate', steps, path=filepath)
#
#
fig, ax = plt.subplots(2, 2, figsize=(5, 3), gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
#
plot_mpc_results(ax, mpc, steps, label='Without Surrogate Model')
plot_mpc_results(ax, mpc_with_surrogate, steps, label='With Surrogate Model')
ax[1,1].legend(loc='center left', bbox_to_anchor=(1.05,0.5))
fig.suptitle(f'Noise {result.noise: 0.4f} {result.observable.name} order {result.order} ({result.strategy})')
fig.show()
fig.savefig(f'{result.filepath}/mpc_results.png',dpi=300, format='png')