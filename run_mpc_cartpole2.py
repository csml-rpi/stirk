import scienceplots
from mpc import *
import config

config.update_device(0)
plt.style.use(['ieee', 'science','no-latex'])
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0010/hyperopt/trial_9/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0010/validation_set_added_2/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0010/optimal_traj+random_act_1/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/optimal_traj+random_act_6/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/optimal_traj+random_act_9/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/optimal_traj+random_act_7/"

# filepath = "./CartPole2/multi_traj/control/dmd/Polyflow/noise_ 0.0010/realization/_14/"
# filepath = "./CartPole2/multi_traj/control/dmd/RBF/noise_ 0.0100/realization/_2/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/random_act_2/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/realizations/seed_100_2/" # doesn't work oscillates
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/realizations/seed_123_1/" # works

# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/realizations/seed_124_1/" # works
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/realizations/seed_125_1/" # doesn't work oscillates
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/realizations/seed_126_1/" # works

# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/random_act_2/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/random_act_2/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/adaptive/with_previous_failed_trajectories_1/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/adaptive/trial_1/"
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0010/test_tuning_4/"
filepath = "./CartPole2/multi_traj/control/dmd/RBF/noise_ 0.0010/may_21/seed_100_1/"
# filepath = "CartPole2/multi_traj/control/dmd/RBF/noise_ 0.0100/new_/seed_100_1/"

result = Result(filepath=filepath)

delta_t = result.gt_model.delta_t
model = CartPole2(delta_t=delta_t)
koopman_model = LinearModelDiscrete(A=result.A, B=result.B)



x0 = torch.tensor([[0.5, torch.pi/8, 0.01, 0.01]])
# x0 = torch.tensor([[2.0, torch.pi/1.5, 0.5, 0.1]])

Q = np.eye(model.num_states)
lower_u = -20
upper_u = 20
n_horizon = 20
n_robust = 0
steps = 200
setup_mpc = {
    'n_horizon': n_horizon,
    't_step': delta_t,
    'n_robust': n_robust,
    'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
}

# mpc = MPC(model=model)
#
# set_and_run_mpc(mpc, steps,
#                 x0, Q,
#                 lower_u, upper_u,
#                 setup_mpc)
# mpc.save('witout_surrogate', steps, path=filepath)
#
result.observable.to_cpu()
mpc_with_surrogate = MPC(model=model,surrogate_model=koopman_model,observable=result.observable)
set_and_run_mpc(mpc_with_surrogate, steps,
                x0, Q,
                lower_u, upper_u,
                setup_mpc)
mpc_with_surrogate.save('with_surrogate', steps, path=filepath)

fig, ax = plt.subplots(6, 1, figsize=(5, 6))


plot_mpc_results(ax, mpc_with_surrogate, steps, label='With Surrogate Model')
ax[-1].legend(loc='center left', bbox_to_anchor=(1.05,0.5))
fig.suptitle(f'Noise {result.noise: 0.4f} {result.observable.name} order {result.order} ({result.strategy})')
fig.savefig(f'{result.filepath}/mpc_results_{np.round(x0.numpy().squeeze(), decimals=2)}.png',dpi=300, format='png')

sim_data = mpc_with_surrogate.simulator.data
x = sim_data['_x'][np.newaxis,:]
fig, ax = plt.subplots()
animate_cartpole2(x, ax, fig, filename=f'movie_surrogate_{np.round(x0.numpy().squeeze(), decimals=2)}', path=result.filepath,fps=20)