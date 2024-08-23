import scienceplots
from mpc import *
import config
plt.style.use(['ieee', 'science','no-latex'])
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filepath', 
                    type=str,
                    default="./CartPole2/multi_traj/control/dissipative/Polyflow/noise_ 0.0359/adaptive/seed_100_1/loop_0/", 
                    help='filepath')
args = parser.parse_args()
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/adaptive/seed_104_l4_1/"
# filepath = "./CartPole2/multi_traj/control/dmd/RBF/noise_ 0.0010/may_21/seed_100_1/"
filepath= args.filepath
# filepath = "./CartPole2/multi_traj/control/standard/Polyflow/noise_ 0.0100/static/num_traj_250_2/"
# makedir(filepath)
# train_data, _, _ = load_data_files(system='cartpole2',
#                              with_control=True,
#                              traj='multi',
#                              noise_level=0,
#                              seed=100,
#                              num_traj=50)
# x0 = train_data['x0']
# num_trajectories = 50

# position_0 = 2* torch.rand(size=(num_trajectories, 1)) - 1  #
# theta_0 = 2*(torch.pi/4) * torch.rand(size=(num_trajectories, 1)) - (torch.pi/4) # 
# velocity_0 = 2*0.1* torch.rand(size=(num_trajectories, 1)) - 0.1  #
# omega_0 = 2*0.1 * torch.rand(size=(num_trajectories, 1)) - 0.1 #

# x0 = torch.concat([position_0,theta_0,velocity_0,omega_0], dim=-1)
# print(x0.shape)
# x0 = torch.tensor([[2.0, torch.pi/1.5, 0.5, 0.1]])



x0 = torch.tensor([[1.0, torch.pi/4, 0.0, 0.0],
                   [0.0, torch.pi/2, 0.1, 0.0],
                   [0.0, -torch.pi/2, -0.1, -0.1],
                   [0.0, -torch.pi/4, 0.1, 0.1]])
# x0 = torch.tensor([[0.0, -torch.pi/4, 0.1, 0.1]])

delta_t = 0.05
model = CartPole2(delta_t=delta_t)
result = Result(filepath=filepath)
koopman_model = LinearModelDiscrete(A=result.A, B=result.B)

task1 = {
    'lower_u': -20,
    'upper_u': 20,
    'n_horizon' : 20,
    'n_robust': 1,
    'steps' : 200,
    'name' : "task1"
}
task2 = {
    'lower_u': -10,
    'upper_u': 10,
    'n_horizon' : 20,
    'n_robust': 1,
    'steps' : 150,
    'name' : "task2"
}

for task in [task1, task2]:
    Q = np.eye(model.num_states)
    lower_u = task['lower_u']
    upper_u = task['upper_u']
    n_horizon = task['n_horizon']
    n_robust = task['n_robust']
    steps = task['steps']

    setup_mpc = {
        'n_horizon': n_horizon,
        't_step': delta_t,
        'n_robust': n_robust,
        'store_full_solution': True,
    }

    result.observable.to_cpu()
    mpc_with_surrogate = MPC(model=model,surrogate_model=koopman_model,observable=result.observable)
    set_mpc(mpc_with_surrogate, Q, lower_u, upper_u, setup_mpc)
    def run_mpc_and_plots(x0):
        run_mpc(mpc_with_surrogate,steps, x0)
        sim_data = mpc_with_surrogate.simulator.data
        filename = f'{np.round(x0.numpy().squeeze(), decimals=2)}_{task["name"]}'
        mpc_with_surrogate.save(filename, steps, path=filepath)
        fig, ax = plt.subplots(6, 1, figsize=(5, 6))
        plot_mpc_results(ax, mpc_with_surrogate, steps, delta_t, label='Without Surrogate Model')
        ax[-1].legend(loc='center left', bbox_to_anchor=(1.05,0.5))
        sim_data = mpc_with_surrogate.simulator.data
        # x = sim_data['_x'][np.newaxis,:]
        fig.savefig(f"{filepath}/mpc_result_{filename}.png")
        # fig, ax = plt.subplots()
        # animate_cartpole2(x, ax, fig, filename=f'movie_{filename}', path=case.filepath,fps=20)
        return sim_data

    # run_mpc_and_plots(x0)

    x_test = []
    u_test = []
    for x_init in x0:
        x_init.unsqueeze_(0)
        sim_data = run_mpc_and_plots(x_init)

# Q = np.eye(model.num_states)
# lower_u = -5
# upper_u = 5
# n_horizon = 20
# n_robust = 1
# steps = 200
# setup_mpc = {
#     'n_horizon': n_horizon,
#     't_step': delta_t,
#     'n_robust': n_robust,
#     'store_full_solution': True,
# }

# result.observable.to_cpu()
# mpc_with_surrogate = MPC(model=model,surrogate_model=koopman_model,observable=result.observable)
# set_mpc(mpc_with_surrogate, Q,
#                     lower_u, upper_u,
#                     setup_mpc)
# path = os.path.join(filepath, f'constraint_{lower_u}_{upper_u}')
# makedir(path)
# def run_mpc_and_plots(x0):
#     run_mpc(mpc_with_surrogate,steps, x0)

#     sim_data = mpc_with_surrogate.simulator.data
#     mpc_with_surrogate.save(f'witout_surrogate_{np.round(x0.numpy().squeeze(), decimals=2)}', steps, path=path)

#     fig, ax = plt.subplots(6, 1, figsize=(5, 6))
#     plot_mpc_results(ax, mpc_with_surrogate, steps, label='Without Surrogate Model')
#     ax[-1].legend(loc='center left', bbox_to_anchor=(1.05,0.5))
#     sim_data = mpc_with_surrogate.simulator.data
#     x = sim_data['_x'][np.newaxis,:]
#     fig.savefig(f"{path}/mpc_result{np.round(x0.numpy().squeeze(), decimals=2)}.png")
#     fig, ax = plt.subplots()
#     animate_cartpole2(x, ax, fig, filename=f'movie_{np.round(x0.numpy().squeeze(), decimals=2)}', path=path,fps=20)
#     return sim_data

# # run_mpc_and_plots(x0)

# x_test = []
# u_test = []
# for x_init in x0:
#     x_init.unsqueeze_(0)
#     sim_data = run_mpc_and_plots(x_init)
#     # x = torch.from_numpy(sim_data["_x"])
#     # u = torch.from_numpy(sim_data["_u"])
#     # x_test.append(x)
#     # u_test.append(u)

# # x_test = torch.stack(x_test, dim= 0)
# # u_test = torch.stack(u_test, dim= 0)
# # print(x_test.shape)
# # print(u_test.shape)
# # print(x0.shape)

# # test_data = dict(x0_test=x0, x_test= x_test, u_test=u_test)
# # save_file(test_data, f"{path}/suboptimal_trajectories_data.pkl")