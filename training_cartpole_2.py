from dmd_algorithms import *
from edmd_rollout import *
from models import *
import scienceplots
from cases import *
import argparse
from utils import * 
import config 
plt.style.use(['science', 'ieee', 'no-latex'])

training_type = 'static'
strategy = 'dissipative'
polyflow_order = 4
start_rollout_length = 2
max_rollout_length = 199
switch_rollout_after_epoch = 500
geometric_progression = True
rollout_increase = 2
scheduler_mode = "cyclicLR"
learning_rate = 0.001
max_lr = 0.01
step_size_up = 1000
step_size_down = 500
cyclic_lr_mode = 'triangular2'
learning_rate_decay = 1.0
batch_size = 2000
num_epochs = 1200
device_id = 0
seed = 100
name = 'jun24'
first_optimizer_is_Adam=True
cycle_optimizer = False
switch_optimizer_after_epoch= 1000
batch_size_lbfgs = 2000
learning_rate_lbfgs = 0.0001
reconstruction_loss_factor = 1.0
noise_level = 8
num_trajectories = 50

parser = argparse.ArgumentParser(description='training cartpole2 multiple trajectories')
parser.add_argument('--strategy', type=str, default=strategy, help='Koopman Model Type')
parser.add_argument('--training-type', type=str, default=training_type, help='Static or Adaptive')

parser.add_argument('--polyflow-order', type=int, default=polyflow_order, help='Polylflow Order')

parser.add_argument('--start-roll-out-length', type=int, default=start_rollout_length, help="initial rollout length")
parser.add_argument('--max-roll-out-length', type=int, default=max_rollout_length, help="max rollout length")
parser.add_argument('--switch-roll-out-after-epoch', type=int, default=switch_rollout_after_epoch, help="switch roll out after epoch")
parser.add_argument('--geometric-progression', type=bool, default=geometric_progression, help="geometric progression")
parser.add_argument('--roll-out-increase', type=int, default=rollout_increase, help="roll out increase")

parser.add_argument('--scheduler-mode', type=str, default=scheduler_mode, help="learning rate scheduler")
parser.add_argument('--learning-rate', type=float, default=learning_rate, help="Learning Rate")
parser.add_argument('--max-lr', type=float, default=max_lr, help="Max Learning Rate")
parser.add_argument('--step-size-up', type=int, default=step_size_up, help="Step size for up")
parser.add_argument('--step-size-down', type=int, default=step_size_down, help="Step size for down")
parser.add_argument('--cyclic-lr-mode', type=str, default=cyclic_lr_mode, help="Cyclic LR mode")

parser.add_argument('--learning-rate-decay', type=float, default=learning_rate_decay, help="learning rate decay")

parser.add_argument('--batch-size', type=int, default=batch_size, help="Batch size")
parser.add_argument('--num-epochs', type=int, default=num_epochs, help="Number of epochs")
parser.add_argument('--device-id', type=int, default=device_id, help="Cuda device")
parser.add_argument('--seed', type=int, default=seed, help="Random Seed")
parser.add_argument('--name', type=str, default=name, help='Naming tuning parameters')


parser.add_argument('--first-optimizer-is-Adam', type=bool, default=first_optimizer_is_Adam, help="Start with Adam or Lbfgs")
parser.add_argument('--cycle-optimizer', type=bool, default=cycle_optimizer, help="optimizer cycling")
parser.add_argument('--switch-optimizer-after-epoch', type=int_or_none, default=switch_optimizer_after_epoch, help="Switch to a different optimizer after certain epochs")

parser.add_argument('--batch-size-lbfgs', type=int_or_none,  default=batch_size_lbfgs, help="Batch size for lbfgs. if none, full batch")
parser.add_argument('--learning-rate-lbfgs', type=float, default=learning_rate_lbfgs, help="Learning rate for LBFGS")
parser.add_argument('--reconstruction-loss-factor', type=float, default=reconstruction_loss_factor, help="Reconstruction loss factor")
parser.add_argument('--noise-level', type=int, default = noise_level, help='0-9 : 0 represents 1e-3 and 9 represents 1e-1')
parser.add_argument('--num-trajectories', type=int, default=num_trajectories, help="Num of trajectories")


args = parser.parse_args()

torch.random.manual_seed(args.seed)
config.update_device(args.device_id)
observer = 'polyflow'
delta_t = 0.05
noise_values = np.logspace(-3, -1, 10) # np.array([0.1, 0.5, 1.0, 1.5, 2.0, 3.5]) #
noise = noise_values[args.noise_level]

model = CartPole2(delta_t=delta_t)


case = Case(gt_model=model,
            noise=noise,
            steps=200,
            strategy=args.strategy)

num_trajectories = args.num_trajectories

train_data, test_data, val_data = load_data_files(system='cartpole2', 
                                                with_control= True,
                                                traj = 'multi', 
                                                noise = noise,
                                                # noise_level= args.noise_level,
                                                seed = args.seed, 
                                                num_traj=num_trajectories,
                                                strategy=args.strategy,
                                                observer=observer,
                                                add_suboptimal_data=False)

case.load_data(train_data=train_data, test_data=test_data, val_data=val_data)


observable = TimedelayImproved(gt_model=case.gt_model,
                               order=args.polyflow_order,
                               delta_t=case.gt_model.delta_t)
case.add_observable(observable)

if case.strategy == "standard":
    koopman_model = StandardKoopmanModel(num_states=case.num_states,
                                         num_inputs=case.num_inputs,
                                         delta_t=case.gt_model.delta_t,
                                         order=case.order,
                                         lifted_states=None,
                                         std_init_B=0.1,
                                         std_init_A=0.01)
elif case.strategy == "dissipative":
    koopman_model = DissipativeKoopmanModel(num_states=case.num_states,
                                            num_inputs=case.num_inputs,
                                            delta_t=case.gt_model.delta_t,
                                            order=case.order,
                                            lifted_states=None,
                                            std_init_B=0.1,
                                            std_init_P=0.1,
                                            std_init_Q=0.1)
elif case.strategy == "dissipative2":
    koopman_model = DissipativeKoopmanModel2(num_states=case.num_states,
                                             num_inputs=case.num_inputs,
                                             delta_t=case.gt_model.delta_t,
                                             order=case.order,
                                             lifted_states=None)
elif case.strategy == "hamiltonian":
    koopman_model = HamiltonianKoopmanModel(num_states=case.num_states,
                                            num_inputs=case.num_inputs,
                                            delta_t=case.gt_model.delta_t,
                                            order=case.order,
                                            lifted_states=None,
                                            std_init_B=0.1,
                                            std_init_P=0.1,
                                            std_init_Q=0.1)

case.create_filepath(trial=True, kw=f'{args.name}/seed_{args.seed}')
case.save_gt_info()
case.save_observable_info()
case.save_test_data()


rollout_edmd = RolloutEDMD(observable=observable,
                           koopman_model=koopman_model,
                           order=case.order,
                           delta_t=case.gt_model.delta_t)

rollout_edmd.setup_comparison(x_true=case.x, x_noisy=case.x_noisy, x0_true=case.x0, steps=case.steps, u_true=case.u)

rollout_edmd.add_validation_data(x0=case.x0_val,
                                 x=case.x_val,
                                 x_noisy=case.x_noisy_val,
                                 u=case.u_val,
                                 )
rollout_edmd.add_test_data(x_true=case.x_test,
                           x_noisy=case.x_noisy_test,
                           u_true=case.u_test,
                           x0_true=case.x0_test)

rollout_edmd.setup(learning_rate=args.learning_rate,
                   plot_history_after_epoch=100,
                   compare_after_epoch=1,
                   num_epochs=args.num_epochs,
                   batch_size=args.batch_size,
                   switch_roll_out_after_epoch=args.switch_roll_out_after_epoch,
                   geometric_progression=args.geometric_progression,
                   roll_out_increase=args.roll_out_increase,
                   max_roll_out_length=args.max_roll_out_length,
                   learning_rate_decay=args.learning_rate_decay,
                   switch_learning_rate_after_epoch=True,
                   first_optimizer_is_Adam=args.first_optimizer_is_Adam,
                   switch_optimizer_after_epoch=args.switch_optimizer_after_epoch,
                   learning_rate_lbfgs=args.learning_rate_lbfgs,
                   batch_size_lbfgs=args.batch_size_lbfgs,
                   reconstruction_loss_factor=args.reconstruction_loss_factor,
                   show=True,
                   cycle_optimizer=args.cycle_optimizer,
                   early_stopping_patience = 1000,
                   early_stopping_threshold = 0.001)

rollout_edmd.filepath = case.filepath

if args.scheduler_mode == "cyclicLR":
    config_cyclic_LR = dict(cycle_momentum=False,
                            max_lr=args.max_lr,
                            base_lr=args.learning_rate,
                            step_size_up=args.step_size_up,
                            step_size_down=args.step_size_down,
                            mode=args.cyclic_lr_mode)
else:
    config_cyclic_LR = None

rollout_edmd.set_optimizers(scheduler_mode=args.scheduler_mode, config_cyclic_lr=config_cyclic_LR)
create_log(case, rollout_edmd)
start_time = time.time()

A, B, error_dict = rollout_edmd.fit(case.x_noisy, u=case.u, strategy=case.strategy, start_roll_out_length=args.start_roll_out_length)
end_time = time.time()
time_taken = end_time -start_time
print(f"Time taken {time_taken}s")
log_file = os.path.join(case.filepath, "log.txt")
with open(log_file, 'a') as f:
    f.write(f"\nTime taken {time_taken}s\n seed {args.seed}")

from mpc import *
# num_trajectories = 50

x0 = torch.tensor([ [1.0, torch.pi/4, 0.0, 0.0],
                    [0.0, torch.pi/2, 0.1, 0.0],
                    [0.0, -torch.pi/2, -0.1, -0.1],
                    [0.0, -torch.pi/4, 0.1, 0.1]])
# train_data, _, _ = load_data_files(system='cartpole2',
#                             with_control=True,
#                             traj='multi',
#                             noise_level=args.noise_level,
#                             seed=args.seed,
#                             num_traj=num_trajectories)
# x0 = train_data['x0']
# print(x0.shape)
# x0 = torch.tensor([[2.0, torch.pi/1.5, 0.5, 0.1]])
# x0 = torch.tensor([[2.0, torch.pi, 0.01, 0]])
delta_t = 0.05
model = CartPole2(delta_t=delta_t)
koopman_model = LinearModelDiscrete(A=A, B=B)

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

for task in [task1]:
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

    observable.to_cpu()
    mpc_with_surrogate = MPC(model=model,surrogate_model=koopman_model,observable=observable)
    set_mpc(mpc_with_surrogate, Q, lower_u, upper_u, setup_mpc)
    def run_mpc_and_plots(x0):
        run_mpc(mpc_with_surrogate,steps, x0)
        sim_data = mpc_with_surrogate.simulator.data
        filename = f'{np.round(x0.numpy().squeeze(), decimals=2)}_{task["name"]}'
        mpc_with_surrogate.save(filename, steps, path=case.filepath)
        fig, ax = plt.subplots(6, 1, figsize=(5, 6))
        plot_mpc_results(ax, mpc_with_surrogate, steps, delta_t, label='Without Surrogate Model')
        ax[-1].legend(loc='center left', bbox_to_anchor=(1.05,0.5))
        sim_data = mpc_with_surrogate.simulator.data
        # x = sim_data['_x'][np.newaxis,:]
        fig.savefig(f"{case.filepath}/mpc_result_{filename}.png")
        # fig, ax = plt.subplots()
        # animate_cartpole2(x, ax, fig, filename=f'movie_{filename}', path=case.filepath,fps=20)
        return sim_data

    # run_mpc_and_plots(x0)

    x_test = []
    u_test = []
    for x_init in x0:
        x_init.unsqueeze_(0)
        sim_data = run_mpc_and_plots(x_init)
    #     x = torch.from_numpy(sim_data["_x"])
    #     u = torch.from_numpy(sim_data["_u"])
    #     x_test.append(x)
    #     u_test.append(u)

    # x_test = torch.stack(x_test, dim= 0)
    # u_test = torch.stack(u_test, dim= 0)
    # print(x_test.shape)
    # print(u_test.shape)
    # print(x0.shape)

    # test_data = dict(x0_test=x0, x_test= x_test, u_test=u_test)
    # save_file(test_data, f"{case.filepath}/suboptimal_trajectories_data.pkl")
# print(case.filepath)