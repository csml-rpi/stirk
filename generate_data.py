import argparse

import config
import scienceplots
from cases import *
from edmd_rollout import *
from rbf import *
plt.style.use(['science', 'ieee', 'no-latex'])

parser = argparse.ArgumentParser(description='Generating Data')
seed =100
system = 'cartpole2'
with_control = True
noise_level = 8
device_id = 0
traj = 'multi'
num_traj = 1 if traj=='single' else 10
parser.add_argument('--seed', type=int, default=seed, help='Manual seed')
parser.add_argument('--system', type=str, default=system, help='Dynamical System vdp or cartpole2 or cstr') # 
parser.add_argument('--control', type=lambda x: (str(x).lower() == 'true'), default=with_control, help='With control or without control')
parser.add_argument('--noise-level', type=int, default = noise_level, help='0-9 : 0 represents 1e-3 and 9 represents 1e-1')
parser.add_argument('--traj', type=str, default=traj, help='Different traj')
parser.add_argument('--num-traj', type=int, default=num_traj, help='Different traj')
parser.add_argument('--device', type=int, default = device_id, help='CUDA Device')

args = parser.parse_args()
seed = args.seed
traj = args.traj
system = args.system
device_id = args.device
with_control = args.control

config.update_device(args.device)
torch.random.manual_seed(args.seed)

if system == 'vdp':
    mu = 1.0
    delta_t = 0.1
    steps = 100

    model = VanDerPolDiscreteControl(mu=mu, delta_t=delta_t)
    if args.traj == 'single':
        mean = torch.tensor([[-1.0, -1.0]])
        std = torch.tensor([[0.05, 0.05]]) 
        x0 = torch.normal(mean, std) #2*torch.rand(size=(1,2)) - 1 
    else:
        x0 = 2*torch.rand(size=(args.num_traj,2)) - 1
    steps = 100
elif system == 'cartpole2':
    delta_t = 0.05
    model = CartPole2(delta_t=delta_t)
    num_trajectories = args.num_traj
    steps = 200
    position_0 = 2 * torch.rand(size=(num_trajectories, 1)) - 1  
    theta_0 = 2 * (torch.pi / 2) * torch.rand(size=(num_trajectories, 1)) - (torch.pi /2)  
    velocity_0 = 2 * 0.1 * torch.rand(size=(num_trajectories, 1)) - 0.1 
    omega_0 = 2 * 0.1 * torch.rand(size=(num_trajectories, 1)) - 0.1
    x0 = torch.concat([position_0, theta_0, velocity_0, omega_0], dim=-1)
    test_data = load_file("CartPole2/ground_truth/test_data.pkl")
    x_test = test_data['x_test']
    x0_test = test_data['x0_test']
    u_test = test_data['u_test']

noise_values = np.logspace(-3, -1, 10)

# if args.system == 'vdp':
#     noise_values = np.logspace(-3, -1, 10)
# elif args.system == 'cartpole2':
#     noise_values = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 3.5])

with_control = args.control
noise = noise_values[args.noise_level]

print(f'system : [{args.system}] noise :[{noise}] seed : [{args.seed}] device id: [{args.device}] control [{args.control}] num_traj [{args.num_traj}]')

if system == 'vdp':
    num_test_trajectories = 50
    x0_test = 2 * torch.rand(num_test_trajectories, 2) - 1
    x0_test = x0_test.to(device=config.device, dtype=config.dtype)
    x_test = None
    if with_control:
        u_test = random_wave_input(num_trajectories=num_test_trajectories,
                                    steps=steps,
                                    delta_t=delta_t,
                                    num_inputs=1,
                                    max_input=1,
                                    min_input=-1,
                                    max_freq=0.5,
                                    min_freq=1)
    else:
        u_test = None

case = Case(gt_model=model,
            noise=noise,
            steps=steps,
            x0=x0)

if with_control:
    if system == 'vdp':
        u = random_wave_input(num_trajectories=case.num_trajectories,
                                steps=case.steps,
                                delta_t=case.gt_model.delta_t,
                                num_inputs=1,
                                max_input=1,
                                min_input=-1,
                                max_freq=0.5,
                                min_freq=1)
    elif system == 'cartpole2':
        u = sine_wave_input(num_trajectories=num_trajectories,
                            steps=case.steps,
                            delta_t=case.gt_model.delta_t,
                            num_inputs=1,
                            max_input=30,
                            min_input=-30,
                            max_freq=0.5,
                            min_freq=0,
                            max_bias=0,
                            min_bias=0,
                            max_decay=10,
                            min_decay=0)
    case.add_control(u)

if args.traj!='single':
    fraction_validation = 0.1
    num_training_trajectories = round(case.num_trajectories * (1-fraction_validation))
    case.create_validation_set(num_training_trajectories)

case.get_data()

case.add_test_data(x0_test=x0_test, u_test=u_test, x_test=x_test)

train_data = dict(x0=case.x0.cpu(),
                  steps= case.steps,
                  x=case.x.cpu(),
                  x_noisy=case.x_noisy.cpu(),
                  u = case.u.cpu() if case.u is not None else None,
                  delta_t=delta_t,
                  noise = noise)

if case.x0_val is not None:
    val_data = dict(x0=case.x0_val.cpu(),
                    steps= case.steps,
                    x=case.x_val.cpu(),
                    x_noisy=case.x_noisy_val.cpu(),
                    u = case.u_val.cpu() if case.u_val is not None else None,
                    delta_t=delta_t,
                    noise = noise)

test_data = dict(x0=case.x0_test.cpu(),
                  steps= case.steps,
                  x=case.x_test.cpu(),
                  x_noisy=case.x_noisy_test.cpu(),
                  u = case.u_test.cpu()  if case.u_test is not None else None,
                  delta_t=delta_t,
                  noise=noise)

filepath = f"./dataset/{args.system}_{args.traj}{'_control' if with_control else ''}/noise_{noise:0.4f}/seed_{args.seed}/num_traj_{args.num_traj}/"

makedir(filepath)

save_file(train_data, 'train_data.pkl', path=filepath)
save_file(test_data, 'test_data.pkl', path=filepath)

if case.x0_val is not None:
    save_file(val_data, 'val_data.pkl', path=filepath)

fig, ax = plt.subplots(model.num_states + (0 if case.u is None else model.num_inputs), 1,
                        figsize=(5, 5))

# plot_all_dynamics(ax, case.x_test, color='red', linewidth=1, label='testing trajectories',
#                     variable_name='x')
plot_all_dynamics(ax, case.x, color='black', linewidth=1, label='training trajectories', variable_name='x')
plot_all_dynamics(ax, case.x_noisy, color='black', marker='x',markersize=0.5, linestyle='none', label='training data', variable_name='x')

if case.x_val is not None:
    plot_all_dynamics(ax, case.x_val, color='blue', linewidth=1, label='validation trajectories', variable_name='x')
    plot_all_dynamics(ax, case.x_noisy_val, color='blue', marker='o' ,markersize=0.5,  linestyle='none', label='validation data', variable_name='x')


if case.u is not None:
    if case.u.shape[-1] ==1:
        plot_dynamics(ax[model.num_states], case.u, color='black', linestyle='solid',
                        label='training control inputs',
                        trajectory_index='all')
    else:
        plot_all_dynamics(ax[model.num_states:], case.u, color='black', linestyle='solid',
                        label='training control inputs', variable_name='u')
    

if case.u_val is not None:
    if case.u_val.shape[-1] ==1:
        plot_dynamics(ax[model.num_states], case.u_val, color='blue', linestyle='solid',
                        label='validation control inputs',
                        trajectory_index='all')
    else:
        plot_all_dynamics(ax[model.num_states:], case.u_val, color='blue', linestyle='solid',
                        label='validation control inputs', variable_name='u')
        
if case.u_test is not None:
    if case.u_test.shape[-1] ==1:
        plot_dynamics(ax[model.num_states], case.u_test, color='red', linestyle='solid', linewidth=1.3,
                        label='testing control inputs', trajectory_index='all')
        ax[model.num_states].set_ylabel('$u$')
    else:
        plot_all_dynamics(ax[model.num_states:], case.u_test, color='red', linestyle='solid', linewidth=1.3,
                        label='testing control inputs', variable_name='u')

fig.savefig(f"{filepath}/training_and_testing_trajectories.png", dpi=300)
