from dmd_algorithms import *
from edmd_rollout import *
from models import *
import scienceplots
from cases import *
import argparse
from pathlib import Path
import datetime
plt.style.use(['science', 'ieee', 'no-latex'])
import config

parser = argparse.ArgumentParser(description='training van der pol multiple trajectories')

today = datetime.datetime.now()
formatted_date = today.strftime("%B_%d_%Y")
name = f"{formatted_date}/" # Folder name

strategy = "dissipative" # "standard"  
std_init_A = 0.01 
std_init_B = 0.01
std_init_P = 0.01
std_init_Q = 0.01
std_init_sigma = 0.01

device_id = 0
seed = 100

traj = 'multi'
with_control = False
noise_level =  8

plot_history_after_epoch = 100
compare_after_epoch = 1
num_epochs = 5000
batch_size = 1000
learning_rate = 0.01

scheduler_mode = "stepLR"
switch_learning_rate_after_epoch = 100
learning_rate_decay = 1.0
max_lr = 0.1
cycliclr_mode = 'triangular2'

geometric_progression = True
switch_roll_out_after_epoch = 100
start_roll_out_length = 99
roll_out_increase = 2
max_roll_out_length = 99


first_optimizer_is_Adam = True
switch_optimizer_after_epoch = 6000
cycle_optimizer = False

learning_rate_lbfgs = 0.01
batch_size_lbfgs = None 
reconstruction_loss_factor = 0.0
show = True
epoch_after_reduce_plateau = np.inf
early_stopping_patience = 1000
early_stopping_threshold = 0.0001

parser.add_argument('--strategy', type=str, default=strategy, help='Koopman Model Type')
parser.add_argument('--scheduler-mode', type=str, default=scheduler_mode, help="learning rate scheduler")
parser.add_argument('--start-roll-out-length', type=int, default=start_roll_out_length, help="initial rollout length")
parser.add_argument('--switch-roll-out-after-epoch', type=int, default=switch_roll_out_after_epoch, help="switch roll out after epoch")
parser.add_argument('--learning-rate', type=float, default=learning_rate, help="Learning Rate")
parser.add_argument('--max-lr', type=float, default=max_lr, help="Max Learning Rate")
parser.add_argument('--device-id', type=int, default=device_id, help='Device ID')
parser.add_argument('--name', type=str, default=name, help='Naming tuning parameters')
parser.add_argument('--seed', type=int, default=seed, help="Random Seed")
parser.add_argument('--traj', type=str, default=traj, help='Different strategies')
parser.add_argument('--control', type=lambda x: (str(x).lower() == 'true'), default=with_control, help='With control or without control')
parser.add_argument('--noise-level', type=int, default = noise_level, help='0-9 : 0 represents 1e-3 and 9 represents 1e-1')

parser.add_argument('--plot-history-after-epoch', type=int, default=plot_history_after_epoch, help="Epoch after which to plot history")
parser.add_argument('--compare-after-epoch', type=int, default=compare_after_epoch, help="Epoch after which to compare models")
parser.add_argument('--num-epochs', type=int, default=num_epochs, help="Number of epochs to run")
parser.add_argument('--batch-size', type=int, default=batch_size, help="Batch size for training")
parser.add_argument('--geometric-progression', type=lambda x: (str(x).lower() == 'true'), default=geometric_progression, help="Use geometric progression for rollout increase")
parser.add_argument('--roll-out-increase', type=int, default=roll_out_increase, help="Factor by which rollout length increases")
parser.add_argument('--max-roll-out-length', type=int, default=max_roll_out_length, help="Maximum rollout length")
parser.add_argument('--learning-rate-decay', type=float, default=learning_rate_decay, help="Factor by which learning rate decays")
parser.add_argument('--switch-learning-rate-after-epoch', type=lambda x: (str(x).lower() == 'true'), default=switch_learning_rate_after_epoch, help="Switch learning rate after specified epoch")
parser.add_argument('--first-optimizer-is-Adam',type=lambda x: (str(x).lower() == 'true'), default=first_optimizer_is_Adam, help="Use Adam optimizer first")
parser.add_argument('--switch-optimizer-after-epoch', type=int, default=switch_optimizer_after_epoch, help="Epoch after which to switch optimizer")
parser.add_argument('--learning-rate-lbfgs', type=float_or_none, default=learning_rate_lbfgs, help="Learning rate for L-BFGS optimizer")
parser.add_argument('--batch-size-lbfgs', type=int_or_none, default=batch_size_lbfgs, help="Batch size for L-BFGS optimizer")
parser.add_argument('--reconstruction-loss-factor', type=float, default=reconstruction_loss_factor, help="Factor for reconstruction loss")
parser.add_argument('--show', type=lambda x: (str(x).lower() == 'true'), default=show, help="Show plots")
parser.add_argument('--epoch-after-reduce-plateau', type=int, default=epoch_after_reduce_plateau, help="Epoch after which to reduce learning rate plateau")
parser.add_argument('--early-stopping-patience', type=int, default=early_stopping_patience, help="Patience")
parser.add_argument('--cycle-optimizer', type=lambda x: (str(x).lower() == 'true'), default=cycle_optimizer, help="Cycle optimizer")
parser.add_argument('--std-init-A', type=float, default=std_init_A, help="Standard Deviation for init A")
parser.add_argument('--std-init-B', type=float, default=std_init_B, help="Standard Deviation for init B")
parser.add_argument('--std-init-P', type=float, default=std_init_P, help="Standard Deviation for init P")
parser.add_argument('--std-init-Q', type=float, default=std_init_Q, help="Standard Deviation for init Q")
parser.add_argument('--std-init-sigma', type=float, default=std_init_sigma, help="Standard Deviation for init sigma")
parser.add_argument('--cycliclr-mode', type=str, default=cycliclr_mode, help='triangular or triangular2')
parser.add_argument('--early-stopping-threshold', type=float, default=early_stopping_threshold, help='Early stopping threshold')



args = parser.parse_args()
config.update_device(device_id=args.device_id)

torch.random.manual_seed(123)

mu = 1.0
delta_t = 0.1
steps = 100

model = VanDerPolDiscreteControl(mu=mu, delta_t=delta_t)

errors = np.array([])
test_errors = np.array([])
noise_values = np.logspace(-3,-1,10)
with_control = args.control
noise = noise_values[args.noise_level]

case = Case(gt_model=model,
            noise=noise,
            steps=steps,
            strategy=args.strategy)

train_data, test_data, val_data = load_data_files(system='vdp', 
                                                with_control= args.control,
                                                traj = args.traj, 
                                                noise_level= args.noise_level,
                                                seed = args.seed)

case.load_data(train_data=train_data, test_data=test_data, val_data=val_data)

observable = TimedelayImproved(gt_model=case.gt_model,
                               order=4,
                               delta_t=case.gt_model.delta_t)
case.add_observable(observable)

if case.strategy == "standard":
    koopman_model = StandardKoopmanModel(num_states=case.num_states,
                                         num_inputs=case.num_inputs,
                                         delta_t=case.gt_model.delta_t,
                                         order=case.order,
                                         lifted_states=None,
                                         std_init_B=args.std_init_B,
                                         std_init_A=args.std_init_A)
elif case.strategy == "dissipative":
    koopman_model = DissipativeKoopmanModel(num_states=case.num_states,
                                            num_inputs=case.num_inputs,
                                            delta_t=case.gt_model.delta_t,
                                            order=case.order,
                                            lifted_states=None,
                                            std_init_P= args.std_init_P,
                                            std_init_B= args.std_init_B,
                                            std_init_Q= args.std_init_Q,
                                            std_init_sigma= args.std_init_sigma)
else:
    koopman_model = DissipativeKoopmanModel2(num_states=case.num_states,
                                             num_inputs=case.num_inputs,
                                             delta_t=case.gt_model.delta_t,
                                             order=case.order,
                                             lifted_states=None)

case.create_filepath(trial=True, kw=f"{args.name}seed_{args.seed}")
case.save_gt_info()
case.save_observable_info()
case.save_test_data()

rollout_edmd = RolloutEDMD(observable=observable,
                           koopman_model=koopman_model,
                           order=case.order,
                           delta_t=case.gt_model.delta_t)

rollout_edmd.setup_comparison(x_true=case.x, x_noisy=case.x_noisy, x0_true=case.x0, steps=case.steps, u_true=case.u)




rollout_edmd.add_validation_data(x0 = case.x0_val,
                                 x = case.x_val,
                                 x_noisy = case.x_noisy_val,
                                 u= case.u_val,
                                 )
rollout_edmd.add_test_data(x_true=case.x_test,
                           x_noisy=case.x_noisy_test,
                           u_true=case.u_test,
                           x0_true=case.x0_test)

rollout_edmd.setup(
    plot_history_after_epoch=args.plot_history_after_epoch,
    compare_after_epoch=args.compare_after_epoch,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    switch_roll_out_after_epoch=args.switch_roll_out_after_epoch,
    geometric_progression=args.geometric_progression,
    roll_out_increase=args.roll_out_increase,
    max_roll_out_length=args.max_roll_out_length,
    learning_rate_decay=args.learning_rate_decay,
    switch_learning_rate_after_epoch=args.switch_learning_rate_after_epoch,
    first_optimizer_is_Adam=args.first_optimizer_is_Adam,
    switch_optimizer_after_epoch=args.switch_optimizer_after_epoch,
    learning_rate_lbfgs=args.learning_rate_lbfgs,
    batch_size_lbfgs=args.batch_size_lbfgs,
    reconstruction_loss_factor=args.reconstruction_loss_factor,
    show=args.show,
    epoch_after_reduce_plateau=args.epoch_after_reduce_plateau,
    early_stopping_patience= args.early_stopping_patience,
    cycle_optimizer=args.cycle_optimizer,
    early_stopping_threshold=args.early_stopping_threshold
)
rollout_edmd.filepath = case.filepath

if args.scheduler_mode == "cyclicLR":
    config_cyclic_LR = dict(cycle_momentum=False,
                           max_lr= args.max_lr,
                           base_lr=args.learning_rate,
                           step_size_up = 300,
                           step_size_down = 300,
                           mode=args.cycliclr_mode)
else:
    config_cyclic_LR = None
reduce_on_plateau = False if args.epoch_after_reduce_plateau == np.inf else True  
rollout_edmd.set_optimizers(scheduler_mode=args.scheduler_mode, config_cyclic_lr=config_cyclic_LR, reduce_on_plateau=reduce_on_plateau)
create_log(case, rollout_edmd)
start_time = time.time()
A,B, error_dict = rollout_edmd.fit(case.x_noisy, u=case.u, strategy=case.strategy, start_roll_out_length=args.start_roll_out_length)
end_time = time.time()
time_taken = end_time -start_time

print(f"Time taken {time_taken}s")
log_file = os.path.join(case.filepath, "log.txt")
with open(log_file, 'a') as f:
    f.write(f"\nTime taken {time_taken}s \n seed {args.seed}")
