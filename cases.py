import config
from solve_dynamics import *
class Case:
    def __init__(self, gt_model, noise, steps, x0=None, order=None, strategy='dmd', u = None):
        self.initial_weights = None
        self.gt_model = gt_model
        self.steps = steps
        self.noise = noise
        self.x0 = x0.to(device=config.device, dtype=config.dtype) if x0 is not None else None
        self.order = order
        self.num_states = self.x0.shape[-1] if x0 is not None else None
        self.num_trajectories = self.x0.shape[0] if x0 is not None else None
        self.u = u.to(device=config.device, dtype=config.dtype) if u is not None else None
        self.num_inputs = self.u.shape[-1] if u is not None else None
        self.params = None
        self.filepath = None
        self.strategy = strategy
        self.x_noisy = None
        self.x = None
        self.observable = None
        self.observable_info = None
        self.x0_test = None
        self.u_test = None
        self.x_test = None
        self.x_noisy_test = None
        self.x0_val = None
        self.u_val = None
        self.x_val = None
        self.x_noisy_val = None 
    
    def load_data(self, train_data, test_data =None, val_data = None):
        self.x0 = train_data['x0'].to(device=config.device, dtype=config.dtype)
        self.num_states = self.x0.shape[-1]
        self.num_trajectories = self.x0.shape[0]
        self.u = train_data['u'].to(device=config.device, dtype=config.dtype) if train_data['u'] is not None else None
        self.num_inputs = self.u.shape[-1] if self.u is not None else None
        self.x = train_data['x'].to(device=config.device, dtype=config.dtype)
        
        self.x_noisy = train_data['x_noisy'].to(device=config.device, dtype=config.dtype)

        if test_data is not None:
            self.x0_test = test_data['x0'].to(device=config.device, dtype=config.dtype)

            self.x_test = test_data['x'].to(device=config.device, dtype=config.dtype)
        
            self.x_noisy_test = test_data['x_noisy'].to(device=config.device, dtype=config.dtype)
        
            self.u_test = test_data['u'].to(device=config.device, dtype=config.dtype) if test_data['u'] is not None else None

        if val_data is not None:
            self.x0_val = val_data['x0'].to(device=config.device, dtype=config.dtype)

            self.x_val = val_data['x'].to(device=config.device, dtype=config.dtype)
        
            self.x_noisy_val = val_data['x_noisy'].to(device=config.device, dtype=config.dtype)
        
            self.u_val = val_data['u'].to(device=config.device, dtype=config.dtype) if test_data['u'] is not None else None



    def set_initial_weights(self, filepath):
        self.initial_weights = load_file("init_weights.pkl", filepath)
    def add_control(self, u):
        self.u = u.to(device=config.device, dtype=config.dtype)
        self.num_inputs = self.u.shape[-1]
    def add_observable(self,observable):
        self.observable = observable
        self.order = observable.order
        self.observable_info = dict(observable=observable)
    def save_observable_info(self, filepath=None):
        save_file(self.observable_info, 'observable_info.pkl', path=filepath if filepath is not None else self.filepath)
    def set_params(self, params):
        self.params = params

    def create_validation_set(self, num_training_traj):
        # self.x0, self.x0_val = split_tensor(self.x0, num_training_traj)
        # self.u , self.u_val = split_tensor(self.u, num_training_traj)
        if self.u is not None:
            (self.x0, self.u), (self.x0_val, self.u_val) = split_tensors((self.x0, self.u), num_training_traj)
        else:
            self.x0 , self.x0_val = split_tensors(self.x0, num_training_traj)
            self.u_val = None
    def get_data(self):
        solver = Simulator(model=self.gt_model, steps=self.steps, x0=self.x0)
        self.x = solver.rollout(self.u)
        # self.x_noisy = self.x + torch.randn(self.x.shape).to(device=self.x.device, dtype=self.x.dtype) * self.noise  #* torch.mean(self.x,dim=(0,1))
        
        gaussian_noise = torch.normal(mean=0, std=1.0, size=self.x.shape).to(device=self.x.device, dtype = self.x.dtype)
        scaled_noise = gaussian_noise * (self.x.abs() * self.noise)
        self.x_noisy = self.x + scaled_noise

        if self.x0_val is not None:
            solver_val = Simulator(model=self.gt_model, steps=self.steps, x0= self.x0_val)
            self.x_val = solver_val.rollout(self.u_val)
        
        
            # self.x_noisy_val = self.x_val + torch.randn(self.x_val.shape).to(device=self.x_val.device,
            #                                       dtype=self.x_val.dtype) * self.noise #* torch.mean(self.x_val,dim=(0,1))
            gaussian_noise_val = torch.normal(mean=0, std=1.0, size=self.x_val.shape).to(device=self.x_val.device, dtype = self.x_val.dtype)
            scaled_noise_val = gaussian_noise_val * (self.x_val.abs() * self.noise)
            self.x_noisy_val = self.x_val + scaled_noise_val

    def add_test_data(self, x0_test, u_test =None, x_test=None):
        self.x0_test = x0_test.to(device=config.device,dtype=config.dtype)
        self.u_test = u_test if u_test is None else u_test.to(device=config.device,dtype=config.dtype)
        if x_test is None:
            solver = Simulator(model=self.gt_model, steps=self.steps, x0=x0_test)
            self.x_test = solver.rollout(self.u_test).to(device=config.device,dtype=config.dtype)
        else:
            self.x_test = x_test.to(device=config.device,dtype=config.dtype)
        self.x_noisy_test = self.x_test + torch.randn(self.x_test.shape).to(device=self.x_test.device,
                                                             dtype=self.x_test.dtype) * self.noise  # * torch.mean(x,dim=(0,1))
    def save_test_data(self, filepath=None):
        if self.x_noisy_test is not None and self.x_test is not None:
            test_info = dict(x0=self.x0_test.cpu(),
                           u=self.u_test if self.u_test is None else self.u_test.cpu(),
                           gt_data=self.x_test.cpu(),
                           data=self.x_noisy_test.cpu())
            if self.filepath is not None:
                save_file(test_info, 'test_info.pkl', path=filepath if filepath is not None else self.filepath)
            else:
                print("Please create filepath first. Cannot save info.")
        else:
            print("Please run the simulation first.")
    def create_filepath(self, trial=False, kw=""):
        traj_type = 'single_traj' if self.num_trajectories == 1 else 'multi_traj'
        control_type = 'control' if self.u is not None else 'no_control'
        noise_level = f"noise_{self.noise: 0.4f}"
        observable_name = f"{self.observable.name}"
        filepath = f"./{self.gt_model.name}/{traj_type}/{control_type}/{self.strategy}/{observable_name}/{noise_level}/"
        if trial:
            filepath = create_next_trial_folder(filepath, kw=kw)
        else:
            filepath = create_next_final_folder(filepath)
        self.filepath = filepath

    def save_gt_info(self, filepath = None):
        if self.x_noisy is not None and self.x is not None:
            gt_info = dict(gt_model=self.gt_model.cpu(),
                            x0=self.x0.cpu(),
                           u=self.u if self.u is None else self.u.cpu(),
                           gt_data=self.x.cpu(),
                           data=self.x_noisy.cpu(),
                           noise=self.noise,
                           delta_t=self.gt_model.delta_t,
                           strategy=self.strategy)
            if self.filepath is not None:
                save_file(gt_info, 'gt_info.pkl', path= filepath if filepath is not None else self.filepath)
            else:
                print("Please create filepath first. Cannot save info.")
        else:
            print("Please run the simulation first.")
    def create_log(self):
        log = f"""
        System = {self.gt_model.name},
        Steps = {self.steps}, delta_t ={self.gt_model.delta_t},
        Number of trajectories: {self.num_trajectories},
        Noise level = {self.noise},
        Polyflow order = {self.order},
        ___________________
        Hyperparameters:
        {disp_dict(self.params)}
        """
        log_file = os.path.join(self.filepath, "log.txt")
        with open(log_file, 'w') as f:
            f.write(log)

        save_file(self.params, 'hyperparameters.pkl', path=self.filepath)

def create_log(case, regressor):
    log = f"""
System = {case.gt_model.name},
Steps = {case.steps}, delta_t ={case.gt_model.delta_t},
Number of trajectories: {case.num_trajectories},
Noise level = {case.noise},
Polyflow order = {case.order},
Strategy = {case.strategy}
___________________
Hyperparameters:
{disp_dict(regressor.params)}
    """
    log_file = os.path.join(case.filepath, "log.txt")
    with open(log_file, 'w') as f:
        f.write(log)
    save_file(regressor.params, 'hyperparameters.pkl', path=case.filepath)

class Result:
    def __init__(self, filepath):
        self.filepath = filepath
        self.gt_info = load_file('gt_info.pkl', filepath)
        self.params = load_file('hyperparameters.pkl', filepath)
        self.history = load_file('training_history.pkl', filepath)
        self.observable_info = load_file("observable_info.pkl", filepath)
        self.error = load_file("trajectory_error_training.pkl", filepath)
        self.test_error = load_file("trajectory_error_test.pkl", filepath)
        self.trajectory_info_training = load_file("trajectory_info_training.pkl", filepath)
        self.trajectory_info_test = load_file('trajectory_info_test.pkl', filepath)
        # if self.history is not None:
        #     self.error = self.history['error_history']

        self.model_params = load_file("model_params.pkl", filepath)
        self.mpc_results = load_file("mpc_resultswith_surrogate.pkl", filepath)
        self.mpc_results_baseline = load_file("mpc_resultswitout_surrogate.pkl", filepath)
        self.test_info = load_file("test_info.pkl", filepath)


        if self.observable_info is not None:
            self.observable = self.observable_info["observable"]
            self.order = self.observable.order
        if self.model_params is not None:
            self.A = self.model_params["A"]
            self.B = self.model_params["B"]
        if self.gt_info is not None:
            self.gt_model = self.gt_info["gt_model"]
            self.x0 = self.gt_info["x0"]
            self.u = self.gt_info["u"]
            self.x = self.gt_info["gt_data"]
            self.x_noisy = self.gt_info["data"]
            self.noise = self.gt_info["noise"]
            self.delta_t = self.gt_info["delta_t"]
            self.strategy = self.gt_info["strategy"]
            self.steps = self.x.shape[1]
        if self.test_info is not None:
            self.x0_test = self.test_info["x0"]
            self.u_test = self.test_info["u"]
            self.x_test = self.test_info["gt_data"]
            self.x_noisy_test = self.test_info["data"]
        if self.mpc_results is not None and self.mpc_results_baseline is not None:
            self.baseline_cost = self.mpc_results_baseline["cost_history"][-1]
            self.mpc_cost = self.mpc_results["cost_history"][-1]

        if self.trajectory_info_training is not None:
            self.x_pred = self.trajectory_info_training['x_pred']
            
        if self.trajectory_info_test is not None:
            self.x_pred_test = self.trajectory_info_test['x_pred']

        checkpoint_path = os.path.join(self.filepath, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            self.checkpoint = torch.load(checkpoint_path)
            self.koopman_model = self.checkpoint["koopman_model"]
            self.koopman_model.load_state_dict(self.checkpoint["koopman_model_state_dict"])
            self.last_epoch = self.checkpoint["epoch"]
            self.last_roll_out_length = self.checkpoint['roll_out_length']
            self.optimizer_1_state_dict = self.checkpoint['optimizer_1_state_dict']
            self.optimizer_2_state_dict = self.checkpoint['optimizer_2_state_dict']
            self.scheduler_state_dict = self.checkpoint['scheduler_state_dict']

class Results:
    def __init__(self, filepath, folder_name="final"):
        self.results = []
        if os.path.exists(filepath) and os.path.isdir(filepath):
            with os.scandir(filepath) as it:
                for entry in it:
                    if entry.is_dir():
                        final_dir_path = os.path.join(entry.path, folder_name)
                        if os.path.isdir(final_dir_path):
                            self.results.append(Result(filepath=final_dir_path))
        if len(self.results) != 0 :
            self.results.sort(key=lambda result: result.noise)
            self.noises = [result.noise for result in self.results]
            self.errors = [result.error for result in self.results]
            if self.results[0].mpc_results is not None:
                self.mpc_costs = [result.mpc_cost for result in self.results]
            self.strategy = self.results[0].strategy
            self.observable_name = self.results[0].observable.name
            self.order = self.results[0].order

if __name__ == '__main__':
    pass
