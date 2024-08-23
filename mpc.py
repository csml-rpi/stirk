from models import *
from cases import *

class MPC:
    def __init__(self, model, surrogate_model=None, observable=None):
        self.model = model
        self.surrogate_model = surrogate_model
        self.observable = observable
        self.model_dmpc = model.get_do_mpc_model()
        self.use_surrogate = True if self.surrogate_model is not None else False
        if self.use_surrogate:
            self.surrogate_model_dmpc = surrogate_model.get_do_mpc_model()
        self.mpc =  do_mpc.controller.MPC(self.model_dmpc
                                          if not self.use_surrogate
                                          else self.surrogate_model_dmpc)
    def setup_mpc(self, setup_mpc):
        self.mpc.set_param(**setup_mpc)
        self.mpc.settings.supress_ipopt_output()
    def setup_objective(self, Q, R=0):
        self.Q = Q
        self.R = R
        if self.use_surrogate:
            C = np.concatenate([np.eye(self.model.num_states),
                                np.zeros((self.model.num_states,
                                          self.surrogate_model.num_states - self.model.num_states))],
                               axis=1)
            Q = C.T @ self.Q @ C
            lterm = self.surrogate_model.x.T @ Q @ self.surrogate_model.x
            mterm = self.surrogate_model.x.T @ Q @ self.surrogate_model.x

        else:
            lterm = self.model.x.T @ self.Q @ self.model.x
            mterm = self.model.x.T @ self.Q @ self.model.x

        self.mpc.set_objective(lterm=lterm, mterm=mterm)
        self.mpc.set_rterm(u=0)
    def set_bounds_on_input(self, lower, upper):
        # Lower bounds on inputs:
        self.mpc.bounds['lower', '_u', 'u'] = lower
        # Lower bounds on inputs:
        self.mpc.bounds['upper', '_u', 'u'] = upper
    def setup(self):
        self.mpc.setup()
    def set_simulator(self):
        self.simulator = do_mpc.simulator.Simulator(self.model_dmpc)
        self.simulator.set_param(t_step=self.model.delta_t)
        self.simulator.setup()
    def set_initial_conditions(self, x0):
        x0_numpy = x0.cpu().numpy().reshape(-1, 1)
        self.x0 = x0_numpy
        self.simulator.x0 = x0_numpy

        if self.use_surrogate:
            xi0 = self.observable.transform(x0)
            xi0_numpy = xi0.cpu().numpy().reshape(-1, 1)
            self.xi0 = xi0_numpy
            self.mpc.x0 = xi0_numpy
        else:
            self.mpc.x0 = x0_numpy

        self.mpc.set_initial_guess()
    def run(self, steps, path = None):
        if self.use_surrogate:
            xi0 = self.xi0.copy()
        x0 = self.x0.copy()
        for i in range(steps):
            if self.use_surrogate:
                print(f'Finding Optimal control input for step: {i}')
                u0 = self.mpc.make_step(xi0)

                x0 = self.simulator.make_step(u0)
                xi0 = self.observable.transform(torch.from_numpy(x0.T)).cpu().numpy().T
            else:
                u0 = self.mpc.make_step(x0)
                x0 = self.simulator.make_step(u0)

    def get_cost_history(self, steps):
        cum_cost = 0
        cum_cost_list = [0]
        for i in range(steps):
            cum_cost += self.simulator.data['_x'][i].T @ self.Q @ self.simulator.data['_x'][i]
            cum_cost_list.append(cum_cost)
        cum_cost_np = np.array(cum_cost_list)
        return cum_cost_np
    def save(self, label, steps, path):
        cost_history = self.get_cost_history(steps=steps)
        sim_data = self.simulator.data
        history = dict(sim_data=sim_data,
                        cost_history=cost_history)
        save_file(history,f'mpc_results{label}.pkl',path=path)

def set_mpc(mpc, Q, lower_u, upper_u, setup_mpc):
    mpc.setup_mpc(setup_mpc=setup_mpc)
    # mpc.mpc.settings.supress_ipopt_output()
    mpc.setup_objective(Q = Q)

    mpc.set_bounds_on_input(lower = lower_u, upper = upper_u)
    mpc.setup()

def run_mpc(mpc, steps, x0):
    mpc.set_simulator()

    mpc.set_initial_conditions(x0)

    mpc.run(steps=steps)

def set_and_run_mpc(mpc, steps, x0, Q, lower_u, upper_u, setup_mpc):
    mpc.setup_mpc(setup_mpc=setup_mpc)
    # mpc.mpc.settings.supress_ipopt_output()
    mpc.setup_objective(Q = Q)

    mpc.set_bounds_on_input(lower = lower_u, upper = upper_u)
    mpc.setup()

    mpc.set_simulator()

    mpc.set_initial_conditions(x0)

    mpc.run(steps=steps)


def plot_mpc_results(ax, mpc, steps, delta_t=0.05, label=''):
    sim_data = mpc.simulator.data
    t = np.arange(0,delta_t*steps,delta_t)
    x = sim_data['_x']
    u = sim_data['_u']
    J = mpc.get_cost_history(steps=steps)
    label = label + f" Cost: {J[-1]: .4f}"
    num_states = x.shape[-1]
    for i in range(num_states):
        ax[i].plot(t, x[:, i], label=label)
        ax[i].set_ylabel(f'$x_{i+1}$')


    ax[num_states].step(t, u, where='post', label=label)
    ax[num_states].set_ylabel('$u$')
    ax[num_states].set_xlabel('Steps')

    ax[num_states+1].plot(t, J[:-1], label=label)
    ax[num_states+1].set_ylabel('$J$')
    ax[num_states+1].set_xlabel('Steps')
    # ax[1, 1].text(0.3, 0.5, f'', transform=ax[1, 1].transAxes)

