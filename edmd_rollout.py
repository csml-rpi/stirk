import numpy as np
import torch

from time_delay import *
from models import *
from utils import *
import config
from dmd_algorithms import *
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


class KoopmanModelOld(nn.Module):
    def __init__(self, order,
                 num_states,
                 num_inputs=None,
                 strategy="standard",
                 delta_t=0.1,
                 initial_weights=None,
                 control_strategy="constant"):
        super().__init__()
        self.order = order
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.delta_t = delta_t
        self.strategy = strategy
        self.control_strategy = control_strategy
        if strategy == "standard":
            if initial_weights is None:
                A_init = torch.randn(size=(self.order * self.num_states,
                                           self.order * self.num_states),
                                     device=config.device,
                                     dtype=config.dtype) * 0.1
            else:
                A_init = initial_weights["A"]
            self.register_parameter("A", torch.nn.Parameter(A_init, requires_grad=True))

            # self.A = torch.nn.Parameter(A_init, requires_grad=True)
        elif strategy == "dissipative":
            if initial_weights is None:
                P_init = torch.randn(size=(self.order * self.num_states,
                                           self.order * self.num_states),
                                     device=config.device,
                                     dtype=config.dtype) * 0.2
                Q_init = torch.randn(size=(self.order * self.num_states,
                                           self.order * self.num_states),
                                     device=config.device,
                                     dtype=config.dtype) * 0.2
                sigma_init = torch.randn(size=(self.order * self.num_states,),
                                         device=config.device,
                                         dtype=config.dtype) * 0.2

            else:
                P_init = initial_weights["P"]
                Q_init = initial_weights["Q"]
                sigma_init = initial_weights["sigma"]

            self.register_parameter("P", torch.nn.Parameter(P_init, requires_grad=True))
            self.register_parameter("Q", torch.nn.Parameter(Q_init, requires_grad=True))
            self.register_parameter("sigma", torch.nn.Parameter(sigma_init, requires_grad=True))

        elif strategy == "dissipative_2":
            P_init = torch.randn(size=(self.order * self.num_states,
                                       self.order * self.num_states),
                                 device=config.device,
                                 dtype=config.dtype) * 0.2
            # P_inv_init = torch.randn(size=(self.order * self.num_states,
            #                           self.order * self.num_states),
            #                     device=config.device,
            #                     dtype=config.dtype) * 0.2
            Q_init = torch.randn(size=(self.order * self.num_states,
                                       self.order * self.num_states),
                                 device=config.device,
                                 dtype=config.dtype) * 0.2
            sigma_init = torch.randn(size=(self.order * self.num_states,),
                                     device=config.device,
                                     dtype=config.dtype) * 0.2

            self.register_parameter("P", torch.nn.Parameter(P_init, requires_grad=True))
            self.register_parameter("P_inv", torch.nn.Parameter(torch.linalg.pinv(P_init), requires_grad=True))

            self.register_parameter("Q", torch.nn.Parameter(Q_init, requires_grad=True))
            self.register_parameter("sigma", torch.nn.Parameter(sigma_init, requires_grad=True))

        if self.num_inputs is not None:
            if initial_weights is not None:
                B_init = initial_weights["B"]
                self.register_parameter("B", torch.nn.Parameter(B_init, requires_grad=True))
            else:
                if control_strategy == 'constant':
                    B_init = torch.rand(size=(self.order * self.num_states, self.num_inputs),
                                        device=config.device,
                                        dtype=config.dtype) * 0.1
                    self.register_parameter("B", torch.nn.Parameter(B_init, requires_grad=True))
                elif control_strategy == 'bilinear':
                    B_init = torch.rand(
                        size=(self.order * self.num_states, self.order * self.num_states * self.num_inputs),
                        device=config.device, dtype=config.dtype) * 0.1
                    self.register_parameter("B", torch.nn.Parameter(B_init, requires_grad=True))
        else:
            self.B = None

    def forward(self, x, u=None):
        if self.B is None and u is None:
            output = x @ self.A.t() if x.ndim == 2 else self.A @ x.t()
        else:
            if self.control_strategy == "constant":
                output = x @ self.A.t() + u @ self.B.t() if x.ndim == 2 else self.A @ x.t() + self.B @ u.t()
            elif self.control_strategy == "bilinear":
                if self.num_inputs == 2:
                    output = x @ self.A.t()
        return output

    def make_step(self, x0, u, step):
        x = torch.matmul(x0, torch.matrix_power(self.A, step).T) \
            + convolution(A=self.A, B=self.B, step=step, x0=x0, u=u)
        return x

    def rollout(self, x0, u, step):
        s1 = time.time()
        outputs = [x0]
        for i in range(step):
            if u is not None:
                x_next = outputs[-1] @ self.A.t() + u[:, i, :] @ self.B.t()  # self.make_step(outputs[-1],u,step)#
            else:
                x_next = outputs[-1] @ self.A.t()
            outputs.append(x_next)
        result = torch.stack(outputs, dim=1)
        s2 = time.time() - s1
        return result

    @property
    def A(self):
        C = torch.diag(-self.sigma ** 2)
        # A_hat = self.Q - self.Q.t() + C
        # var = torch.linalg.matrix_exp(self.P @ A_hat @ torch.linalg.pinv(self.P) * self.delta_t)
        A_hat = torch.linalg.matrix_exp((self.Q - self.Q.t() + C) * self.delta_t)
        if self.strategy == "dissipative_2":
            var = self.P @ A_hat @ self.P_inv
        else:
            var = self.P @ A_hat @ torch.linalg.pinv(self.P)
            # var = A_hat
        return var

    def loss_for_inverse(self):
        if self.strategy == "dissipative_2":
            loss = torch.norm(self.P @ self.P_inv - torch.eye(self.P.shape[0]).to(device=config.device, dtype=config.dtype),
                              'fro') ** 2
            return loss
        return 0


class KoopmanModel(nn.Module):
    def __init__(self, num_states,
                 num_inputs=None,
                 delta_t=0.1,
                 lifted_states=None,
                 order=None):
        super().__init__()

        self.num_states = num_states
        self.num_inputs = num_inputs
        self.delta_t = delta_t

        if lifted_states is None and order is not None:
            self.lifted_states = order * self.num_states
        elif lifted_states is not None and order is None:
            self.lifted_states = lifted_states

    def forward(self, x, u=None):
        # if self.num_inputs is None:
        #     output = x @ self.A.t() if x.ndim == 2 else self.A @ x.t()
        # else:
        #     output = x @ self.A.t() + u @ self.B.t() if x.ndim == 2 else self.A @ x.t() + self.B @ u.t()
        output = self._next_state(x, u)

        return output

    def rollout(self, x0, step, u=None):
        outputs = [x0]
        for i in range(step - 1):
            if u is not None:
                x_next = self._next_state(outputs[-1], u[:, i, :])
            else:
                x_next = self._next_state(outputs[-1])
            outputs.append(x_next)
        result = torch.stack(outputs, dim=1)
        return result

    def _next_state(self, x, u=None):
        pass

    def loss_for_inverse(self):
        return 0


class StandardKoopmanModel(KoopmanModel):
    def __init__(self, num_states,
                 num_inputs=None,
                 delta_t=0.1,
                 lifted_states=None,
                 order=None,
                 initial_weights=None,
                 A=None,
                 std_init_A=0.01,
                 std_init_B=0.01):
        super().__init__(num_states, num_inputs, delta_t, lifted_states, order)
        if A is not None:
            self.A = A
        else:
            if initial_weights is None:
                A_init = torch.randn(size=(self.lifted_states,
                                           self.lifted_states),
                                     device=config.device,
                                     dtype=config.dtype) * std_init_A
            else:
                A_init = initial_weights["A"]

            self.register_parameter("A", torch.nn.Parameter(A_init, requires_grad=True))

        if self.num_inputs is not None:
            if initial_weights is not None:
                B_init = initial_weights["B"]
            else:
                B_init = torch.rand(size=(self.lifted_states, self.num_inputs),
                                    device=config.device,
                                    dtype=config.dtype) * std_init_B
            self.register_parameter("B", torch.nn.Parameter(B_init, requires_grad=True))
        else:
            self.B = None

    def _next_state(self, x, u=None):
        if self.num_inputs is None:
            x_next = x @ self.A.t()
        else:
            x_next = x @ self.A.t() + u @ self.B.t()
        return x_next


class DissipativeKoopmanModel(KoopmanModel):
    def __init__(self, num_states,
                 num_inputs=None,
                 delta_t=0.1,
                 lifted_states=None,
                 order=None,
                 std_init_P=0.01,
                 std_init_Q=0.01,
                 std_init_sigma=0.01,
                 std_init_B=0.01):
        super().__init__(num_states, num_inputs, delta_t, lifted_states, order)
        P_init = torch.randn(size=(self.lifted_states,
                                   self.lifted_states),
                             device=config.device,
                             dtype=config.dtype) * std_init_P
        Q_init = torch.randn(size=(self.lifted_states,
                                   self.lifted_states),
                             device=config.device,
                             dtype=config.dtype) * std_init_Q
        sigma_init = torch.randn(size=(self.lifted_states,),
                                 device=config.device,
                                 dtype=config.dtype) * std_init_sigma
        self.register_parameter("P", torch.nn.Parameter(P_init, requires_grad=True))
        self.register_parameter("Q", torch.nn.Parameter(Q_init, requires_grad=True))
        self.register_parameter("sigma", torch.nn.Parameter(sigma_init, requires_grad=True))
        if self.num_inputs is not None:
            B_init = torch.rand(size=(self.lifted_states, self.num_inputs),
                                device=config.device,
                                dtype=config.dtype) * std_init_B
            self.register_parameter("B", torch.nn.Parameter(B_init, requires_grad=True))
        else:
            self.B = None

    def _next_state(self, x, u=None):
        if self.num_inputs is None:
            x_next = x @ self.A.t()
        else:
            x_next = x @ self.A.t() + u @ self.B.t()
        return x_next

    @property
    def A(self):
        C = torch.diag(-self.sigma ** 2)
        A_hat = torch.linalg.matrix_exp((self.Q - self.Q.t() + C) * self.delta_t)
        var = self.P @ A_hat @ torch.linalg.pinv(self.P)

        # Q, _ = torch.linalg.qr(self.Q)
        # D = torch.diag(torch.tanh(self.sigma))
        # var = Q @ D @ Q.T
        # print(torch.abs(torch.linalg.eigvals(A)))
        return var


class HamiltonianKoopmanModel(KoopmanModel):
    def __init__(self, num_states,
                 num_inputs=None,
                 delta_t=0.1,
                 lifted_states=None,
                 order=None,
                 std_init_P=0.1,
                 std_init_Q=0.1,
                 std_init_B=0.01):
        super().__init__(num_states, num_inputs, delta_t, lifted_states, order)
        P_init = torch.randn(size=(self.lifted_states,
                                   self.lifted_states),
                             device=config.device,
                             dtype=config.dtype) * std_init_P
        Q_init = torch.randn(size=(self.lifted_states,
                                   self.lifted_states),
                             device=config.device,
                             dtype=config.dtype) * std_init_Q

        self.register_parameter("P", torch.nn.Parameter(P_init, requires_grad=True))
        self.register_parameter("Q", torch.nn.Parameter(Q_init, requires_grad=True))
        self.register_parameter("P_inv", torch.nn.Parameter(P_init, requires_grad=True))
        if self.num_inputs is not None:
            B_init = torch.rand(size=(self.lifted_states, self.num_inputs),device=config.device,
                        dtype=config.dtype) * std_init_B
            self.register_parameter("B", torch.nn.Parameter(B_init, requires_grad=True))
        else:
            self.B = None

    def _next_state(self, x, u=None):
        if self.num_inputs is None:
            x_next = x @ self.A.t()
        else:
            x_next = x @ self.A.t() + u @ self.B.t()
        return x_next

    @property
    def A(self):
        A_hat = torch.linalg.matrix_exp((self.Q - self.Q.t()) * self.delta_t)
        # A_hat = matrix_exp_taylor(A= (self.Q - self.Q.t()) * self.delta_t, terms=10)
        # var = A_hat 
        # Q, _ = torch.linalg.qr(self.Q)

        var = self.P @ A_hat @ torch.linalg.pinv(self.P)
        # var = self.P @ A_hat @ self.P_inv
        # var = Q 
        
        return var
    
    def loss_for_inverse(self):
        loss = torch.mean(torch.square(self.P @ self.P_inv - torch.eye(self.P.shape[0]).to(device=config.device, dtype=config.dtype)))
        return loss

class DissipativeKoopmanModel2(KoopmanModel):
    def __init__(self, num_states,
                 num_inputs=None,
                 delta_t=0.1,
                 lifted_states=None,
                 order=None):
        super().__init__(num_states, num_inputs, delta_t, lifted_states, order)
        P_init = torch.randn(size=(self.lifted_states,
                                   self.lifted_states),
                             device=config.device,
                             dtype=config.dtype) * 0.2
        Q_init = torch.randn(size=(self.lifted_states,
                                   self.lifted_states),
                             device=config.device,
                             dtype=config.dtype) * 0.2
        sigma_init = torch.randn(size=(self.lifted_states,),
                                 device=config.device,
                                 dtype=config.dtype) * 0.2

        self.register_parameter("P", torch.nn.Parameter(P_init, requires_grad=True))
        self.register_parameter("P_inv", torch.nn.Parameter(torch.linalg.pinv(P_init), requires_grad=True))

        self.register_parameter("Q", torch.nn.Parameter(Q_init, requires_grad=True))
        self.register_parameter("sigma", torch.nn.Parameter(sigma_init, requires_grad=True))

        if self.num_inputs is not None:
            B_init = torch.rand(size=(self.lifted_states, self.num_inputs),
                                device=config.device,
                                dtype=config.dtype) * 0.1
            self.register_parameter("B", torch.nn.Parameter(B_init, requires_grad=True))
        else:
            self.B = None

    def _next_state(self, x, u=None):
        if self.num_inputs is None:
            x_next = x @ self.A.t()
        else:
            x_next = x @ self.A.t() + u @ self.B.t()
        return x_next

    @property
    def A(self):
        C = torch.diag(-self.sigma ** 2)
        A_hat = torch.linalg.matrix_exp((self.Q - self.Q.t() + C) * self.delta_t)
        var = self.P @ A_hat @ self.P_inv
        return var

    def loss_for_inverse(self):
        loss = torch.norm(self.P @ self.P_inv - torch.eye(self.P.shape[0]).to(device=config.device, dtype=config.dtype), 'fro') ** 2
        return loss


def convolution(A, B, step, x0, u=None):
    if u is None:
        return 0
    s = torch.zeros_like(x0, device=u.device, dtype=u.dtype)
    for j in range(step):
        conv = torch.matmul(torch.matmul(u[:, j, :], B.t()), torch.matrix_power(A, step - j - 1).t())
        s += conv
    return s


def plot_history(loss_history,
                 error_history,
                 roll_out_length_history,
                 use_optimizer_history,
                 learning_rate_history,
                 testing_error_history=None,
                 val_error_history=None,
                 filepath=None):
    loss_history = np.array(loss_history)
    error_history = np.array(error_history)
    roll_out_length_history = np.array(roll_out_length_history)
    use_optimizer_history = np.array(use_optimizer_history)
    learning_rate_history = np.array(learning_rate_history)

    if len(testing_error_history) != 0:
        testing_error_history = np.array(testing_error_history)
    if len(val_error_history) != 0:
        val_error_history = np.array(val_error_history)
    fig, axs = plt.subplots(3, 1, figsize=(3, 5), sharex='all', gridspec_kw={'height_ratios': [1, 2, 1]})
    ax2 = axs[0].twinx()
    axs[0].plot(roll_out_length_history[:, 0], roll_out_length_history[:, 1], 'blue')
    axs[0].set_ylabel('Roll out length', color='blue')
    axs[0].tick_params(axis='y', colors='blue')
    ax2.plot(use_optimizer_history[:, 0], use_optimizer_history[:, 1], 'red')
    ax2.set_ylabel('Optimizer', color='red')
    ax2.set_yticks([0, 1], ['LBFGS', 'Adam'])
    ax2.tick_params(axis='y', colors='red')

    axs[2].plot(learning_rate_history[:, 0], learning_rate_history[:, 1], 'green')
    axs[2].set_ylabel('Learning rate', color='green')
    axs[2].tick_params(axis='y', colors='green')

    ax3 = axs[1].twinx()
    axs[1].semilogy(loss_history[:, 0], loss_history[:, 1], color='b', label='Epoch Losses')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Epoch Losses', color='blue')
    axs[1].tick_params(axis='y', colors='blue')

    ax3.semilogy(error_history[:, 0], error_history[:, 1], color='r', label='Errors over training trajectories')
    ax3.set_ylabel('Error over training trajectories', color='red')
    ax3.tick_params(axis='y', colors='red')

    if len(testing_error_history) != 0:
        ax3.semilogy(testing_error_history[:, 0], testing_error_history[:, 1], color='y',
                     label='Errors over Testing trajectories')
    if len(val_error_history) != 0:
        ax3.semilogy(val_error_history[:, 0], val_error_history[:, 1], color='y',
                     label='Errors over Validation trajectories')
    max_loss = np.max(loss_history[:, 1]) if np.max(loss_history[:, 1]) < 1e4 else 1e4
    max_error = np.max(error_history[:, 1]) if np.max(error_history[:, 1]) < 1e4 else 1e4
    # min_loss = np.min(loss_history[:, 1])
    # min_error = np.min(error_history[:, 1])
    axs[1].set_ylim(None, max_loss + 1)
    ax3.set_ylim(None, max_error + 1)

    axs[1].legend(loc='center left', bbox_to_anchor=(1.35, 0.4))
    ax3.legend(loc='center left', bbox_to_anchor=(1.35, 0.6))
    if filepath is not None:
        filename = os.path.join(filepath, "error_vs_epoch.png")
        fig.savefig(f'{filename}', dpi=300, format='png')
    else:
        fig.show()


class RolloutEDMD(DMDBase):
    def __init__(self, observable, koopman_model, order, delta_t, normalizer=None):
        super().__init__(observable)

        self.epoch_after_reduce_plateau = None
        self.scheduler_plateau = None
        self.first_optimizer_is_Adam = None
        self.observable = observable
        self.order = order
        self.normalizer = normalizer
        self.koopman_model = koopman_model
        self.delta_t = delta_t
        self.name = f"Rollout eDMD"
        self.num_states = None
        self.plot_history_after_epoch = None
        self.decay_weight = None
        self.geometric_progression = None
        self.regularization_weight = None
        self.print_epoch_loss_after_epoch = None
        self.filepath = None
        self.compare_after_epoch = None
        self.roll_out_increase = None
        self.sequential_update_roll_out = None
        self.progressive_training = None
        self.switch_roll_out_after_epoch = None
        self.switch_optimizer_after_epoch = None
        self.batch_size = None
        self.batch_size_lbfgs = None
        self.num_epochs = None
        self.learning_rate = None
        self.max_roll_out_length = None
        self.A_init = None
        self.B_init = None
        self.iterator_for_rollout = 1
        self.switch_learning_rate_after_epoch = None
        self.num_inputs = None
        self.parameters = None
        self.learning_rate_decay = None
        self.learning_rate_lbfgs = None
        self.history = None
        self.scheduler = None
        self.optimizer_2 = None
        self.optimizer_1 = None
        self.inverse_loss_factor = None
        self.epoch_loss = None
        self.error_history = []
        self.currently_using_Adam = None
        self.early_stopping_patience = None
        # def transform(self, x, order=None):

    #     if order is None:
    #         order = self.order
    #     xi = []
    #     for N in range(order):
    #         time_delay_func = time_delay(model=self.gt_model, order=N)
    #         time_delay_value = torch.vmap(time_delay_func)(x) if x.ndim >= 2 else time_delay_func(x)
    #         xi.append(time_delay_value)
    #     return torch.hstack(xi) if x.ndim <= 2 else torch.cat(xi, dim=2)

    def setup(self, A_init=None,
              B_init=None,
              max_roll_out_length=None,
              learning_rate=0.01,
              num_epochs=100,
              batch_size=10,
              switch_optimizer_after_epoch=None,
              switch_roll_out_after_epoch=None,
              roll_out_increase=5,
              compare_after_epoch=10,
              filepath=None,
              print_epoch_loss_after_epoch=1,
              regularization_weight=None,
              switch_learning_rate_after_epoch=None,
              geometric_progression=False,
              decay_weight=1.0,
              plot_history_after_epoch=100,
              learning_rate_decay=1.0,
              learning_rate_lbfgs=0.01,
              inverse_loss_factor=1.0,
              first_optimizer_is_Adam=True,
              batch_size_lbfgs=None,
              reconstruction_loss_factor=0.0,
              show=True,
              epoch_after_reduce_plateau = np.inf,
              early_stopping_patience = 100,
              cycle_optimizer = False,
              early_stopping_threshold = 0.001):
        self.A_init = A_init
        self.B_init = B_init
        self.max_roll_out_length = max_roll_out_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.switch_optimizer_after_epoch = switch_optimizer_after_epoch
        self.switch_roll_out_after_epoch = switch_roll_out_after_epoch
        self.roll_out_increase = roll_out_increase
        self.compare_after_epoch = compare_after_epoch
        self.filepath = filepath
        self.print_epoch_loss_after_epoch = print_epoch_loss_after_epoch
        self.regularization_weight = regularization_weight
        self.switch_learning_rate_after_epoch = switch_learning_rate_after_epoch
        self.geometric_progression = geometric_progression
        self.decay_weight = decay_weight
        self.learning_rate_decay = learning_rate_decay
        self.plot_history_after_epoch = plot_history_after_epoch
        self.learning_rate_lbfgs = learning_rate_lbfgs
        self.inverse_loss_factor = inverse_loss_factor
        self.first_optimizer_is_Adam = first_optimizer_is_Adam
        self.batch_size_lbfgs = batch_size_lbfgs
        self.reconstruction_loss_factor = reconstruction_loss_factor
        self.show = show
        self.epoch_after_reduce_plateau = epoch_after_reduce_plateau
        self.early_stopping_patience = early_stopping_patience
        self.cycle_optimizer = cycle_optimizer
        self.early_stopping_threshold = early_stopping_threshold
        
        self.params = dict(A_init=A_init,
                           B_init=B_init,
                           max_roll_out_length=max_roll_out_length,
                           learning_rate=learning_rate,
                           num_epochs=num_epochs,
                           batch_size=batch_size,
                           switch_optimizer_after_epoch=switch_optimizer_after_epoch,
                           switch_roll_out_after_epoch=switch_roll_out_after_epoch,
                           roll_out_increase=roll_out_increase,
                           compare_after_epoch=compare_after_epoch,
                           filepath=filepath,
                           print_epoch_loss_after_epoch=print_epoch_loss_after_epoch,
                           regularization_weight=regularization_weight,
                           switch_learning_rate_after_epoch=switch_learning_rate_after_epoch,
                           geometric_progression=geometric_progression,
                           decay_weight=decay_weight,
                           learning_rate_decay=learning_rate_decay,
                           plot_history_after_epoch=plot_history_after_epoch,
                           learning_rate_lbfgs=learning_rate_lbfgs,
                           inverse_loss_factor=inverse_loss_factor,
                           first_optimizer_is_Adam=first_optimizer_is_Adam,
                           batch_size_lbfgs=batch_size_lbfgs,
                           reconstruction_loss_factor=reconstruction_loss_factor,
                           show=show,
                           epoch_after_reduce_plateau=epoch_after_reduce_plateau,
                           early_stopping_patience = early_stopping_patience,
                           cycle_optimizer=cycle_optimizer,
                           early_stopping_threshold = early_stopping_threshold)

    def loss_fun(self, xi, roll_out_length, u=None):
        # loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        inverse_loss = self.inverse_loss_factor * self.koopman_model.loss_for_inverse()
        xi0 = xi[:, 0, :]
        pred_states = self.koopman_model.rollout(x0=xi0, u=u, step=roll_out_length + 1)
        true_states = xi
        # reconstruction_loss = torch.square(torch.norm(pred_states[:, :, :self.num_states] - true_states[:, :, :self.num_states]))/self.num_states

        # loss = (torch.square(torch.norm(pred_states - true_states))/pred_states.shape[-1]
        #         + self.reconstruction_loss_factor * reconstruction_loss)
        
        reconstruction_loss = torch.sum(torch.square(pred_states[:, :, :self.num_states] - true_states[:, :, :self.num_states]))/self.num_states

        loss = (torch.sum(torch.square(pred_states - true_states))/pred_states.shape[-1]
                + self.reconstruction_loss_factor * reconstruction_loss)
        # for r in range(1, roll_out_length + 1):
        #     # pred_next_state = self.koopman_model.make_step(x0=xi0, u=u, step=r)
        #     pred_next_state = pred_states[r]
        #     # true_next_state = self.observable.transform(x[:, r, :])
        #     true_next_state = true_states[:,r,:]
        #     reconstruction_loss = torch.square(torch.norm(pred_next_state[:,:self.num_states] - true_next_state[:,:self.num_states], 'fro'))
        #     loss += (0.1 * reconstruction_loss +
        #              self.decay_weight ** r * torch.square(torch.norm(pred_next_state - true_next_state, 'fro')))

        return loss / roll_out_length + inverse_loss

    # def loss_fun(self, x, xi0, roll_out_length, u=None):
    #     # loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    #     inverse_loss = self.inverse_loss_factor *  self.koopman_model.loss_for_inverse()
    #     pred_states = self.koopman_model.rollout(x0=xi0,u=u, step = roll_out_length)
    #     true_states = self.observable.transform(x[:,1:,:].reshape(-1,x.shape[-1])).reshape(x.shape[0],x.shape[1]-1,-1)
    #     reconstruction_loss = torch.square(torch.norm(pred_states[:,:, :self.num_states] - true_states[:, :, :self.num_states], 'fro'))
    #
    #     loss = torch.square(torch.norm(pred_states - true_states, 'fro')) + 0.1 * reconstruction_loss
    #     # for r in range(1, roll_out_length + 1):
    #     #     # pred_next_state = self.koopman_model.make_step(x0=xi0, u=u, step=r)
    #     #     pred_next_state = pred_states[r]
    #     #     # true_next_state = self.observable.transform(x[:, r, :])
    #     #     true_next_state = true_states[:,r,:]
    #     #     reconstruction_loss = torch.square(torch.norm(pred_next_state[:,:self.num_states] - true_next_state[:,:self.num_states], 'fro'))
    #     #     loss += (0.1 * reconstruction_loss +
    #     #              self.decay_weight ** r * torch.square(torch.norm(pred_next_state - true_next_state, 'fro')))
    #
    #     return loss / roll_out_length + inverse_loss

    def choose_adam_or_lbfgs(self, epoch):
        # if is_plateau(error_history=self.error_history, patience=10,threshold=0.01):
        #     if self.currently_using_Adam =='adam':
        #         self.currently_using_Adam = 'lbfgs'
        #     else:
        #         self.currently_using_Adam = 'adam'
        # if self.sequential_update_roll_out is True:
        #     if self.switch_roll_out_after_epoch > self.switch_optimizer_after_epoch:
        #         if epoch % self.switch_roll_out_after_epoch < self.switch_optimizer_after_epoch:
        #             self.currently_using_Adam = True
        #         else:
        #             self.currently_using_Adam = False
        #     else:
        #         if epoch < self.switch_optimizer_after_epoch:
        #             self.currently_using_Adam = True
        #         else:
        #             self.currently_using_Adam = False
        if not self.cycle_optimizer:
            if epoch < self.switch_optimizer_after_epoch:
                self.currently_using_Adam = self.first_optimizer_is_Adam
            else:
                self.currently_using_Adam = not self.first_optimizer_is_Adam
        else:
            if self.first_optimizer_is_Adam:
                self.currently_using_Adam = ((epoch // self.switch_optimizer_after_epoch) % 2) == 0
            else:
                self.currently_using_Adam = ((epoch // self.switch_optimizer_after_epoch) % 2) != 0

        return self.currently_using_Adam

    def update_rollout_length(self, roll_out_length):
        if self.geometric_progression is True:
            roll_out_progress = roll_out_length * self.roll_out_increase
        else:
            roll_out_progress = self.roll_out_increase * self.iterator_for_rollout
            self.iterator_for_rollout += 1
        roll_out_length = roll_out_progress if roll_out_progress < self.max_roll_out_length else self.max_roll_out_length
        print(f"Current Roll out length: {roll_out_length}")
        return roll_out_length

    # def update_rollout_length(self, epoch, roll_out_length):
    #     # if is_plateau(error_history=self.error_history):
    #     #     if self.geometric_progression is True:
    #     #         roll_out_progress = roll_out_length * self.roll_out_increase
    #     #     else:
    #     #         roll_out_progress = self.roll_out_increase * self.iterator_for_rollout
    #     #         self.iterator_for_rollout += 1
    #     #     roll_out_length = roll_out_progress if roll_out_progress < self.max_roll_out_length else self.max_roll_out_length
    #     #     print(f"Current Roll out length: {roll_out_length}")
    #     if self.progressive_training:
    #         if self.sequential_update_roll_out:
    #             if (is_plateau(self.error_history,threshold=0.01,patience=10)
    #                     or (epoch >= self.switch_roll_out_after_epoch
    #                         and epoch % self.switch_roll_out_after_epoch == 0)):
    #                 if self.geometric_progression is True:
    #                     roll_out_progress = roll_out_length * self.roll_out_increase
    #                 else:
    #                     roll_out_progress = self.roll_out_increase * self.iterator_for_rollout
    #                     self.iterator_for_rollout += 1
    #                 roll_out_length = roll_out_progress if roll_out_progress < self.max_roll_out_length else self.max_roll_out_length
    #                 print(f"Current Roll out length: {roll_out_length}")
    #         else:
    #             if epoch > self.switch_roll_out_after_epoch:
    #                 roll_out_length = self.max_roll_out_length
    #     else:
    #         roll_out_length = self.max_roll_out_length
    #
    #     return roll_out_length
    def set_history(self, history):
        self.history = history

    def set_optimizers(self,
                       scheduler_mode="stepLR",
                       optimizer_1_dict=None,
                       optimizer_2_dict=None,
                       scheduler_dict=None,
                       config_cyclic_lr=None,
                       reduce_on_plateau=False):
        self.optimizer_1 = torch.optim.Adam(self.koopman_model.parameters(),
                                            lr=self.learning_rate)
        self.optimizer_2 = torch.optim.LBFGS(self.koopman_model.parameters(),
                                             lr=self.learning_rate_lbfgs,
                                             history_size=100,
                                             max_iter=20,
                                             line_search_fn="strong_wolfe"
                                             )
        if scheduler_mode == "stepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_1,
                                                             step_size=self.switch_learning_rate_after_epoch,
                                                             gamma=self.learning_rate_decay)
        elif scheduler_mode == "cyclicLR":
            if config_cyclic_lr is None:
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer_1,
                                                                   cycle_momentum=False,
                                                                   max_lr=0.1,
                                                                   base_lr=self.learning_rate,
                                                                   step_size_up=500,
                                                                   step_size_down=500,
                                                                   mode='triangular2')
            else:
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer_1,
                                                                   **config_cyclic_lr)
        if reduce_on_plateau:
            self.scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_1, mode='min', factor=0.1, patience=10)

        if optimizer_1_dict is not None:
            self.optimizer_1.load_state_dict(optimizer_1_dict)
        if optimizer_2_dict is not None:
            self.optimizer_2.load_state_dict(optimizer_2_dict)
        if scheduler_dict is not None:
            self.scheduler.load_state_dict(scheduler_dict)

    def fit(self, x, u=None, strategy=None,
            start_epoch=1, num_epochs=None,
            start_roll_out_length=1):
        print("current rollout length = ", start_roll_out_length)
        print('current device: ', x.device)
        num_epochs = self.num_epochs if num_epochs is None else num_epochs
        self.num_states = x.shape[-1]
        self.num_inputs = u.shape[-1] if u is not None else None
        roll_out_length = start_roll_out_length
        self.currently_using_Adam = self.first_optimizer_is_Adam
        if self.history is None:
            loss_history = []
            error_history = []
            testing_error_history = [] if self.test_data_available else None
            roll_out_length_history = []
            use_optimizer_history = []
            learning_rate_history = []
            val_error_history = []
        else:
            loss_history = self.history["loss_history"]
            error_history = self.history["error_history"]
            roll_out_length_history = self.history["roll_out_length_history"]
            use_optimizer_history = self.history["use_optimizer_history"]
            learning_rate_history = self.history["learning_rate_history"]
            if self.test_data_available:
                testing_error_history = self.history["testing_error_history"]
            else:
                testing_error_history = []
            if self.val_data_available:
                val_error_history = self.history["val_error_history"]
            else:
                val_error_history = []

        x_segmented, u_segmented = get_segmented_data(x, u, roll_out_length=roll_out_length)
        xi_segmented = self.observable.transform(x_segmented)

        def closure(xi, roll_out_length, u=None):
            self.optimizer_2.zero_grad()
            loss = self.loss_fun(xi=xi, roll_out_length=roll_out_length, u=u) / xi.shape[0]
            loss.backward()
            self.epoch_loss = loss.item()
            return loss.item()

        # def closure(x, xi0, roll_out_length, u=None):
        #     self.optimizer_2.zero_grad()
        #     # loss = self.loss_fun(xi=xi,roll_out_length=roll_out_length,u=u)
        #     loss = self.loss_fun(x=x, xi0=xi0, u=u, roll_out_length=roll_out_length) /x.shape[0]
        #     loss.backward()
        #     self.epoch_loss = loss.item()
        #     return loss.item()
        # wait = 0
        for epoch in range(start_epoch, num_epochs + start_epoch):
            self.epoch_loss = 0
            # wait -= 1
            # if is_plateau(self.error_history, threshold=0.01, patience=50):
            #     if wait <= 0 and roll_out_length != self.max_roll_out_length:
            #         print('plateau detected: increasing rollout length')
            #         roll_out_length = self.update_rollout_length_2(roll_out_length)
            #         wait = 50
            #         x_segmented, u_segmented = get_segmented_data(x, u, roll_out_length=roll_out_length)
            #         x_segmented_0 = x_segmented[:, 0, :]
            #         xi_segmented_0 = self.observable.transform(x_segmented_0)

            #
            # if is_oscillatory(self.error_history, patience=50, threshold=1):
            #     print('error oscillating : decreasing learning rate')
            #     current_optimizer = self.optimizer_1 if self.currently_using_Adam else self.optimizer_2
            #     for param_group in current_optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] * 0.8

            # if is_plateau(self.error_history, threshold=0.1, patience=10):
            #     probabilities = torch.tensor([1.0,0.0,0.0])
            #     result = torch.multinomial(probabilities, 1, replacement=True)
            #     if result.item()==0:
            #         roll_out_length = self.update_rollout_length_2(roll_out_length)
            #     elif result.item()==1:
            #         current_optimizer = self.optimizer_1 if self.currently_using_Adam else self.optimizer_2
            #         for param_group in current_optimizer.param_groups:
            #             param_group['lr'] = param_group['lr'] * 0.9
            #     elif result.item()==2:
            #         self.currently_using_Adam = not self.currently_using_Adam

            if self.switch_optimizer_after_epoch is not None:
                self.currently_using_Adam = self.choose_adam_or_lbfgs(epoch)

            if self.switch_roll_out_after_epoch is not None:
                if self.max_roll_out_length is not None:
                    if (epoch % self.switch_roll_out_after_epoch == 0 and roll_out_length != self.max_roll_out_length) or is_plateau(error_history, threshold=self.early_stopping_threshold, patience=self.early_stopping_patience//4 if np.isfinite(self.early_stopping_patience) else np.inf):
                        if is_plateau(error_history, threshold=self.early_stopping_threshold, patience=self.early_stopping_patience//4 if np.isfinite(self.early_stopping_patience) else np.inf):
                            print('plateau detected. switching rollout length')
                        roll_out_length = self.update_rollout_length(roll_out_length)
                        x_segmented, u_segmented = get_segmented_data(x, u, roll_out_length=roll_out_length)
                        xi_segmented = self.observable.transform(x_segmented)

            use_optimizer_history.append([epoch, self.currently_using_Adam])
            roll_out_length_history.append([epoch, roll_out_length])

            if self.currently_using_Adam:
                batches = get_all_batches_discrete_traj(xi_segmented, u_segmented, batch_size=self.batch_size)
                #
                # batches = get_all_batches_discrete_traj(x_segmented, u_segmented, batch_size=self.batch_size)
                for batch_xi, batch_x0, batch_u in batches:
                    self.optimizer_1.zero_grad()

                    # batch_xi0 = self.observable.transform(batch_x0)
                    # loss = self.loss_fun(x=batch_x,
                    #                      xi0=batch_xi0,
                    #                      u=batch_u,
                    #                      roll_out_length=roll_out_length) / batch_x.shape[0]

                    # batch_xi = self.observable.transform(batch_x.reshape(-1,batch_x.shape[-1])).reshape(batch_x.shape[0],batch_x.shape[1],-1)
                    # batch_xi = self.normalize_xi.transform(batch_xi)
                    # batch_u = self.normalize_u.transform(batch_u)
                    loss = self.loss_fun(xi=batch_xi,
                                         u=batch_u,
                                         roll_out_length=roll_out_length) / batch_xi.shape[0]
                    loss.backward()
                    # clip_grad_norm_(self.koopman_model.parameters(), max_norm=1000.0)
                    self.optimizer_1.step()
                    self.epoch_loss += loss.item()
                    
                if self.switch_learning_rate_after_epoch is not None and epoch < self.epoch_after_reduce_plateau:
                    self.scheduler.step()
                self.epoch_loss = self.epoch_loss / len(batches)
                current_learning_rate = self.optimizer_1.param_groups[-1]['lr']
            else:
                if epoch == self.switch_optimizer_after_epoch:
                    print("Using LBFGS")
                if self.batch_size_lbfgs is None:
                    # xi = self.observable.transform(x_segmented.reshape(-1,x_segmented.shape[-1])).reshape(x_segmented.shape[0],x_segmented.shape[1],-1)
                    # xi = self.normalize_xi.transform(xi)
                    # u_segmented = self.normalize_u.transform(u_segmented)
                    self.optimizer_2.step(lambda: closure(xi=xi_segmented,
                                                          u=u_segmented,
                                                          roll_out_length=roll_out_length))
                    # self.optimizer_2.step(lambda: closure(x=x_segmented,
                    #                                       xi0=xi_segmented_0,
                    #                                       u=u_segmented,
                    #                                       roll_out_length=roll_out_length))
                else:
                    # batches = get_all_batches_discrete_traj(x_segmented, u_segmented, batch_size=self.batch_size_lbfgs)
                    batches = get_all_batches_discrete_traj(xi_segmented, u_segmented, batch_size=self.batch_size_lbfgs)
                    for batch_xi, batch_x0, batch_u in batches:
                        # batch_xi = self.observable.transform(batch_x.reshape(-1, batch_x.shape[-1])).reshape(
                        #     batch_x.shape[0], batch_x.shape[1], -1)
                        # batch_xi = self.normalize_xi.transform(batch_xi)
                        # batch_u = self.normalize_u.transform(batch_u)
                        self.optimizer_2.step(lambda: closure(xi=batch_xi,
                                                              u=batch_u,
                                                              roll_out_length=roll_out_length))
                        # batch_xi0 = self.observable.transform(batch_x0)
                        # self.optimizer_2.step(lambda: closure(x=batch_x,
                        #                                  xi0=batch_xi0,
                        #                                  u=batch_u,
                        #                                  roll_out_length=roll_out_length))
                current_learning_rate = self.optimizer_2.param_groups[-1]['lr']

            learning_rate_history.append([epoch, current_learning_rate])
            loss_history.append([epoch, self.epoch_loss])

            if epoch % self.print_epoch_loss_after_epoch == 0:
                print(f"Epoch {epoch}/{start_epoch + num_epochs - 1}: Epoch Loss ={self.epoch_loss}")

            if epoch % self.compare_after_epoch == 0:
                if epoch % self.plot_history_after_epoch == 0:
                    with torch.no_grad():
                        error = self.compare(model=self.koopman_model, epoch=epoch, filepath=self.filepath,
                                             show=self.show)
                        if self.test_data_available:
                            testing_error = self.compare(model=self.koopman_model, epoch=epoch, filepath=self.filepath,
                                                         show=self.show,
                                                         x0_true=self.x0_test, x_true=self.x_test, steps=self.steps,
                                                         u_true=self.u_test,
                                                         x_noisy=self.x_noisy_test, tag='test', plot=self.show)
                        if self.val_data_available:
                            val_error = self.compare(model=self.koopman_model, epoch=epoch, filepath=self.filepath,
                                                         show=self.show,
                                                         x0_true=self.x0_val, x_true=self.x_val, steps=self.steps,
                                                         u_true=self.u_val,
                                                         x_noisy=self.x_noisy_val, tag='validation', plot=self.show)


                else:
                    with torch.no_grad():
                        error = self.compare(model=self.koopman_model, epoch=epoch, filepath=self.filepath, plot=False)
                        if self.test_data_available:
                            testing_error = self.compare(model=self.koopman_model, epoch=epoch, filepath=self.filepath,
                                                         show=self.show,
                                                         x0_true=self.x0_test, x_true=self.x_test, steps=self.steps,
                                                         u_true=self.u_test,
                                                         x_noisy=self.x_noisy_test, plot=False,
                                                         tag='test')
                        if self.val_data_available:
                            val_error = self.compare(model=self.koopman_model, epoch=epoch, filepath=self.filepath,
                                                         show=self.show,
                                                         x0_true=self.x0_val, x_true=self.x_val, steps=self.steps,
                                                         u_true=self.u_val,
                                                         x_noisy=self.x_noisy_val, plot=False, tag='validation')
                if epoch > self.epoch_after_reduce_plateau and self.scheduler_plateau is not None and self.val_data_available:
                    self.scheduler_plateau.step(val_error)

                error_history.append([epoch, error])
                if self.test_data_available:
                    testing_error_history.append([epoch, testing_error])
                if self.val_data_available:
                    val_error_history.append([epoch, val_error])
                self.error_history = np.append(self.error_history, error)

                print(f"Epoch {epoch}/{start_epoch + num_epochs - 1}: Trajectory error = {error}")
                if self.test_data_available:
                    print(f"Testing errors = {testing_error}")
                if self.val_data_available:
                    print(f"Validation errors = {val_error}")

            if epoch % self.plot_history_after_epoch == 0:
                history = dict(loss_history=loss_history,
                               error_history=error_history,
                               roll_out_length_history=roll_out_length_history,
                               use_optimizer_history=use_optimizer_history,
                               learning_rate_history=learning_rate_history,
                               testing_error_history=testing_error_history,
                               val_error_history=val_error_history)
                save_file(history, 'training_history.pkl', path=self.filepath)
                plot_history(**history, filepath=self.filepath)
                checkpoint = dict(epoch=epoch,
                                  koopman_model=self.koopman_model,
                                  koopman_model_state_dict=self.koopman_model.state_dict(),
                                  optimizer_1_state_dict=self.optimizer_1.state_dict(),
                                  optimizer_2_state_dict=self.optimizer_2.state_dict(),
                                  roll_out_length=roll_out_length,
                                  scheduler_state_dict=self.scheduler.state_dict(),
                                  use_Adam=self.currently_using_Adam)
                checkpoint_path = os.path.join(self.filepath, 'checkpoint.pth')
                torch.save(checkpoint, checkpoint_path)
            if is_plateau(error_history, threshold=self.early_stopping_threshold, patience=self.early_stopping_patience):
                print('plateu detected. early stopping.')
                break

        history = dict(loss_history=loss_history,
                       error_history=error_history,
                       roll_out_length_history=roll_out_length_history,
                       use_optimizer_history=use_optimizer_history,
                       learning_rate_history=learning_rate_history,
                       testing_error_history=testing_error_history,
                       val_error_history=val_error_history
                       )
        save_file(history, 'training_history.pkl', path=self.filepath)
        plot_history(**history, filepath=self.filepath)

        checkpoint = dict(epoch=epoch,
                          koopman_model=self.koopman_model,
                          koopman_model_state_dict=self.koopman_model.state_dict(),
                          optimizer_1_state_dict=self.optimizer_1.state_dict(),
                          optimizer_2_state_dict=self.optimizer_2.state_dict(),
                          scheduler_state_dict=self.scheduler.state_dict(),
                          roll_out_length=roll_out_length)
        checkpoint_path = os.path.join(self.filepath, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        A = self.koopman_model.A.detach().cpu()
        B = self.koopman_model.B.detach().cpu() if self.koopman_model.B is not None else None
        save_file(dict(A=A, B=B), 'model_params.pkl', path=self.filepath)
        error_dict = dict(error=error)
        if self.test_data_available:
            error_dict["testing_error"] = testing_error
        if self.val_data_available:
            error_dict["val_error"] = val_error
        return A, B, error_dict

    # def compare_with(self, x_true,x_noisy, x0_true, steps, u_true=None):
    #     self.x_true = x_true
    #     self.x0_true = x0_true
    #     self.steps = steps
    #     self.u_true = u_true
    #     self.xi0_true = self.observable.transform(self.x0_true)
    #     self.x_noisy = x_noisy
    #     return self

    # def compare(self, A, epoch, filepath, B=None):
    #     model_with_rollout = LinearModelDiscrete(A=A, B=B)
    #     solver_with_rollout = Simulator(model=model_with_rollout, steps=self.steps, x0=self.xi0_true)
    #     xi_with_rollout = solver_with_rollout.rollout(u=self.u_true)
    #     fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    #     error_with_roll_out = torch.norm(xi_with_rollout[:, :, :2] - self.x_true)
    #     plot_pplane(
    #         ax,
    #         x=self.x_true,
    #         color='blue',
    #         linestyle='solid',
    #         label='Ground Truth',
    #     )
    #     plot_pplane(
    #         ax,
    #         x=self.x_noisy,
    #         color='blue',
    #         linestyle='None',
    #         marker='o',
    #         markersize=1.5,
    #         label='Data',
    #     )
    #     plot_pplane(ax, x=xi_with_rollout, color='red', linestyle='dashed',
    #                                     label=f'EDMD with polyflow basis, error : {error_with_roll_out}',
    #                                     marker='o', markersize=1.2)
    #     ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    #     fig.show()
    #     fig.savefig(f'{filepath}/epoch{epoch}.png', format='png', dpi=300)
    #     save_file(error_with_roll_out.cpu().numpy(),"trajectory_error.pkl",filepath)
    #     return error_with_roll_out.cpu().numpy()


def is_oscillatory(error_history, threshold=0.01, patience=10):
    if len(error_history) < patience:
        return False
    recent_errors = error_history[-patience:]
    diffs = [abs(recent_errors[i] - recent_errors[i - 1]) for i in range(1, len(recent_errors))]
    oscillating = sum(diffs) / len(diffs) < threshold
    return oscillating


def is_plateau(error_history, threshold=0.01, patience=10):
    """
    Checks if the error rate has plateaued over the last few epochs.

    Parameters:
    - error_history: List of recorded error rates (e.g., validation loss).
    - threshold: Minimum significant change in error.
    - patience: Number of epochs to consider for detecting a plateau.

    Returns:
    - True if a plateau is detected, False otherwise.
    """
    # Ensure we have enough data to check for a plateau
    if len(error_history) < patience:
        return False

    # Look at the change in error over the patience period
    recent_errors = np.array(error_history[-patience:])[:,1]
    min_recent_error = min(recent_errors)
    max_recent_error = max(recent_errors)

    # Check if the change in error is less than the threshold
    return (max_recent_error - min_recent_error) < threshold
