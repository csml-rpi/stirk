import matplotlib.pyplot as plt
import torch.nn

from solve_dynamics import *
from abc import abstractmethod
import do_mpc
import matplotlib.patches as patches
import matplotlib.animation as anim
# Todo: Base class for nonlinear models
import time


class Normalizer:
    def __init__(self, tensor = None):
        if tensor is not None:
            num_dims = len(tensor.shape)
            dims_to_reduce = tuple(range(num_dims - 1))
            self.mean = tensor.mean(dim=dims_to_reduce, keepdim=True)
            self.std = tensor.std(dim=dims_to_reduce, keepdim=True)
            self.fitted = True
        else:
            self.fitted = False
    def fit(self, tensor):
        num_dims = len(tensor.shape)
        dims_to_reduce = tuple(range(num_dims - 1))
        self.mean = tensor.mean(dim=dims_to_reduce, keepdim=True)
        self.std = tensor.std(dim=dims_to_reduce, keepdim=True)
        self.fitted = True
        return self
    def transform(self, tensor):
        if self.fitted:
            result =(tensor - self.mean) / (self.std + 1e-8)
            if tensor.ndim == 2:
                return result.squeeze(0) # TODO: need to make it general
            else:
                return result
        else:
            print('Not fitted')
    def inverse_transform(self, tensor):
        if self.fitted:
            return tensor * (self.std + 1e-8) + self.mean
        else:
            print('Not fitted')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        pass

class LinearModelDiscrete(nn.Module):
    def __init__(self, A, B = None):
        super().__init__()
        self.A = A
        self.B = B
        self.num_states = self.A.shape[1]
        if self.B is not None:
            self.num_inputs = self.B.shape[1]
        self.name = 'Linear Model Discrete'
    def forward(self, x, u = None):
        if self.B is None and u is None:
            output = x @ self.A.t() if x.ndim == 2 else self.A @ x.t()
        else:
            output = x @ self.A.t() + u @ self.B.t() if x.ndim == 2 else self.A @ x.t() + self.B @ u.t()
        return output
    def get_do_mpc_model(self):
        do_mpc_model = do_mpc.model.Model(model_type = "discrete")
        self.x = do_mpc_model.set_variable(var_type='_x', var_name='x', shape=(self.num_states, 1))
        self.u = do_mpc_model.set_variable(var_type='_u', var_name='u', shape=(self.num_inputs, 1))

        A_numpy = self.A.cpu().detach().numpy()
        B_numpy = self.B.cpu().detach().numpy()

        x_next = A_numpy @ self.x + B_numpy @ self.u
        do_mpc_model.set_rhs('x', x_next)

        do_mpc_model.setup()

        return do_mpc_model


# class VanDerPolDiscrete(nn.Module):
#     def __init__(self, mu, delta_t):
#         super().__init__()
#         self.name= "Van Der Pol Discrete"
#         self.mu = mu
#         self.delta_t = delta_t
#
#     def odefun(self, t, x):
#         x1 = x[:, 0] if x.ndim == 2 else x[0]
#         x2 = x[:, 1] if x.ndim == 2 else x[1]
#
#         x1dot = -x2
#         x2dot = self.mu * (-1 + x1**2) * x2 + x1
#
#         output = torch.stack([x1dot, x2dot], dim=1) if x.ndim == 2 else torch.stack([x1dot, x2dot])
#         return output
#
#     def forward(self, x):
#         x_next = rk4_step(func = self.odefun, t=0, x=x, delta_t=self.delta_t)
#         return x_next

class DiscretizedODEModel(nn.Module):
    def __init__(self):
        super(DiscretizedODEModel, self).__init__()
        self.num_states = None
        self.num_inputs = None
        self.delta_t = None
    def odefun(self, t, x, u):
        x_tuple = ()
        u_tuple = () if u is not None else None
        if x.ndim ==2:
            for i in range(x.shape[-1]):
                x_tuple += (x[:, i],)
        else:
            x_tuple = x

        if u is not None:
            if u.ndim ==2:
                for i in range(u.shape[-1]):
                    u_tuple += (u[:,i],)

        xdot_tuple = self.get_x_dot(x=x_tuple,u=u_tuple)

        output = torch.stack([*xdot_tuple], dim=1) if x.ndim == 2 else torch.stack([*xdot_tuple])
        return output
    def get_do_mpc_model(self):
        model_type = "discrete"
        do_mpc_model = do_mpc.model.Model(model_type)

        self.x = do_mpc_model.set_variable(var_type='_x', var_name='x', shape=(self.num_states, 1))
        self.u = do_mpc_model.set_variable(var_type='_u', var_name='u', shape=(self.num_inputs, 1))

        # Define continuous dynamics
        def odefun(t, x, u):
            xdot = self.get_x_dot(x=x, u=u)
            return np.array([*xdot])

        # Implement RK4 for discretization
        x_next = rk4_step_control(func = odefun, t=0, x=self.x, u=self.u, delta_t=self.delta_t)
        do_mpc_model.set_rhs('x', x_next)

        do_mpc_model.setup()

        return do_mpc_model


class VanDerPolDiscreteControl(DiscretizedODEModel):
    def __init__(self, mu=1, delta_t=0.1):
        super().__init__()
        self.name = "Van Der Pol Discrete Control"
        self.delta_t = delta_t
        self.mu = mu
        self.num_states = 2
        self.num_inputs = 1
    def get_x_dot(self, x, u):
        x1 = x[0]
        x2 = x[1]
        u1 = u[0] if u is not None else 0
        x1dot = -x2
        x2dot = self.mu * (-1 + x1 ** 2) * x2 + x1 + u1
        return x1dot, x2dot
    def forward(self, x, u=None):
        x_next = rk4_step_control(func=self.odefun, t=0, x=x, u=u, delta_t=self.delta_t)
        return x_next
def animate_cartpole2(x, ax,
            fig,
            filename="movie",
            cart_width=1,
            cart_height=1,
            cart_color='yellow',
            init_cond_index=0,
            save=True,
            path=None,
            fps=30,
            writer='ffmpeg',
            ext='mp4'):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()  # [trajectories, timesteps, states]
    steps = x.shape[1]
    cart_pos = (-cart_width / 2, -cart_height)
    cart = patches.Rectangle(cart_pos,
                             width=cart_width,
                             height=cart_height,
                             edgecolor='k',
                             facecolor=cart_color)
    ax.add_patch(cart)
    ax.set(xlim=(-5, 5), ylim=(-4, 4))
    ax.set_aspect('equal', adjustable='box')
    pendulum, = ax.plot([], [], 'r-', lw=2, markevery=2, marker='o')
    ground = ax.plot([-5, 5], [-cart_height, -cart_height], 'k-', lw=3)
    # ax.text(0.5,1.0, f'{np.round(xs[0,init_cond_index,:],decimals=4)}', transform = ax.transAxes)
    ax.set_title(f'{np.round(x[init_cond_index,0,:], decimals=4)}')
    ax.axis('equal')
    def update(frame, i):
        # Get the position of the cart and the pendulum bob at the current frame
        x_pos = x[i, frame, 0]
        theta = x[i, frame, 1]

        # Calculate the coordinates of the pendulum bob
        bob_x = x_pos + 3.0 * np.sin(theta)
        bob_y = 3.0 * np.cos(theta)

        # Update the plot elements for the current frame
        cart.set_x(x_pos - cart_width / 2)
        pendulum.set_data([bob_x, x_pos], [bob_y, 0.0])

        return cart, pendulum

    animation = anim.FuncAnimation(fig, update,
                                   fargs=(init_cond_index,),
                                   repeat=False,
                                   interval=10,
                                   frames=steps-1,
                                   blit=True)
    if save is True:
        filepath = os.path.join(path,f"{filename}.{ext}")
        animation.save(filepath, writer=writer, fps=fps)

    return update, animation

def plot_cartpole_states(x, ax,
                         cart_width=1,
                         cart_height=1,
                         cart_color='yellow',
                         alpha_step=0.05,
                         jump=50):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()  # [trajectories, timesteps, states]
    steps = x.shape[0]
    alpha = 0.0
    for i in range(0, steps, jump):  # Increment by 10
        alpha += 100* i/steps   # Adjust alpha based on number of steps shown
        x_pos = x[i, 0]
        theta = x[i, 1]

        # Calculate the coordinates of the pendulum bob
        bob_x = x_pos + 2.0 * np.sin(theta)
        bob_y = 2.0 * np.cos(theta)

        # Plot the cart
        cart_pos = (x_pos - cart_width / 2, -cart_height / 2)
        cart = patches.Rectangle(cart_pos, width=cart_width, height=cart_height,
                                 edgecolor='k', facecolor=cart_color, alpha=alpha)
        ax.add_patch(cart)

        # Plot the pendulum
        ax.plot([x_pos, bob_x], [0, bob_y], 'k-', alpha=alpha, lw=2)

    ax.set(xlim=(-5, 5), ylim=(-4, 4))
    ax.set_aspect('equal', adjustable='box')
    ax.plot([-5, 5], [-cart_height / 2, -cart_height / 2], 'k-', lw=3)  # ground


# def plot_cartpole_states(x, ax,
#                          cart_width=1,
#                          cart_height=1,
#                          cart_color='yellow',
#                          alpha_step=0.05):
#     if isinstance(x, torch.Tensor):
#         x = x.cpu().numpy()  # [trajectories, timesteps, states]
#     steps = x.shape[0]
#     for i in range(steps):
#         alpha = 1 - i * alpha_step  # Decrease alpha to make earlier states more transparent
#         x_pos = x[i, 0]
#         theta = x[i, 1]

#         # Calculate the coordinates of the pendulum bob
#         bob_x = x_pos + 2.0 * np.sin(theta)
#         bob_y = 2.0 * np.cos(theta)

#         # Plot the cart
#         cart_pos = (x_pos - cart_width / 2, -cart_height / 2)
#         cart = patches.Rectangle(cart_pos, width=cart_width, height=cart_height,
#                                  edgecolor='k', facecolor=cart_color, alpha=alpha)
#         ax.add_patch(cart)

#         # Plot the pendulum
#         ax.plot([x_pos, bob_x], [0, bob_y], 'r-', alpha=alpha, lw=2)

#     ax.set(xlim=(-5, 5), ylim=(-4, 4))
#     ax.set_aspect('equal', adjustable='box')
#     ax.plot([-5, 5], [-cart_height / 2, -cart_height / 2], 'k-', lw=3)  # ground


def animate_cartpole(x, ax,
            fig,
            filename="movie",
            cart_width=1,
            cart_height=1,
            cart_color='yellow',
            init_cond_index=0,
            save=True,
            path=None,
            fps=30,
            writer='ffmpeg',
            ext='mp4'):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()  # [trajectories, timesteps, states]
    steps = x.shape[1]
    cart_pos = (-cart_width / 2, -cart_height)
    cart = patches.Rectangle(cart_pos,
                             width=cart_width,
                             height=cart_height,
                             edgecolor='k',
                             facecolor=cart_color)
    ax.add_patch(cart)
    ax.set(xlim=(-5, 5), ylim=(-4, 4))
    ax.set_aspect('equal', adjustable='box')
    pendulum, = ax.plot([], [], 'r-', lw=2, markevery=2, marker='o')
    ground = ax.plot([-5, 5], [-cart_height, -cart_height], 'k-', lw=3)
    # ax.text(0.5,1.0, f'{np.round(xs[0,init_cond_index,:],decimals=4)}', transform = ax.transAxes)
    ax.set_title(f'{np.round(x[init_cond_index,0,:], decimals=4)}')
    ax.axis('equal')
    def update(frame, i):
        # Get the position of the cart and the pendulum bob at the current frame
        x_pos = x[i, frame, 0]
        theta = x[i, frame, 2]

        # Calculate the coordinates of the pendulum bob
        bob_x = x_pos + 3.0 * np.sin(theta)
        bob_y = 3.0 * np.cos(theta)

        # Update the plot elements for the current frame
        cart.set_x(x_pos - cart_width / 2)
        pendulum.set_data([bob_x, x_pos], [bob_y, 0.0])

        return cart, pendulum

    animation = anim.FuncAnimation(fig, update,
                                   fargs=(init_cond_index,),
                                   repeat=False,
                                   interval=10,
                                   frames=steps-1,
                                   blit=True)
    if save is True:
        animation.save(f'{path}{filename}.{ext}', writer=writer, fps=fps)

    return update, animation

class CartPole2(DiscretizedODEModel):
    def __init__(self, delta_t = 0.2):
        super().__init__()
        self.name = "CartPole2"
        self.num_states = 4
        self.num_inputs = 1
        self.delta_t = delta_t
        self.g = 9.8 # m/s^2
        self.m_c = 4 # kg
        self.m_p = 1 # kg
        self.l = 1 # m

    def get_x_dot(self, x, u):
        if isinstance(x[0], torch.Tensor):
            lib = torch
        else:
            lib = np # for do-mpc, x is casadi.casadi.SX

        x1 = x[0] # position
        x2 = x[1] # velocity
        x3 = x[2] # angle
        x4 = x[3] # angular velocity
        u1 = u[0] if u is not None else 0 # Force on cart
        x1dot = x3
        x2dot = x4
        x3dot = (u1 + self.m_p * lib.sin(x2)*(self.l * x4**2 - self.g*lib.cos(x2)))/(self.m_c+self.m_p * lib.sin(x2)**2)
        x4dot = (u1*lib.cos(x2) + self.m_p * self.l* x4**2 * lib.cos(x2) * lib.sin(x2) -(self.m_c + self.m_p) * self.g * lib.sin(x2))/(self.l * (self.m_c + self.m_p * lib.sin(x2)**2))
        return x1dot, x2dot, x3dot, x4dot

    def forward(self, x, u=None):
        x_next = rk4_step_control(func=self.odefun, t=0, x=x, u=u, delta_t=self.delta_t)
        return x_next



class CartPole(DiscretizedODEModel):
    def __init__(self, delta_t = 0.02):
        super().__init__()
        self.name = "CartPole"
        self.num_states = 4
        self.num_inputs = 1
        self.delta_t = delta_t
        self.g = 9.8 # m/s^2
        self.m_c = 1 # kg
        self.m = 0.1 # kg
        self.l = 0.5 # m
        self.mu_c = 0.0005 #
        self.mu_p = 0.000002 #

    def get_x_dot(self, x, u):
        if isinstance(x[0], torch.Tensor):
            lib = torch
        else:
            lib = np # for do-mpc, x is casadi.casadi.SX

        x1 = x[0] # position
        x2 = x[1] # velocity
        x3 = x[2] # angle
        x4 = x[3] # angular velocity
        u1 = u[0] if u is not None else 0 # Force on cart
        x1dot = x2
        x3dot = x4

        term_1 = self.g * lib.sin(x3)
        term_2a = lib.cos(x3)
        term_2b = self.m * self.l * x4**2 *lib.sin(x3)
        term_2c = self.mu_c * lib.sign(x2)
        term_2d = self.m_c + self.m
        term_2 = term_2a * (-u1 - term_2b + term_2c)/term_2d
        term_3 = (self.mu_p * x3)/(self.m * self.l)
        term_4 = self.l * (4/3 - self.m * lib.cos(x3) **2/term_2d)

        x4dot = (term_1 + term_2 - term_3) / term_4

        term_5 = u1 + self.m * self.l * (x4**2 *lib.sin(x3) - x4dot * lib.cos(x3))
        term_6 = term_2c
        term_7 = term_2d

        x2dot = (term_5 + term_6) / term_7

        return x1dot, x2dot, x3dot, x4dot
    def forward(self, x, u=None):
        x_next = rk4_step_control(func=self.odefun, t=0, x=x, u=u, delta_t=self.delta_t)
        return x_next

class CSTRModel(DiscretizedODEModel):
    def __init__(self, delta_t=0.02):
        super().__init__()
        self.name = "CSTR"
        self.num_states = 4
        self.num_inputs = 2
        self.delta_t = delta_t
        self.C_a_mean = 1
        self.C_b_mean = 1
        self.T_R_mean = 95
        self.T_K_mean = 95
        self.C_a_std = 0.5
        self.C_b_std = 0.5
        self.T_R_std = 25
        self.T_K_std = 25
        self.F_mean = 15
        self.F_std = 1
        self.Q_dot_mean = -8000
        self.Q_dot_std = 1000

    def normalize(self, x, u =None):
        if x.ndim==2:
            C_a = (x[:,0] - self.C_a_mean) / self.C_a_std #  concentration of reactant A
            C_b = (x[:,1] -self.C_b_mean) / self.C_b_std # concentration of reactant B
            T_R = (x[:,2] - self.T_R_mean) / self.T_R_std # temperature inside the reactor
            T_K = (x[:,3] - self.T_K_mean) / self.T_K_std # temperature of the cooling jacket
        if x.ndim==3:
            C_a = (x[:,:, 0] - self.C_a_mean) / self.C_a_std  # concentration of reactant A
            C_b = (x[:,:, 1] - self.C_b_mean) / self.C_b_std  # concentration of reactant B
            T_R = (x[:,:, 2] - self.T_R_mean) / self.T_R_std  # temperature inside the reactor
            T_K = (x[:,:, 3] - self.T_K_mean) / self.T_K_std  # temperature of the cooling jacket

        F = (u[:,:,0]- self.F_mean) / self.F_std if u is not None else 0 # feed
        Q_dot = (u[:,:,1]- self.Q_dot_mean) / self.Q_dot_std if u is not None else 0 # heat flow
        x_tilde = torch.stack([C_a, C_b, T_R, T_K], dim=-1).to(x.device, x.dtype)
        u_tilde = torch.stack([F, Q_dot], dim=-1).to(u.device, u.dtype) if u is not None else None
        if u_tilde is not None:
            return x_tilde, u_tilde
        else:
            return x_tilde

    def denormalize(self, x_tilde, u_tilde=None):
        C_a = x_tilde[:,0] * self.C_a_std + self.C_a_mean  # concentration of reactant A
        C_b = x_tilde[:,1] * self.C_b_std + self.C_b_mean  # concentration of reactant B
        T_R = x_tilde[:,2] * self.T_R_std + self.T_R_mean  # temperature inside the reactor
        T_K = x_tilde[:,3] * self.T_K_std + self.T_K_mean  # temperature of the cooling jacket

        x = torch.stack([C_a, C_b, T_R, T_K], dim=-1).to(x_tilde.device, x_tilde.dtype)

        if u_tilde is not None:
            F = u_tilde[:,0] * self.F_std + self.F_mean  # feed
            Q_dot = u_tilde[:,1] * self.Q_dot_std + self.Q_dot_mean  # heat flow
            u = torch.stack([F, Q_dot], dim=-1).to(u_tilde.device, u_tilde.dtype)
        else:
            u = None
        if u is None:
            return x
        return x, u

    def get_x_dot(self, x, u):
        if isinstance(x[0], torch.Tensor):
            lib = torch
        else:
            lib = np # for do-mpc, x is casadi.casadi.SX

        C_a = x[0]  * self.C_a_std + self.C_a_mean  #  concentration of reactant A
        C_b = x[1]  * self.C_b_std + self.C_b_mean  # concentration of reactant B
        T_R = x[2]  * self.T_R_std + self.T_R_mean  # temperature inside the reactor
        T_K = x[3]  * self.T_K_std + self.T_K_mean # temperature of the cooling jacket
        F = u[0] * self.F_std + self.F_mean if u is not None else 0 # feed
        Q_dot = u[1] * self.Q_dot_std + self.Q_dot_mean if u is not None else 0 # heat flow
  
        # C_a = x[0]    #  concentration of reactant A
        # C_b = x[1]   # concentration of reactant B
        # T_R = x[2]   # temperature inside the reactor
        # T_K = x[3]  # temperature of the cooling jacket
        # F = u[0]  if u is not None else 0 # feed
        # Q_dot = u[1]  if u is not None else 0 # heat flow
        # Certain parameters

        K0_ab = 1.287e12  # K0 [h^-1]
        K0_bc = 1.287e12  # K0 [h^-1]
        K0_ad = 9.043e9  # K0 [l/mol.h]
        R_gas = 8.3144621e-3  # Universal gas constant
        E_A_ab = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_bc = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_ad = 8560.0 * 1.0  # * R_gas# [kj/mol]
        H_R_ab = 4.2  # [kj/mol A]
        H_R_bc = -11.0  # [kj/mol B] Exothermic
        H_R_ad = -41.85  # [kj/mol A] Exothermic
        rho = 0.9342  # Density [kg/l]
        Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
        Cp_k = 2.0  # Coolant heat capacity [kj/kg.k]
        A_R = 0.215  # Area of reactor wall [m^2]
        V_R = 10.01  # 0.01 # Volume of reactor [l]
        m_k = 5.0  # Coolant mass[kg]
        T_in = 130.0  # Temp of inflow [Celsius]Pl
        K_w = 4032.0  # [kj/h.m^2.K]
        C_A0 = (5.7 + 4.5) / 2.0 * 1.0  # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]
        alpha = 1 #uncertain parameters: taken to be certain
        beta = 1 #uncertain parameters: taken to be certain
        # Auxiliary terms
        K_1 = beta * K0_ab * lib.exp((-E_A_ab) / (T_R + 273.15))
        K_2 = K0_bc * lib.exp((-E_A_bc) / (T_R + 273.15))
        K_3 = K0_ad * lib.exp((-alpha * E_A_ad) / (T_R + 273.15))
        T_dif = T_R-T_K


        C_a_dot = F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2)
        C_b_dot = -F*C_b + K_1*C_a - K_2*C_b
        T_R_dot = (K_1 *C_a* H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-rho * Cp) \
                  + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif)) / (rho * Cp * V_R))
        T_K_dot = (Q_dot + K_w * A_R * T_dif) / (m_k * Cp_k)

        C_a_dot = C_a_dot / self.C_a_std #  concentration of reactant A
        C_b_dot = C_b_dot / self.C_b_std # concentration of reactant B
        T_R_dot = T_R_dot / self.T_R_std # temperature inside the reactor
        T_K_dot = T_K_dot / self.T_K_std # temperature of the cooling jacket

        return C_a_dot, C_b_dot, T_R_dot, T_K_dot
    def forward(self, x, u=None):
        x_next = rk4_step_control(func=self.odefun, t=0, x=x, u=u, delta_t=self.delta_t)
        return x_next


class CSTRModelNondimensional(DiscretizedODEModel):
    def __init__(self, delta_t=0.02):
        super().__init__()
        self.name = "CSTR"
        self.num_states = 4
        self.num_inputs = 2
        self.delta_t = delta_t
        self.T_ref = 100
    def get_x_dot(self, x, u):
        if isinstance(x[0], torch.Tensor):
            lib = torch
        else:
            lib = np  # for do-mpc, x is casadi.casadi.SX

        C_a = x[0]  # concentration of reactant A
        C_b = x[1]  # concentration of reactant B
        T_R = x[2]  # temperature inside the reactor
        T_K = x[3]  # temperature of the cooling jacket
        F = u[0] if u is not None else 0  # feed
        Q_dot = u[1] if u is not None else 0  # heat flow
        # Certain parameters

        K0_ab = 1.287e12  # K0 [h^-1]
        K0_bc = 1.287e12  # K0 [h^-1]
        K0_ad = 9.043e9  # K0 [l/mol.h]
        R_gas = 8.3144621e-3  # Universal gas constant
        E_A_ab = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_bc = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_ad = 8560.0 * 1.0  # * R_gas# [kj/mol]
        H_R_ab = 4.2  # [kj/mol A]
        H_R_bc = -11.0  # [kj/mol B] Exothermic
        H_R_ad = -41.85  # [kj/mol A] Exothermic
        rho = 0.9342  # Density [kg/l]
        Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
        Cp_k = 2.0  # Coolant heat capacity [kj/kg.k]
        A_R = 0.215  # Area of reactor wall [m^2]
        V_R = 10.01  # 0.01 # Volume of reactor [l]
        m_k = 5.0  # Coolant mass[kg]
        T_in = 130.0  # Temp of inflow [Celsius]Pl
        K_w = 4032.0  # [kj/h.m^2.K]
        C_A0 = (5.7 + 4.5) / 2.0 * 1.0  # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]
        alpha = 1  # uncertain parameters: taken to be certain
        beta = 1  # uncertain parameters: taken to be certain
        # Auxiliary terms
        K_1 = beta * K0_ab * lib.exp((-E_A_ab) / (T_R + 273.15))
        K_2 = K0_bc * lib.exp((-E_A_bc) / (T_R + 273.15))
        K_3 = K0_ad * lib.exp((-alpha * E_A_ad) / (T_R + 273.15))
        T_dif = T_R - T_K
        C_a_dot = F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a ** 2)
        C_b_dot = -F * C_b + K_1 * C_a - K_2 * C_b
        T_R_dot = (K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * (C_a ** 2) * H_R_ad) / (-rho * Cp) \
                  + F * (T_in - T_R) + (((K_w * A_R) * (-T_dif)) / (rho * Cp * V_R))
        T_K_dot = (Q_dot + K_w * A_R * T_dif) / (m_k * Cp_k)
        return C_a_dot, C_b_dot, T_R_dot, T_K_dot

    def forward(self, x, u=None):
        x_next = rk4_step_control(func=self.odefun, t=0, x=x, u=u, delta_t=self.delta_t)
        return x_next

# class VanDerPolDiscreteControl(nn.Module):
#     def __init__(self, mu=1, delta_t=0.1):
#         super().__init__()
#         self.name= "Van Der Pol Discrete Control"
#         self.delta_t = delta_t
#         self.mu = mu
#         self.num_states = 2
#         self.num_inputs = 1
#     def get_x_dot(self,x,u):
#         x1 = x[0]
#         x2 = x[1]
#         x1dot = -x2
#         x2dot = self.mu * (-1 + x1**2) * x2 + x1 + u
#         return x1dot, x2dot
#     def odefun(self, t, x, u):
#         x1 = x[:, 0] if x.ndim == 2 else x[0]
#         x2 = x[:, 1] if x.ndim == 2 else x[1]
#
#         if u is not 0:
#             u =  u[:, 0] if u.ndim==2 else u[0]
#
#         x1dot, x2dot = self.get_x_dot(x=(x1,x2),u=u)
#
#         output = torch.stack([x1dot, x2dot], dim=1) if x.ndim == 2 else torch.stack([x1dot, x2dot])
#         return output
#
#     def forward(self, x, u=0):
#         x_next = rk4_step_control(func = self.odefun, t=0, x=x, u=u, delta_t=self.delta_t)
#         return x_next
#     def get_do_mpc_model(self):
#         model_type = "discrete"
#         do_mpc_model = do_mpc.model.Model(model_type)
#
#         self.x = do_mpc_model.set_variable(var_type='_x', var_name='x', shape=(self.num_states, 1))
#         self.u = do_mpc_model.set_variable(var_type='_u', var_name='u', shape=(self.num_inputs, 1))
#
#         # Define continuous dynamics
#         def odefun(t, x, u):
#             x1 = x[0]
#             x2 = x[1]
#             x1dot, x2dot = self.get_x_dot(x=(x1, x2), u=u)
#             return np.array([x1dot, x2dot])
#
#         # Implement RK4 for discretization
#         x_next = rk4_step_control(func = odefun, t=0, x=self.x, u=self.u, delta_t=self.delta_t)
#         do_mpc_model.set_rhs('x', x_next)
#
#         do_mpc_model.setup()
#
#         return do_mpc_model

# class CSTRModel(DiscretizedODEModel):
#     def __init__(self, mu=1, delta_t=0.1):
#         super().__init__()
#         self.name= "CSTR Model"
#         self.delta_t = delta_t
#         self.mu = mu
#         self.num_states = 4
#         self.num_inputs = 2
#     def get_x_dot(self,x,u):
#         x1 = x[0]
#         x2 = x[1]
#         u1 = u[0] if u is not None else 0
#         x1dot = -x2
#         x2dot = self.mu * (-1 + x1**2) * x2 + x1 + u1
#         return x1dot, x2dot
#
#     def forward(self, x, u=None):
#         x_next = rk4_step_control(func = self.odefun, t=0, x=x, u=u, delta_t=self.delta_t)
#         return x_next





if __name__ == "__main__":

    import scienceplots
    plt.style.use(['science', 'ieee'])
    delta_t = 0.1
    steps = 100
    model = VanDerPolDiscreteControl(mu = 1, delta_t=delta_t)

    x0 = create_combination_tensor(lower_bounds = [-1, -1],
                                   upper_bounds=[1,1],
                                   step_sizes = [0.5,0.5])

    # x0 shape : num_trajectories, num_states
    simulation = Simulator(model, steps=steps, x0=x0)
    x = simulation.rollout()
    # x shape : num_trajectories, num_steps, num_states
    num_trajectories = x0.shape[0]

    # Adding control inputs:
    u = random_wave_input(num_trajectories=num_trajectories, steps=steps, delta_t=delta_t, num_inputs=1, max_input=1,
                          min_input=-1, max_freq=50, min_freq=1)

    fig, ax = plt.subplots(2,1,figsize=(3,4), gridspec_kw={'height_ratios': [1, 2]})
    plot_pplane(ax[1], x=x, color='k', linestyle='solid', linewidth=1, label='With no disturbances')

    # u shape: num_trajectories, num_steps, num_inputs

    x_disturbed = simulation.rollout(u=u)
    plot_pplane(ax[1], x=x_disturbed, color='b', linestyle='--', linewidth=0.5, label='With random disturbances')
    plot_dynamics(ax=ax[0], x=x_disturbed, trajectory_index=0, dimension_index=1)
    plot_dynamics(ax=ax[0], x=x, trajectory_index=0, dimension_index=1)
    plot_dynamics(ax=ax[0], x=u, trajectory_index=0, dimension_index=0)

    fig.show()
