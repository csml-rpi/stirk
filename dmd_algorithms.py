import torch
import numpy as np
import matplotlib.pyplot as plt
from time_delay import time_delay
from models import *
import os
from scipy.linalg import sqrtm

def perform_pod(snapshot_matrix, number_of_modes):
    """
    Perform Proper Orthogonal Decomposition (POD) on the given snapshot matrix.

    :param snapshot_matrix: torch.Tensor, the snapshot matrix to be decomposed.
    :param number_of_modes: int, the number of modes to be retained in the decomposition.
    :return: torch.Tensor, the reconstructed field using the selected modes.
    """
    # Step 2: Compute Covariance Matrix
    covariance_matrix = snapshot_matrix.T @ snapshot_matrix

    # Step 3: Solve Eigenvalue Problem
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Step 4: Sort Eigenvalues and Eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Determine Significant Modes
    selected_modes = sorted_eigenvectors[:, :number_of_modes]

    # Step 6: Project Data onto Modes
    coefficients = snapshot_matrix @ selected_modes

    # Step 7: Reconstruct Field
    reconstructed_field = coefficients @ selected_modes.T

    return reconstructed_field


class DMDBase:
    def __init__(self, observable):
        self.x_noisy = None
        self.xi0_true = None
        self.u_true = None
        self.steps = None
        self.x0_true = None
        self.x_true = None
        self.observable = observable
        self.name = None
        self.x_pred = None
        self.x_true_test = None
        self.x0_true_test = None
        self.steps_test = None
        self.u_true_test = None
        self.x_noisy_test = None
        self.test_data_available = None
        self.x_val = None
        self.x0_val = None
        self.u_val = None
        self.x_noisy_val = None
        self.val_data_available = None

    def setup_comparison(self, x_true,x_noisy, x0_true, steps, u_true=None,plot_type = 'plot_dynamics'):
        self.x_true = x_true
        self.x0_true = x0_true
        self.steps = steps
        self.u_true = u_true
        self.x_noisy = x_noisy
        self.plot_type = plot_type # "pplane", "plot_dynamics"
        return self
    def add_test_data(self, x_true, x_noisy, x0_true, u_true=None):
        self.x_test = x_true
        self.x0_test = x0_true
        self.u_test = u_true
        self.x_noisy_test = x_noisy
        self.test_data_available = True if x0_true is not None else False

    def add_validation_data(self, x, x_noisy, x0, u=None):
        self.x_val = x
        self.x0_val = x0
        self.u_val = u
        self.x_noisy_val = x_noisy
        self.val_data_available = True if x0 is not None else False

    def compare(self, model=None, A=None, epoch=None, filepath=None, B=None, plot=True, show = True,
                x_true=None, x0_true=None, steps=None, u_true=None, x_noisy=None, tag = "training"):
        if model is not None:
            xi0_true = self.observable.transform(self.x0_true if x0_true is None else x0_true)
            xi = model.rollout(x0 = xi0_true,
                               u=self.u_true if u_true is None else u_true,
                               step=self.steps if steps is None else steps)
        else:
            if self.x_pred is None:
                model = LinearModelDiscrete(A=A, B=B)
                xi0_true = self.observable.transform(self.x0_true if x0_true is None else x0_true)
                # xi0_true = self.normalize_xi.transform(xi0_true)
                simulation = Simulator(model=model, steps=self.steps if steps is None else steps, x0=xi0_true)
                # u_true= self.normalize_u.transform(self.u_true)
                xi = simulation.rollout(u=self.u_true if u_true is None else u_true)
                # xi = self.normalize_xi.inverse_transform(xi)
            else:
                xi = self.x_pred # this is for optdmd only
        num_states = self.x_true.shape[-1]
        error = (torch.norm(xi[:, :, :num_states] - (self.x_true if x_true is None else x_true))/
                 torch.norm(self.x_true if x_true is None else x_true))
        if plot:
            if self.plot_type == 'pplane':
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                plot_pplane(
                    ax=ax,
                    x=self.x_true if x_true is None else x_true,
                    color='blue',
                    linestyle='solid',
                    label='Ground Truth',
                )
                if not(x_true is not None and x_noisy is None):
                    plot_pplane(
                        ax=ax,
                        x=self.x_noisy if x_noisy is None else x_noisy,
                        color='blue',
                        linestyle='None',
                        marker='o',
                        markersize=1.5,
                        label=f'Data',
                    )
                plot_pplane(ax=ax,
                            x=xi,
                            color='red',
                            linestyle='dashed',
                            label=f'{self.name} with {self.observable.description}, error : {error}',
                            marker='o',
                            markersize=1.2)
                ax.legend(loc='center left',
                          bbox_to_anchor=(1.05, 0.5))
                if show:
                    fig.show()
                if filepath is not None:
                    filename = f"{self.name}_{tag}" if epoch is None else f'epoch_{tag}_{epoch}'
                    fig.savefig(f'{filepath}/{filename}.png', format='png', dpi=300)
            elif self.plot_type == 'plot_dynamics':
                fig, ax = plt.subplots(num_states, 1, figsize=(5, 2*num_states))
                plot_all_dynamics(ax=ax,
                                    x=self.x_true if x_true is None else x_true,
                                    color='blue',
                                    linestyle='solid',
                                    label='Ground Truth')
                if not (x_true is not None and x_noisy is None):
                    plot_all_dynamics(
                        ax=ax,
                        x=self.x_noisy if x_noisy is None else x_noisy,
                        color='blue',
                        linestyle='None',
                        marker='o',
                        markersize=1.5,
                        label=f'Data',
                    )
                num_states = self.x_true.shape[-1]
                plot_all_dynamics(ax=ax,
                            x=xi,
                            color='red',
                            linestyle='dashed',
                            label=f'{self.name} with {self.observable.name} basis with {self.observable.order}, error : {error}',
                            marker='o',num_states=num_states,
                            markersize=1.2)
                if show:
                    fig.show()
                if filepath is not None:
                    filename = f"{self.name}_{tag}" if epoch is None else f'epoch_{tag}_{epoch}'
                    fig.savefig(f'{filepath}/{filename}.png', format='png', dpi=300)

        if filepath is not None:
            save_file(error.cpu().numpy(), f"trajectory_error_{tag}.pkl", filepath)
            trajectory_info = dict(x_true=self.x_true.cpu().numpy(), x_pred = xi[:, :, :num_states].cpu().numpy())
            save_file(trajectory_info, f"trajectory_info_{tag}.pkl", filepath)
            # if A is not None and B is not None:
            #     save_file(dict(A=A,B=B),"model_params.pkl",filepath)
        return error.cpu().numpy()
    def fit(self, x, u = None):
        pass


class DMD(DMDBase):
    def __init__(self, observable):
        super().__init__(observable)
        self.name = "eDMD"


    def fit(self, x, u = None):
        data, u = get_segmented_data(x, u, roll_out_length=1)
        lifted_data  = self.observable.transform(data.reshape(-1,data.shape[-1])).reshape(data.shape[0], data.shape[1], -1)
        X = lifted_data[:, 0, :].t()
        Y = lifted_data[:, 1, :].t()
        control_inputs = u[:, 0, :].t() if u is not None else None

        if control_inputs is None:
            U,S,Vh = torch.linalg.svd(X, full_matrices=False)
            A = Y @ Vh.H @ torch.diag(1/S) @ U.H  #U.H @ Y @ Vh.H @ torch.diag(1 / S)
            B = None
        else:
            num_states = X.shape[0]
            num_inputs = control_inputs.shape[0]
            input_space = torch.cat([X, control_inputs], dim=0)
            U,S,Vh = torch.linalg.svd(input_space, full_matrices=False)
            # Uy, _, _ = torch.linalg.svd(Y, full_matrices = False)
            U1 = U[:num_states]
            U2 = U[num_states:]
            A = Y @ Vh.H @ torch.diag(1 / S) @ U1.H # Uy.H @ Y @ Vh.H @ torch.diag(1 / S) @ U1.H @ Uy
            B = Y @ Vh.H @ torch.diag(1 / S) @ U2.H #Uy.H @ Y @ Vh.H @ torch.diag(1 / S) @ U2.H
        return A, B

class TLSDMD(DMDBase):
    def __init__(self, observable, rank =None):
        super().__init__(observable)
        self.rank =  rank
        self.name = "TLSDMD"
    def fit(self, x, u = None):
        data, u = get_segmented_data(x, u, roll_out_length=1)
        lifted_data = self.observable.transform(data)
        X = lifted_data[:, 0, :].t()
        Y = lifted_data[:, 1, :].t()
        data_new = torch.cat([X, Y], dim=0)
        #TODO: Include control to tlsdmd
        control_inputs = u[:, 0, :].t() if u is not None else None
        # POD modes
        n = X.shape[0] # num_states
        m = X.shape[1] # timesteps
        r = self.rank if self.rank is not None else n
        if n > m//2:
            data_new = perform_pod(data_new, number_of_modes = r)
        else:
            r = n

        U ,_,_ = torch.linalg.svd(data_new)
        U11 = U[:r,:r]
        U21 = U[r:,:r]
        A = U21 @ torch.linalg.pinv(U11)
        return A, None



class FBDMD(DMDBase):
    def __init__(self, observable):
        super().__init__(observable)
        self.name = "FBDMD"
    def fit(self, x, u = None):
        # TODO: Include control version
        data, u = get_segmented_data(x, u, roll_out_length=1)
        lifted_data = self.observable.transform(data)
        X = lifted_data[:, 0, :].t()
        Y = lifted_data[:, 1, :].t()
        # Forward dmd:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        A_f = Y @ Vh.H @ torch.diag(1/S) @ U.H #U.H @ Y @ Vh.H @ torch.diag(1/S)
        # Backward dmd:
        U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        A_b_inv = X @ Vh.H @ torch.diag(1 / S) @ U.H  #U.H @ X @ Vh.H @ torch.diag(1 / S)
        A_b = torch.linalg.pinv(A_b_inv)
        # Convert matrices to numpy
        A_f_np = A_f.cpu().numpy()
        A_b_np = A_b.cpu().numpy()
        A_np = np.real(sqrtm(A_f_np @ A_b_np))
        A = torch.from_numpy(A_np).to(device= X.device, dtype=X.dtype)
        return A, None



class OptDMD(DMDBase):
    def __init__(self, observable, delta_t):
        super().__init__(observable)
        self.x_pred = None
        self.name = "OptDMD"
        self.delta_t = delta_t

    def fit(self, x, u = None):
        lifted_x = self.observable.transform(x)
        xdata = lifted_x.squeeze().T.cpu().numpy()
        steps = lifted_x.shape[1]
        num_states = lifted_x.shape[-1]
        ts = np.arange(0, steps * self.delta_t, self.delta_t)[None,:]
        r = 2
        imode = 0
        self.w, self.e, self.b = optdmd(xdata, ts, r, imode)
        # A_np = self.w @ np.diag(self.e) @ np.linalg.pinv(self.w)
        # A = torch.from_numpy(A_np).to(device=device, dtype=dtype)
        x_pred_np = unroll_dynamics_optdmd(w=self.w,lambda_vals=self.e,b=self.b, timesteps=steps ,delta_t=self.delta_t)
        self.x_pred = torch.from_numpy(x_pred_np).to(device=x.device, dtype=x.dtype).t().unsqueeze(0)
        return None, None

