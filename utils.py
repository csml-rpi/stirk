import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torchdiffeq import odeint_adjoint
import control
import scipy as sp
import random
import pickle

import argparse



# os related functions
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def create_next_trial_folder(directory, kw=''):
    i = 1
    while True:
        if kw =="":
            folder_name = f"trial_{i}"
        else:
            folder_name = f"{kw}_{i}"
        folder_path = os.path.join(directory, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            break
        else:
            i += 1
    return folder_path
def create_next_final_folder(directory):
    folder_name = "final"
    folder_path = os.path.join(directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def save_file(var, filename, path='./'):
    file_path = os.path.join(path,filename)
    with open(file_path, 'wb') as f:
        pickle.dump(var, f)
def load_file(filename, path='./'):
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        var = pickle.load(f)
    return var
# def load_file(filename, path):
#     file_path = os.path.join(path, filename)
#     if not os.path.exists(file_path):
#         return None
    
#     with open(file_path, 'rb') as f:
#         try:
#             # Attempt to load the file with pickle
#             var = pickle.load(f)
#             return var
#         except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError) as pickle_error:
#             try:
#                 # If pickle fails, reset the file pointer and try torch.load
#                 f.seek(0)
#                 var = torch.load(f, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
#                 return var
#             except Exception as torch_error:
#                 # If both pickle and torch fail, raise an informative error
#                 raise RuntimeError(f"Failed to load file {filename}. Pickle error: {pickle_error}, Torch error: {torch_error}")
def disp_dict(dictionary):
    string = ""
    for key, value in dictionary.items():
        string += f"{key} = {value}\n"
    return string
# visualization functions
COLORS = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown',
         'black', 'gray', 'magenta', 'olive', 'teal', 'navy']
def choose_color(available_colors):
    if len(available_colors) == 0:
        available_colors = [
            'red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown',
            'black', 'gray', 'magenta', 'olive', 'teal', 'navy'
        ]
    color = random.sample(available_colors, 1)[0]
    available_colors.remove(color)
    return color, available_colors

# Example control inputs
def random_control_input(num_trajectories, steps, num_inputs, max_input, min_input):
    u = (max_input - min_input) * torch.rand(size=(num_trajectories, steps - 1, num_inputs)) + min_input
    return  u

# def sine_wave_input(num_trajectories, steps, delta_t, num_inputs, max_input, min_input, max_freq, min_freq):
#     amp = (max_input - min_input) * torch.rand(size=(num_inputs,)) + min_input
#     freq = (max_freq - min_freq) * torch.rand(size=(num_inputs,)) + min_freq
#     # t_u = torch.arange(0, round(delta_t * (steps - 1), 2), delta_t)
#     t_u = torch.linspace(0, delta_t*(steps-1), steps=steps)[:-1]
#     u_single_traj = amp * torch.sin(freq * t_u) + amp * torch.cos(freq * t_u)
#     u = u_single_traj.unsqueeze(0).unsqueeze(2).expand(num_trajectories, steps - 1, num_inputs)
#     return u

def split_tensors(data_tensors, num_first_split):
    """
    Split a tensor or each tensor in a tuple of tensors into two parts based on the specified number of items for the first part.
    All tensors are split consistently using the same indices.

    Parameters:
    - data_tensors: A single torch.Tensor or a tuple of torch.Tensor. The tensor(s) to be split.
    - num_first_split (int): The number of items in the first split.

    Returns:
    - If a single tensor is provided: Two torch.Tensor objects representing the first and second parts of the split.
    - If a tuple of tensors is provided: Two tuples of torch.Tensor objects, each containing the corresponding parts of the split.
    """
    # Handle both single tensor and tuple of tensors
    is_single_tensor = not isinstance(data_tensors, tuple)

    # If it's a single tensor, make it a tuple for consistent processing
    tensors = (data_tensors,) if is_single_tensor else data_tensors

    # Generate shuffled indices from the size of the first tensor
    indices = torch.randperm(tensors[0].size(0))

    # Split indices for the first and second parts
    first_split_indices = indices[:num_first_split]
    second_split_indices = indices[num_first_split:]

    # Split each tensor using the indices
    first_splits = tuple(tensor[first_split_indices] for tensor in tensors)
    second_splits = tuple(tensor[second_split_indices] for tensor in tensors)

    # Return format based on input type
    if is_single_tensor:
        return first_splits[0], second_splits[0]
    else:
        return first_splits, second_splits
def sine_wave_input(num_trajectories, steps,
                    delta_t,
                    num_inputs,
                    max_input,
                    min_input,
                    max_freq,
                    min_freq,
                    min_bias=0,
                    max_bias=0,
                    max_decay=10,
                    min_decay=0):
    t = torch.linspace(0, (steps - 1) * delta_t, steps)
    output = torch.zeros(num_trajectories, steps - 1, num_inputs)
    for traj in range(num_trajectories):
        for input_idx in range(num_inputs):
            amplitude = (max_input - min_input) * torch.rand(1) + min_input
            frequency = (max_freq - min_freq) * torch.rand(1) + min_freq
            bias = (max_bias - min_bias) * torch.rand(1) + min_bias
            phase = 2*torch.pi * torch.rand(1) - torch.pi
            decay = (max_decay - min_decay)*torch.rand(1) + min_decay
            sine_wave = amplitude * torch.exp(-decay*t) * torch.sin(2 * np.pi * frequency * t + phase) + bias
            output[traj, :, input_idx] = sine_wave[:-1]
    return output

def step_input(num_trajectories, steps,
                    delta_t,
                    num_inputs,
                    max_input,
                    min_input):
    t = torch.linspace(0, (steps - 1) * delta_t, steps)
    output = torch.zeros(num_trajectories, steps - 1, num_inputs)
    for traj in range(num_trajectories):
        for input_idx in range(num_inputs):
            amplitude = (max_input - min_input) * torch.rand(1) + min_input
            constant_input = amplitude * torch.ones_like(t)
            output[traj, :, input_idx] = constant_input[:-1]
    return output

def random_wave_input(num_trajectories,
                      steps,
                      delta_t,
                      num_inputs,
                      max_input,
                      min_input,
                      max_freq,
                      min_freq,
                      min_bias=0,
                      max_bias=0,
                      max_decay=10,
                      min_decay=0):
    u = sine_wave_input(num_trajectories=num_trajectories,
                        steps=steps,
                        delta_t=delta_t,
                        num_inputs=num_inputs,
                        max_input=max_input,
                        min_input=min_input,
                        max_freq=max_freq,
                        min_freq=min_freq,
                        min_bias=min_bias,
                        max_bias=max_bias,
                        max_decay=max_decay,
                        min_decay=min_decay)
    for i in range(199):
        u += sine_wave_input(num_trajectories=num_trajectories,
                            steps=steps,
                            delta_t=delta_t,
                            num_inputs=num_inputs,
                            max_input=max_input,
                            min_input=min_input,
                            max_freq=max_freq,
                            min_freq=min_freq,
                            min_bias=min_bias,
                            max_bias=max_bias,
                            max_decay=max_decay,
                            min_decay=min_decay)
    u = u /np.sqrt(200)
    # u = u_single_traj_normed.unsqueeze(0).unsqueeze(2).expand(num_trajectories, steps - 1, num_inputs)
    return u


# Root finding
def find_roots(function, x1lim = (-1, 5), x2lim =(-1, 15), num_starting_points = 100):
    init_space = gridspace(*x1lim,*x2lim, num_starting_points).cpu().numpy()
    def function_np(x):
        x_np = np.array(x)
        x_torch = torch.from_numpy(x_np).cuda()
        y_torch = function(x_torch)
        y_np = y_torch.cpu().detach().numpy()
        return y_np
    roots = []
    for x0 in init_space:
        result = sp.optimize.root(function_np, x0)
        if result.success:
            root = result.x
            roots.append(root)
    roots = np.array(roots)
    return np.unique(roots, axis=0)

# discrete systems
def advance_linear_system(A, x0s, roll_out_length):
    traj, states = x0s.shape
    xs = torch.empty(size=(traj, roll_out_length, states), dtype=x0s.dtype, device= x0s.device)
    for r in range(1, roll_out_length +1):
        xs[:,r-1,:] = torch.matmul(x0s, torch.matrix_power(A,r).T)
    return xs
def find_fixed_points_discrete(function, x1lim = (-1, 5), x2lim =(-1, 15), num_starting_points = 100):
    init_space = gridspace(*x1lim,*x2lim, num_starting_points).cpu().numpy()
    def fixed_points_equation(function):
        return lambda x: function(x) - x
    def function_np(x):
        x_np = np.array(x)
        x_torch = torch.from_numpy(x_np).cuda()
        y_torch = fixed_points_equation(function)(x_torch)
        y_np = y_torch.cpu().detach().numpy()
        return y_np
    roots = []
    for x0 in init_space:
        result = sp.optimize.root(function_np, x0, tol=1e-4)
        if result.success:
            root = result.x
            root[np.isclose(root, 0)] = 0.0
            roots.append(root)
    roots = np.array(roots)
    return np.unique(roots, axis=0)


# Batch and data processing
def generate_random_integers(start, end, count):
    random_integers = []
    for _ in range(count):
        random_integers.append(random.randint(start, end))
    return random_integers



def gridspace_step(lb1, ub1, lb2, ub2, step_size):
    x1 = torch.arange(lb1, ub1, step_size).cuda().type(torch.float64)
    x2 = torch.arange(lb2, ub2, step_size).cuda().type(torch.float64)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    x = torch.stack([X1, X2], dim=0)
    x = x.reshape(2,-1)
    return x.t()


def gridspace(lb1, ub1, lb2, ub2, num_points):
    x1 = torch.linspace(lb1, ub1, num_points).cuda().type(torch.float64)
    x2 = torch.linspace(lb2, ub2, num_points).cuda().type(torch.float64)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    x = torch.stack([X1, X2], dim=0)
    x = x.reshape(2,-1)
    return x.t()

def create_combination_tensor(lower_bounds, upper_bounds, step_sizes):
    column_values = [torch.arange(lower, upper + step, step) for lower, upper, step in
                     zip(lower_bounds, upper_bounds, step_sizes)]
    meshgrids = torch.meshgrid(*column_values, indexing='ij')
    tensor_combinations = torch.stack([grid.flatten() for grid in meshgrids], dim=-1)
    return tensor_combinations

def get_X1_X2(x):
    m = int(np.sqrt(x.shape[0]))
    XX = x.t().reshape(2,m,m)
    return XX

def get_segmented_data(x, u=None, roll_out_length=1):
    num_trajectories, timesteps, state_dim = x.shape
    num_segments = timesteps - roll_out_length
    # Pre-allocate tensor memory
    final_states_shape = (num_trajectories * num_segments, roll_out_length + 1, state_dim)
    final_states = torch.empty(final_states_shape, dtype=x.dtype, device = x.device)
    # Create an index matrix that will help in picking out the right slices
    idx_matrix_states = torch.arange(0, num_segments).view(-1, 1) + torch.arange(0, roll_out_length + 1).view(1, -1)

    for i, trajectory in enumerate(x):
        # Use the index matrix to get the segments from the trajectory
        segments = trajectory[idx_matrix_states]
        # Place the segments in the final_data tensor
        final_states[i * num_segments:(i + 1) * num_segments] = segments
    if u is not None:
        num_trajectories_u, timesteps_u, input_dim = u.shape
        assert num_trajectories == num_trajectories_u
        assert timesteps == timesteps_u + 1
        final_input_shape = ((num_trajectories * num_segments), roll_out_length, input_dim)
        final_inputs = torch.empty(final_input_shape, dtype=u.dtype, device=u.device)
        idx_matrix_inputs = torch.arange(0, num_segments).view(-1, 1) + torch.arange(0, roll_out_length).view(1, -1)
        for i, trajectory in enumerate(u):
            # Use the index matrix to get the segments from the trajectory
            segments = trajectory[idx_matrix_inputs]
            # Place the segments in the final_data tensor
            final_inputs[i * num_segments:(i + 1) * num_segments] = segments
        return final_states, final_inputs
    final_inputs = None
    return final_states, final_inputs



## Discretization

def rk4_step(func, t, x, delta_t):
    k1 = delta_t * func(t, x)
    k2 = delta_t * func(t + 0.5 * delta_t, x + 0.5 * k1)
    k3 = delta_t * func(t + 0.5 * delta_t, x + 0.5 * k2)
    k4 = delta_t * func(t + delta_t, x + k3)

    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x_next

def rk4_step_control(func, t, x, u, delta_t):
    k1 = delta_t * func(t, x, u)
    k2 = delta_t * func(t + 0.5 * delta_t, x + 0.5 * k1, u)
    k3 = delta_t * func(t + 0.5 * delta_t, x + 0.5 * k2, u)
    k4 = delta_t * func(t + delta_t, x + k3, u)

    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x_next


# Batch


def get_batch(t, xs, batch_time, batch_size):
    batch_xs = []
    batch_x0s = []
    batch_ts = []
    data_size = t.shape[0]
    for x in xs:
        random_index = torch.from_numpy(
            np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
        batch_x0 = x[random_index].type(torch.float64)
        batch_t = t[:batch_time]
        batch_x = torch.stack([x[random_index + i] for i in range(batch_time)], dim=0)
        batch_xs.append(batch_x.cuda())
        batch_x0s.append(batch_x0.cuda())
        batch_ts.append(batch_t.cuda())
    return batch_x0s, batch_ts, batch_xs
def get_batch_new(t, xs, batch_time, batch_size):
    data_size = t.shape[0]
    random_indices = torch.randperm(data_size - batch_time)[:batch_size].cuda()
    if type(xs) == list:
        xs = torch.stack(xs)  # Shape: [num_trajectories, data_size, 2]
    batch_indices = torch.arange(batch_time).unsqueeze(1).cuda()
    batch_xs = xs[:, (random_indices + batch_indices).t()]  # Shape: [num_trajectories, batch_time, batch_size, 2]
    batch_xs = batch_xs.permute(2, 1, 0, 3).reshape(batch_time, -1, xs.shape[2]) # Shape: [batch_time, num_trajectories*batch_size, num_states]
    shuffled_indices = torch.randperm(batch_xs.shape[1])
    batch_xs = batch_xs[:,shuffled_indices,:]
    batch_x0s = batch_xs[0,:,:]
    batch_t = t[:batch_time]
    return batch_x0s, batch_t, batch_xs

# def get_batch_discrete(xs, batch_time, batch_size):
#     xs = xs.permute(1,0,2)
#     data_size = xs.shape[1]
#     random_indices = torch.randperm(data_size - batch_time)[:batch_size].cuda()
#     batch_indices = torch.arange(batch_time).unsqueeze(1).cuda()
#     batch_xs = xs[:, (random_indices + batch_indices).t()]  # Shape: [num_trajectories, batch_time, batch_size, 2]
#     batch_xs = batch_xs.permute(2, 1, 0, 3).reshape(batch_time, -1, xs.shape[2]) # Shape: [batch_time, num_trajectories*batch_size, num_states]
#     shuffled_indices = torch.randperm(batch_xs.shape[1])
#     batch_xs = batch_xs[:,shuffled_indices,:]
#     batch_x0s = batch_xs[0,:,:]
#
#     return batch_xs

def get_batch_discrete_traj(x, u= None, batch_size=10):
    traj = x.shape[0]
    select_idx = torch.randperm(traj)[:batch_size]
    batch_x = x[select_idx]
    batch_x0 = batch_x[:, 0, :]
    if u is not None:
        batch_u = u[select_idx]
        return batch_x, batch_x0, batch_u
    return batch_x, batch_x0


def get_all_batches_discrete_traj(x, u=None, batch_size=10):
    traj = x.shape[0]
    batches = []
    random_indices = torch.randperm(traj)
    for i in range(0, traj, batch_size):
        select_idx = torch.arange(i, min(i + batch_size, traj))
        batch_x = x[random_indices[select_idx]]
        batch_x0 = batch_x[:, 0, :]
        if u is not None:
            batch_u = u[random_indices[select_idx]]
            batches.append((batch_x, batch_x0, batch_u))
        else:
            batches.append((batch_x, batch_x0, None))

    return batches

def get_batch_discrete(xs, batch_time, batch_size):
    data_size = xs.shape[0]
    states = xs.shape[2]
    random_index = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_xs = []
    for r in random_index:
        batch_x = xs[r:r+batch_time,:,:]
        batch_xs.append(batch_x.cuda())
    batch_xs = torch.stack(batch_xs) # shape batch_size, batch_time, trajectories, state
    batch_xs = batch_xs.permute(1,0,2,3) # batch_time, batch_size, trajectories, state
    batch_xs = batch_xs.reshape(batch_time,-1,states) # batch_time, batch_size * trajectories, state
    batch_x0s = batch_xs[0,:,:]
    # batch_xs = batch_xs.permute(1,0,2)
    return batch_xs, batch_x0s

#

def matrix_exp_taylor(A, terms=20):
    """
    Approximate the matrix exponential using Taylor series expansion.

    Args:
    A (torch.Tensor): The square matrix for which to compute the exponential.
    terms (int): The number of terms in the Taylor series expansion.

    Returns:
    torch.Tensor: The approximated exponential of matrix A.
    """
    n = A.size(0)
    A_exp = torch.eye(n, dtype=A.dtype, device=A.device)  # Start with the identity matrix
    A_power = torch.eye(n, dtype=A.dtype, device=A.device)  # A^0
    factorial = 1.0  # 0!

    for i in range(1, terms):
        A_power = torch.matmul(A_power, A) / i
        A_exp += A_power

    return A_exp

def int_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer or 'None'")
def float_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a float or 'None'")

def load_data_files(system, with_control, traj, seed, noise_level =None, noise = None, num_traj=50, add_suboptimal_data=False, strategy= 'dmd', observer = 'RBF'):
    if noise is None:
        noise_values = np.logspace(-3, -1, 10)
        if traj=='single':
            num_traj = 1
        noise = noise_values[noise_level]
    filepath = f"./dataset/{system}_{traj}{'_control' if with_control else ''}/noise_{noise:0.4f}/seed_{seed}/num_traj_{num_traj}/"
    train_data = load_file('train_data.pkl', path=filepath)
    test_data = load_file('test_data.pkl', path=filepath)
    val_data = load_file('val_data.pkl', path =filepath)
    if add_suboptimal_data:
        if observer =='rbf':
            observer = 'RBF'
        if observer =='polyflow':
            observer = 'Polyflow'
        if observer == 'identity':
            observer = 'Identity'
        suboptimal_data = load_file(
            f"{'CartPole2' if system == 'cartpole2' else 'Van Der Pol Discrete Control'}/"
            f"{traj}_traj/control/{strategy}/{observer}/noise_ {noise:0.4f}/adaptive/"
            f"num_traj_{num_traj}_1/suboptimal_trajectories_data.pkl"
        )
        train_data['x0'] = torch.cat([train_data['x0'], suboptimal_data['x0_test']], dim=0)
        train_data['x'] = torch.cat([train_data['x'], suboptimal_data['x_test']], dim=0)
        train_data['x_noisy'] = torch.cat([train_data['x_noisy'], suboptimal_data['x_test']], dim=0)
        train_data['u'] = torch.cat([train_data['u'], suboptimal_data['u_test'][:,:-1,:]], dim=0)

    return train_data, test_data, val_data