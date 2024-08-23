from utils import *
import time

def plot_pplane(ax=None, x=None,
                color = None, linestyle=None,
                linewidth=None, label='',
                *args, **kwargs):
    if x is not None:
        if type(x) is not np.ndarray:
            x_np = x.cpu().numpy() if x is not None else 0
        else:
            x_np = x
        if ax is not None:
            ax.plot(x_np[:, :, 0].T, x_np[:, :, 1].T, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.5, *args, **kwargs)
            ax.plot([], [], color=color, linestyle=linestyle, label=label, linewidth=linewidth, *args, **kwargs)
        else:
            fig, ax = plt.subplots(1,1, figsize=(4,4))
            ax.plot(x_np[:, :, 0].T, x_np[:, :, 1].T, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.5,
                     *args, **kwargs)
            ax.plot([], [], color=color, linestyle=linestyle, label=label, linewidth=linewidth, *args, **kwargs)
            ax.legend()
            fig.show()


def plot_dynamics(ax = None,x=None, trajectory_index=0, dimension_index = 0, label='', *args, **kwargs):
    if type(x) is not np.ndarray:
        x_np = x.cpu().numpy() if x is not None else 0
    else:
        x_np = x
    if trajectory_index == 'all':
        if ax is not None:
            ax.plot(x_np[:, :, dimension_index].T, label=label, *args, **kwargs)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(x_np[:, :, dimension_index].T, label=label, *args, **kwargs)
            ax.legend()
            fig.show()
    else:
        if ax is not None:
            ax.plot(x_np[trajectory_index, :, dimension_index], label=label, *args, **kwargs)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(x_np[trajectory_index, :, dimension_index], label=label, *args, **kwargs)
            ax.legend()
            fig.show()
def plot_all_dynamics(ax = None,
                      x=None,
                      color = 'black',
                      linestyle='solid',
                      linewidth=0.5,
                      label='',
                      num_states=None,
                      variable_name='x',
                      *args, **kwargs):
    if num_states is None:
        num_states = x.shape[-1]
    ax[num_states//2].plot([], [], color=color, linestyle=linestyle, label=label, linewidth=linewidth,
                           *args, **kwargs)
    ax[num_states//2].legend(loc='center left', bbox_to_anchor=(1.05,0.5))
    for i in range(num_states):
        plot_dynamics(ax=ax[i],
                      x=x,
                      trajectory_index='all',
                      dimension_index=i,
                      color=color,
                      linestyle=linestyle,
                      linewidth=linewidth,
                      *args, **kwargs)
        ax[i].set_ylabel(f"${variable_name}_{i+1}$")
        ax[i].set_xlabel("$t$")


class Simulator:
    """Discrete Dynamical System with/without control inputs Simulator"""
    def __init__(self, model, steps, x0):
        self.model = model
        self.steps = steps
        self.x0 = x0

    def rollout(self, u=None):
        s1 = time.time()
        outputs = [self.x0.unsqueeze(1)]
        for step in range(1, self.steps):
            if u is None:
                next_x = self.model(outputs[-1].squeeze(1))
            else:
                next_x = self.model(outputs[-1].squeeze(1), u[:, step - 1, :])
            outputs.append(next_x.unsqueeze(1))
        # Stack all the outputs to create a new tensor
        x = torch.cat(outputs, dim=1)
        time_taken = time.time() - s1

        return x
    # def __init__(self, model, steps, x0):
    #     self.model = model
    #     self.steps = steps
    #     self.x = torch.stack([x0] * steps)
    #     self.x = self.x.permute(1, 0, 2)
    #     self.x0 = x0
    #
    # def rollout(self, u= None):
    #     x = self.x.clone()
    #     for step in range(1, self.steps):
    #         if u is None:
    #             x[:,step,:] = self.model(x[:, step - 1, :])
    #         else:
    #             x[:,step,:] = self.model(x[:, step - 1, :], u[:, step - 1, :])
    #     self.x = x.clone()
    #     return x


