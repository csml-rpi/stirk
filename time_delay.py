from models import *


def time_delay(model, order):
    f = lambda x: model(x)  # if model takes x and u, u is set to 0. if model takes only x, u
    if order == 0:
        return lambda x: x
    else:
        time_delay_func = time_delay(model=model, order=order - 1)
        td = lambda x: f(time_delay_func(x))
        return td

class Timedelay:
    def __init__(self, gt_model,order,delta_t):
        self.gt_model = gt_model
        self.order = order
        self.delta_t = delta_t
        self.name = "Polyflow"
    def transform(self, x, order=None):
        if order is None:
            order = self.order
        xi = []
        for N in range(order):
            time_delay_func = time_delay(model=self.gt_model, order=N)
            time_delay_value = torch.vmap(time_delay_func)(x) if x.ndim >= 2 else time_delay_func(x)
            xi.append(time_delay_value)
        return torch.hstack(xi) if x.ndim <= 2 else torch.cat(xi, dim=2)

class TimedelayImproved:
    def __init__(self, gt_model,order,delta_t):
        self.gt_model = gt_model
        self.order = order
        self.delta_t = delta_t
        self.name = "Polyflow"
        self.description = f"Polyflow basis function with order {order}"
    
    def transform(self, x, order=None):
        if x.ndim ==2:
            xi = self._transform(x,order=order)
        elif x.ndim ==3:
            xi = self._transform(x.reshape(-1, x.shape[-1])).reshape(x.shape[0],x.shape[1],-1)
        else:
            raise ValueError("Dimension of x is not 2 or 3")
        return xi
    
    def _transform(self, x, order=None):
        if order is None:
            order = self.order
        outputs = [x]
        for N in range(order - 1):
            x_next = self.gt_model(outputs[-1])
            outputs.append(x_next)

        xi = torch.cat(outputs, dim=1)
        return xi
    
    def to_cpu(self):
        pass

class Identity:
    def __init__(self):
        self.name = "Identity"
        self.description = f""
        self.order = 1

    def transform(self, x):
        return x
    
    def to_cpu(self):
        pass