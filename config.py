import torch
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def update_device(device_id):
    global device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")