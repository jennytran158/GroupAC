from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for i in range(n_layers):
        if isinstance(size,list):
            layers.append(nn.Linear(in_size, size[i]))
            in_size = size[i]
        else:
            layers.append(nn.Linear(in_size, size))
            in_size = size
        layers.append(activation)
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)
device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(model.named_parameters())" to visualize the gradient flow'''
    import matplotlib.pyplot as plt
    import numpy as np
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            p = p.clone()
            p.cpu()
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)
def from_list(*args, **kwargs):
    return torch.as_tensor(*args, **kwargs).float().to(device)
def to_list(*args, **kwargs):
    return tensor.to('cpu').detach().tolist()
def to_device(t):
    return t.to(device)
def flat_to_paddedsequences(flat,lengths,pad):
    """
        flat (tensor): tensor of shape (total # of obs , ob_dim), sorted based on length of its corresponding rollout
        lengths (list int): length of rollouts
        pad (scalar int): pad value
    """
    # print("lengths",lengths)
    # with torch.cuda.device(ptu.device if ptu.device.type == 'cuda' else None):
    local_ids = torch.cat([torch.arange(l) for l in lengths]).to(device)
    # print("local_ids",local_ids)
    max_length = lengths[0]
    start_idx = torch.arange(len(lengths),device = device) * max_length
    # print("start_idx",start_idx)
    t_lengths = torch.tensor(lengths).to(device)
    start_idx = torch.repeat_interleave(start_idx,t_lengths.to(device))
    scatter_idx = local_ids + start_idx
    # print("scatter_idx",scatter_idx)
    padded_seq = torch.full((len(lengths)*max_length,flat.shape[-1]),0).float().to(device)
    # flat = flat[None]
    # padded_seq = torch.full((lengths.shape[0],max_length,flat.shape[-1]),pad).float()
    # print("padded_seq new",padded_seq.shape,padded_seq)
    # print("flat",flat.shape,flat)
    padded_seq[scatter_idx] = flat
    # padded_seq = torch.scatter(padded_seq,1,scatter_idx,flat)
    # print("padded_seq scattered",padded_seq)
    padded_seq = padded_seq.view(len(lengths),max_length,flat.shape[-1])
    # print("padded_seq view",padded_seq)
    return padded_seq, scatter_idx
def paddedsequences_to_flat(padded_seq,scatter_idx):
    """
        flat (torch tensor): tensor of shape (total # of obs , ob_dim), sorted based on length of its corresponding rollout
        lengths (torch tensor): length of rollouts
        pad (torch tensor): pad value
    """
    padded_seq = padded_seq.flatten(end_dim = len(padded_seq.shape)-2)
    # print("padded_seq to flat",padded_seq.shape,padded_seq)
    # flat = torch.full((lengths.sum(),padded_seq.shape[-1])).float()
    flat = padded_seq[scatter_idx]
    return flat
def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
def estimate_n_step_return(self, v_t, re_n,lens,gamma,n_steps):
    returns = re_n.clone()
    #last ob always is a terminal state
    end_indices = torch.cumsum(torch.tensor(lens,device = device), dim = 0)
    cur_end_idx = end_indices.shape[0]-1
    cur_end = end_indices[cur_end_idx]
    mask = torch.zeros(returns.shape)
    for step in reversed(range(returns.shape[0])):
        if n_steps + step < cur_end:
            returns[step] += gamma*(returns[step+1] - re_n[step+n_steps]*(gamma**(n_steps-1)))
            mask[step] = 1
        else:
            returns[step] += (gamma*returns[step+1]) if step+1 < cur_end else 0
        if step == end_indices[cur_end_idx-1]:
            cur_end_idx -= 1
            cur_end = end_indices[cur_end_idx]
    returns[mask == 1] += v_t[torch.roll(mask==1,n_steps,0)]*(gamma**n_steps)
    return returns
