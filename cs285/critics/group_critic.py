from .base_critic import BaseCritic
from torch import nn
from torch import optim
import torch

from cs285.infrastructure import pytorch_util as ptu


class GroupCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.n_agents = hparams['n_agents']
        self.ob_dim = hparams['ob_dim']
        self.size = hparams['layer_size_per_agent']* self.n_agents
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        output_size = self.size if not isinstance(self.size, list) else self.size[-1]
        self.critic_network_logits = ptu.build_mlp(input_size=self.ob_dim,
                                       output_size=1,
                                       n_layers=self.n_layers,
                                       size=self.size,activation= 'relu')
        self.critic_network_logits.to(ptu.device)

    def forward(self, obs):
        # print("obs.shape ",obs.shape," input dim ",self.ob_dim)
        # for name, param in self.named_parameters():
        #     print(name, param)
        out = self.critic_network_logits(obs).squeeze(1)
        return out

    def save(self, filepath,iteration = None):
        saved_dict = get_dict(iteration)
        torch.save(saved_dict, filepath)
    def get_dict(self,iteration = None):
        saved_dict =  {
            'critic_model': self.state_dict(),
            'critic_ob_dim': self.ob_dim,
            'critic_size': self.size,
            'critic_n_layers': self.n_layers
            }
        if iteration:
            saved_dict['iter'] = iteration
        return saved_dict
    def load(self,filepath=None,iteration=False,state_dict=None):
        if filepath:
            chkpt = torch.load(filepath, map_location=lambda storage, loc: storage)
            self.load_state_dict(chkpt['critic_model'])
        if state_dict:
            self.load_state_dict(state_dict)
