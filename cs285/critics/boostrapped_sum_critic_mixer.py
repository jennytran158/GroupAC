from .base_critic import BaseCritic
from torch import nn
from torch import optim
import torch
from collections import OrderedDict
import numpy as np
from cs285.infrastructure import pytorch_util as ptu


class BootstrappedSumCriticMixer(nn.Module, BaseCritic):
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
    def __init__(self, hparams,input_critics=[]):
        super().__init__()
        self.learning_rate = hparams['learning_rate']
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.n_steps = hparams['n_steps']
        self.debugging = hparams['debugging']
        self.max_grad_norm = hparams['max_grad_norm']
        # for c in input_critics:
        #     critics_parameters.extend(list(c.critic_network_logits.parameters()))
        self.input_critics = nn.ModuleList(input_critics)
        self.n_agents = len(self.input_critics)
        self.start_ob_idx = torch.zeros(self.n_agents,dtype = torch.int)
        self.end_ob_idx = torch.zeros_like(self.start_ob_idx)
        for i, c in enumerate(self.input_critics):
            self.end_ob_idx[i] = c.ob_dim
        self.end_ob_idx=torch.cumsum(self.end_ob_idx,dim=0)
        self.start_ob_idx[1:self.n_agents] = self.end_ob_idx[:self.n_agents-1]
        self.loss = nn.MSELoss()
        self.critics_optimizer = optim.Adam(
            self.parameters(),
            self.learning_rate,
        )
        self.input_critics.to(ptu.device)
    def forward(self, obs,lens):
        if self.debugging:
            print('----forward critic----')
        critic_outputs = torch.zeros(obs.shape[0],len(self.input_critics)).to(ptu.device)
        for i, c in enumerate(self.input_critics):
            critic_outputs[:,i] = c.forward(obs[:,self.start_ob_idx[i]:self.end_ob_idx[i]],lens)
        return critic_outputs.sum(dim=1)

    def update(self, ob_no, next_ob_no, reward_n, terminal_n, lens):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, agents, agent's ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, agents, agent's ob_dim). The observation after taking one step forward
                reward_n: shape: (sum_of_path_lengths,). Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: (sum_of_path_lengths,). Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        if self.debugging:
            print('----update critic----')

        ob_no = ob_no.to(ptu.device)
        reward_n = reward_n.to(ptu.device)
        next_ob_no = next_ob_no.to(ptu.device)
        terminal_n = terminal_n.to(ptu.device)
        log = OrderedDict()
        for j in range(self.num_target_updates):
            V_t = self.forward(ob_no,lens).detach()
            V_t_target = ptu.estimate_n_step_return(self, V_t, reward_n,lens,self.gamma,self.n_steps)
            for k in range(self.num_grad_steps_per_target_update):
                V_t = self.forward(ob_no,lens)
                self.critics_optimizer.zero_grad()
                loss = self.loss(V_t, V_t_target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(),self.max_grad_norm)
                self.critics_optimizer.step()
                log['Centralized Critic Value Loss'] = loss.item()
                log['Centralized Critic Residual Variance'] = (V_t_target - V_t).var()/V_t_target.var()
                log['Centralized Critic Mean Value And Reward'] = {'Value': V_t.mean(),'Reward':reward_n.mean()}
                # log['Centralized Critic Layer Grad Norm'] = {}
                # for name, param in self.named_parameters():
                #     if param.grad is not None:
                #         log['Centralized Critic Layer Grad Norm'][name] = param.grad.norm()
        return log
