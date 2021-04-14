from .base_critic import BaseCritic
from torch import nn
from torch import optim
import torch

from cs285.infrastructure import pytorch_util as ptu


class SupervisedCriticGroup(nn.Module, BaseCritic):
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
    def __init__(self, hparams,preprocess_models=None,input_critics=[]):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        # self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']
        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.preprocess_models = preprocess_models
        self.shared_critics_gradient = hparams['shared_critics_gradient']
        self.t = 0
        self.updating_optimizer = True
        # if self.preprocess_models is not None:
        #     input_dim = 0
        #     for model in self.preprocess_models:
        #         input_dim += model.size[model.n_layers-1]
        #     self.critic_network = ptu.build_mlp(
        #         input_dim,
        #         1,
        #         n_layers=self.n_layers,
        #         size=self.size,
        #     )
        # else:
        output_size = self.size if not isinstance(self.size, list) else self.size[-1]

        self.critic_network_logits = ptu.build_mlp(input_size=self.ob_dim,
                                       output_size=output_size,
                                       n_layers=self.n_layers-1,
                                       size=self.size,output_activation= 'tanh')
        self.out = nn.Linear(output_size,1)
        # self.critic_network = ptu.build_mlp(
        #     self.ob_dim,
        #     1,
        #     n_layers=self.n_layers,
        #     size=self.size,
        # )
        self.critic_network_logits.to(ptu.device)
        self.out.to(ptu.device)
        self.input_critics = input_critics
        self.loss = nn.MSELoss()
        other_parameters = []
        for c in input_critics:
            other_parameters.extend(list(c.parameters()))
        self.optimizer = optim.Adam(
            list(self.critic_network_logits.parameters()) + list(self.out.parameters()),
            self.learning_rate,
        )
        self.critics_optimizer = optim.Adam(
            other_parameters,
            self.learning_rate,
        )

    def forward(self, obs):
        # print("critic obs")
        # print(obs)
        idx = 0
        critic_outputs = None
        for c in self.input_critics:
            out = c.forward(ptu.from_numpy(obs[idx]))[:,None]
            if critic_outputs is not None:
                critic_outputs = torch.cat([critic_outputs,out],dim =1)
            else:
                critic_outputs = out
            idx += 1
            # if idx == len(self.input_critics)-1:
            #     print("Supervised critic_outputs: ", critic_outputs)
        logits = self.critic_network_logits(critic_outputs)
        out = self.out(logits).squeeze(1)
        return out

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, v_target):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward
        # print("ob_no")
        # print(ob_no.shape)
        # print("ac_na")
        # print(ac_na.shape)
        # print("reward_n")
        # print(reward_n.shape)
        # print("terminal_n")
        # print(terminal_n.shape)
        # print("next_ob_no")
        # print(next_ob_no.shape)
        # loss = 0
        v_target = ptu.from_numpy(v_target)
        for _ in range(self.num_grad_steps_per_target_update):
            v = self.forward(ob_no)
            # print("Supervised v: ", v)
            # print("Supervised v_target: ", v_target)
            self.optimizer.zero_grad()
            self.critics_optimizer.zero_grad()
            loss = self.loss(v, v_target)
            loss.backward()
            # if self.shared_critics_gradient:
            self.optimizer.step()
            self.critics_optimizer.step()
            # elif self.updating_optimizer:
            #     self.optimizer.step()
            # else:
            #     self.critics_optimizer.zero_grad()
        return loss.item()
    def save(self, filepath,iteration = None):
        if iteration:
            torch.save({
                'critic_model': self.state_dict(),
                'critic_optimizer': self.optimizer.state_dict(),
                'iter': iteration,
                'critic_ob_dim': self.ob_dim,
                # 'critic_ac_dim': self.ac_dim,
                'critic_discrete': self.discrete,
                'critic_size': self.size,
                'critic_n_layers': self.n_layers
                }, filepath)
        else:
            torch.save({
                'critic_model': self.state_dict(),
                'critic_optimizer': self.optimizer.state_dict(),
                'critic_ob_dim': self.ob_dim,
                # 'critic_ac_dim': self.ac_dim,
                'critic_discrete': self.discrete,
                'critic_size': self.size,
                'critic_n_layers': self.n_layers
                }, filepath)
    def get_dict(self,iteration = None):
        if iteration:
            return {
                'critic_model': self.state_dict(),
                'critic_optimizer': self.optimizer.state_dict(),
                'iter': iteration,
                'critic_ob_dim': self.ob_dim,
                # 'critic_ac_dim': self.ac_dim,
                'critic_discrete': self.discrete,
                'critic_size': self.size,
                'critic_n_layers': self.n_layers
                }
        else:
            return {
                'critic_model': self.state_dict(),
                'critic_optimizer': self.optimizer.state_dict(),
                'critic_ob_dim': self.ob_dim,
                # 'critic_ac_dim': self.ac_dim,
                'critic_discrete': self.discrete,
                'critic_size': self.size,
                'critic_n_layers': self.n_layers
                }
    def load(self,filepath,iteration=False):
        chkpt = torch.load(filepath, map_location=lambda storage, loc: storage)
        # if iteration:
        self.load_state_dict(chkpt['critic_model'])
        self.optimizer.load_state_dict(chkpt['critic_optimizer'])
