from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.critics.bootstrapped_continuous_critic_individual_sum import \
    BootstrappedContinuousCriticIndividualSum
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from cs285.policies.MLP_policy import PPO
from .base_agent import BaseAgent
from cs285.infrastructure import pytorch_util as ptu
import torch

class ACAgent(BaseAgent):
    def __init__(self, env, agent_params,preprocess_models_path=None):

        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.actor = PPO(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate']
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)
        self.replay_buffer = ReplayBuffer()

    def save(self, path):
        save_dict = self.critic.get_dict()
        save_dict.update(self.actor.get_dict())
        torch.save(save_dict,path)
    def load(self, path=None,state_dict=None):
        if path:
            self.critic.load(filepath=path)
            self.actor.load(filepath=path)
        if state_dict:
            self.critic.load(state_dict=state_dict['critic_model'])
            self.actor.load(state_dict=state_dict['actor_model'])
    def train(self, ob_no, ac_na, ac_na_mask,re_n, next_ob_no, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).type(torch.int64)
        ac_na_mask = ptu.from_numpy(ac_na_mask).type(torch.int64)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)
        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']) :
          actor_log = self.actor.update(ob_no, ac_na,ac_na_mask, advantage)
        return actor_log

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        v_t = self.critic.forward(ob_no)
        v_tp1 = self.critic.forward(next_ob_no)
        q_t = re_n + self.gamma*v_tp1 * (1-terminal_n)
        adv_n = ptu.to_numpy(q_t - v_t)
        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return ptu.from_numpy(adv_n)

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
