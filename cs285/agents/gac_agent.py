from collections import OrderedDict

from cs285.critics.rnn_group_critic import RNNGroupCritic
from cs285.infrastructure.utils import *
from cs285.policies.rnn_policy import RNNPolicyAC
from .base_agent import BaseAgent
from cs285.infrastructure import pytorch_util as ptu
import torch

class GACAgent(BaseAgent):
    def __init__(self, env, agent_params,num_agents):

        super(GACAgent, self).__init__()
        self.env = env
        self.n_agents = num_agents
        self.gamma = agent_params['gamma']
        self.n_steps = agent_params['n_steps']
        self.standardize_advantages = agent_params['standardize_advantages']
        critic_params = agent_params['critic']
        actor_params = agent_params['actor']
        self.num_actor_updates_per_agent_update = agent_params['num_actor_updates_per_agent_update']
        agent_params['critic']['n_agents'] = self.n_agents
        agent_params['actor']['n_agents'] = self.n_agents
        self.debugging = agent_params['debugging']
        self.critic = RNNGroupCritic(agent_params['critic'])
        self.actor = RNNPolicyAC(agent_params['actor'])

    def save(self, path):
        save_dict = self.critic.get_dict()
        save_dict.update(self.actor.get_dict())
        torch.save(save_dict,path)
    def load(self, path):
        self.critic.load(path)
        self.actor.load(path)
    def train(self, ob_no, ac_na, ac_na_mask,re_n, next_ob_no, terminal_n, lens):
        """ ob_no: a packed sequence of observations - batch_size of packed sequence across time
            ac_na: a packed sequence of action masks
            ac_na_mask: a packed sequence of actions
            re_n: a packed sequence of rewards
            next_ob_no: a packed sequence of next observations
            terminal_n: a packed sequence of next terminals
            lens: a tensor of length of trajectories
         """
        ob_no = ptu.to_device(ob_no)
        ac_na = ptu.to_device(ac_na)
        ac_na_mask = ptu.to_device(ac_na_mask)
        re_n = ptu.to_device(re_n)
        next_ob_no = ptu.to_device(next_ob_no)
        # terminal_n = ptu.to_device(terminal_n)
        v_t = self.critic.forward(ob_no,lens)
        returns = ptu.estimate_n_step_return(self, v_t, re_n,lens,self.gamma,self.n_steps)
        advantage = self.estimate_advantage(returns,v_t)
        actor_log = None
        for _ in range(self.num_actor_updates_per_agent_update) :
            actor_log = self.actor.update(ob_no, ac_na,ac_na_mask, advantage,lens)
        return actor_log
    def estimate_advantage(self,returns,v_t):
        adv = returns - v_t
        if self.standardize_advantages:
            adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)
        return adv
