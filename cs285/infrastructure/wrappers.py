from gym import Wrapper
from gym import Env
from gym.spaces import Discrete, Box, Dict
import copy
import numpy as np
import random
class SC2Wrapper(Wrapper):
    def __init__(self,env,groups):
        self._env = env
        n_agents = len(groups)
        self.observation_space = [None]*n_agents
        self.action_space = [None]*n_agents
        self.groups = groups
        for g_idx in range(len(groups)):
            obs_size = groups[g_idx]*self._env.get_obs_size()
            self.action_space[g_idx] = [Discrete(self._env.get_total_actions())] * groups[g_idx]
            action_size = self._env.get_total_actions()* groups[g_idx]
            self.observation_space[g_idx] = {
                "action_mask":Box(0, 1, shape=(action_size,)),
                "obs": [Box(-1, 1, shape=(1,))]*obs_size
            }

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
        obs (dict): New observations for each ready agent.
        """
        obs_list, state_list = self._env.reset()
        return_obs = {}
        cur = 0
        for g_idx in range(len(self.groups)):
            action_mask = []
            obs = []
            for j in range(self.groups[g_idx]):
                action_mask.extend(self._env.get_avail_agent_actions(cur))
                obs.extend(obs_list[cur])
                cur += 1
                return_obs[g_idx] = {
                "action_mask": action_mask,
                "obs": obs,
                }
        return return_obs

    def step(self, actions):
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
        obs (dict): New observations for each ready agent.
        rewards (dict): Reward values for each ready agent. If the
        episode is just started, the value will be None.
        dones (dict): Done values for each ready agent. The special key
        "__all__" (required) is used to indicate env termination.
        infos (dict): Optional info values for each agent id.
        """

        rew, done, info = self._env.step(actions)
        obs_list = self._env.get_obs()
        return_obs = {}
        cur = 0
        rews = {}
        infos = {}
        dones = {}
        for g_idx in range(len(self.groups)):
            action_mask = []
            obs = []
            for j in range(self.groups[g_idx]):
                action_mask.extend(self._env.get_avail_agent_actions(cur))
                obs.extend(obs_list[cur])
                cur += 1
            return_obs[g_idx] = {
            "action_mask": action_mask,
            "obs": obs,
            }
            rews[g_idx] = rew
            infos[g_idx] = info
            dones[g_idx] = done
            dones["__all__"] = done
        return return_obs, rews, dones, infos

    def close(self):
        """Close the environment"""
        self._env.close()
    def num_agents(self):
        return len(self.groups)
    class ParticleWrapper(Env):
    # load scenario from script
        def __init__(self, scenario_name='simple_spread',seed = 1,groups= [2],replay_dir=''):
            super(ParticleWrapper, self).__init__()
            scenario = scenarios.load(scenario_name + ".py").Scenario()
            world = scenario.make_world()
            self._env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
            self.groups = groups
            self.action_space = []
            self.observation_space = []
            idx = 0
            for g_idx in range(len(groups)):
                self.action_space.append([])
                self.action_space[g_idx] = []
                obs_dim = 0
                for _ in range(groups[g_idx]):
                    self.action_space[g_idx].append(self._env.action_space[idx])
                    obs_dim += len(self._env.observation_space[idx])
                    print("self._env.observation_space[idx]")
                    print(self._env.observation_space[idx])
                    idx+=1
                self.observation_space.append({
                    "action_mask": None,
                    "obs": Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
                })
                print("self.observation_space[g_idx]")
                print(self.observation_space[g_idx])
        def reset(self):
            obs_list = self._env.reset()
            return_obs = {}
            idx = 0
            for g_idx in range(len(self.groups)):
                obs = []
                for j in range(self.groups[g_idx]):
                    obs.extend(obs_list[idx])
                    idx+=1
                return_obs[g_idx] = {
                "action_mask": None,
                "obs":obs
                }
            return return_obs

        def step(self,actions):
            acs = []
            for g_idx in range(len(self.groups)):
                for i in range(self.groups[g_idx]):
                    acs.append(actions[g_idx][i])
            obs_list,rew,done,info = self._env.step(acs)
            return_obs = {}
            idx =  0
            rews={}
            infos = {}
            dones = {}
            for g_idx in range(len(self.groups)):
                obs = []
                for j in range(self.groups[g_idx]):
                    obs.extend(obs_list[idx])
                    rews[g_idx] = rew[idx]
                    dones[g_idx] = done[idx]
                    idx += 1
                return_obs[g_idx] = {
                    "action_mask": None,
                    "obs": obs,
                }
            dones["__all__"] = False
            return return_obs,rews,dones,infos
        def close(self):
            pass
        def num_agents(self):
            return len(self.groups)
class Test(Env):
    def __init__(self,groups= [3],init_state=[6,6,5],goal_state=[4,8,7]):
        super(Test, self).__init__()
        self.groups = groups
        n_agents = len(self.groups)
        self.init_state = init_state
        self.avail_ac_dim = 3
        self.ac_dim = 1
        self.init_ac_mask = [[1]*self.avail_ac_dim,[1]*self.avail_ac_dim,[1]*self.avail_ac_dim]
        self._agents_state = copy.deepcopy(self.init_state)
        self.observation_space = [None]*n_agents
        self.action_space = [None]*n_agents
        self._goal =  goal_state
        self.size = 12
        for g_idx in range(len(groups)):
            action_size = self.avail_ac_dim * groups[g_idx]
            self.action_space[g_idx] = [Discrete(self.avail_ac_dim)]*groups[g_idx]
            self.observation_space[g_idx] = {
                "action_mask":Box(0, 1, shape=(action_size,)),
                "obs": [Discrete(self.size)]*groups[g_idx],
            }

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        print("----------Reset------------")
        self._agents_state = copy.deepcopy(self.init_state)
        return_obs = {}
        cur = 0
        for g_idx in range(len(self.groups)):
            obs = []
            action_mask = []
            for j in range(self.groups[g_idx]):
                obs.append(self._agents_state[cur])
                action_mask.extend(self.init_ac_mask[cur])
                cur += 1
            return_obs[g_idx] = {
                "action_mask": action_mask,
                "obs": obs,
            }
        return return_obs

    def step(self, actions):
        """
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        acs = [a - 1 for a in actions]
        for i in range(len(acs)):
            self._agents_state[i] = self._agents_state[i] + acs[i]
        rew = sum([self._agents_state[i] == self._goal[i] for i in range(len(self._agents_state))])
        done = 11 in self._agents_state or 0 in self._agents_state
        return_obs = {}
        rews={}
        infos={}
        dones={}
        cur = 0
        for g_idx in range(len(self.groups)):
            action_mask = []
            obs = []
            for j in range(self.groups[g_idx]):
                if self._agents_state[cur] >= 10:
                    action_mask.extend([1,1,0])
                elif self._agents_state[cur] <= 1:
                    action_mask.extend([0,1,1])
                else:
                    action_mask.extend([1,1,1])
                obs.append(self._agents_state[cur])
                cur += 1
            return_obs[g_idx] = {
                "action_mask": action_mask,
                "obs": obs,
            }
            rews[g_idx]=rew
            infos[g_idx] =None
            dones[g_idx] =done
        print(acs)
        print(self._agents_state)
        print("step rews",rews)
        dones["__all__"] = done
        self.print_env()
        return return_obs, rews, dones, infos

    def close(self):
        """Close the environment"""
        pass
    def num_agents(self):
        return len(self.groups)
    def print_env(self):
        arr = [0] * 12
        for i in range(len(self._agents_state)):
            arr[self._agents_state[i]] = i+1
        print(arr)
