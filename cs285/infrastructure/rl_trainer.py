from collections import OrderedDict
import time
from cs285.agents.gac_agent import GACAgent
import torch
from cs285.infrastructure import pytorch_util as ptu
from cs285.critics.boostrapped_sum_critic_mixer import BootstrappedSumCriticMixer
from gym.spaces import Discrete, Box, Dict
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
import random
class GAC_Trainer(object):

    def __init__(self, params):
        self.params = params
        self.params['agent_params']['gamma'] = self.params['gamma']
        # Set random seeds
        seed = self.params['seed']
        torch.manual_seed(seed)
        # Setup GPU
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )
        self.logger = Logger(self.params['logdir'])
        #############
        ## ENV
        #############
        groups = self.params['groups']
        self.env_name = self.params['env']
        if self.env_name == 'StarCraft2Env':
            from smac.env import StarCraft2Env
            from cs285.infrastructure.wrappers import SC2Wrapper
            self.env = SC2Wrapper(StarCraft2Env(map_name=self.params['env_map'],seed = seed),\
            groups = groups)
        elif self.env_name == "Paticles":
            from multiagent.environment import MultiAgentEnv
            import multiagent.scenarios as scenarios
            from cs285.infrastructure.wrappers import ParticlesWrapper
            scenario = scenarios.load(scenario_name + ".py").Scenario()
            world = scenario.make_world()
            self.env = ParticlesWrapper(\
            MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation),\
            groups = groups)
        elif self.env_name == "Test":
            if self.params['random_init_state']:
                init_state = [random.randint(2, 9),random.randint(2, 9),random.randint(2, 9)]
            else:
                init_state = self.params['init_state']
            from cs285.infrastructure.wrappers import Test
            self.env = Test(groups,init_state=init_state,goal_state=self.params['goal_state'])

        #############
        ## AGENT
        #############
        self.agents =[]
        agent_critics = []
        for g_idx in range(len(groups)):
            ob_dim = len(self.env.observation_space[g_idx]['obs'])
            ac_dim = len(self.env.action_space[g_idx])
            avail_ac_dim = sum([ac.n for ac in self.env.action_space[g_idx]])
            self.params['agent_params']['n_agents'] = groups[g_idx]
            self.params['agent_params']['actor']['avail_ac_dim'] = avail_ac_dim
            self.params['agent_params']['actor']['ac_dim'] = ac_dim
            self.params['agent_params']['actor']['ob_dim'] = ob_dim
            self.params['agent_params']['critic']['ob_dim'] = ob_dim
            self.params['agent_params']['critic']['gamma'] = self.params['agent_params']['gamma']
            agent = GACAgent(self.env,self.params['agent_params'],groups[g_idx])
            self.agents.append(agent)
            agent_critics.append(agent.critic)
        self.centralized_mixer = BootstrappedSumCriticMixer(self.params['agent_params']['critic'],agent_critics)
    def run_training_loop(self):
        self.total_envsteps = 0
        self.start_time = time.time()
        n_iter = self.params['n_iter']
        use_batchsize = self.params['batch_size']
        policies = [a.actor for a in self.agents]
        n_agents = len(self.agents)
        for itr in range(n_iter):
            print("**********iteration: {}*************".format(itr))
            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False
            if self.params['chkpt_log_freq'] == -1:
                self.logmetrics = False
            if itr % self.params['chkpt_log_freq'] == 0:
                self.logchkpt = True
            else:
                self.logchkpt = False
            paths, envsteps_this_batch = utils.sample_trajectories(self.env,\
             policies, use_batchsize, self.params['ep_len'])
            self.total_envsteps += envsteps_this_batch
            # sort paths based on its length for rnn operations
            obs, acs,acs_mask, next_obs, terminals, rewards, lens = utils.convert_listofrollouts(paths,n_agents)
            all_logs = self.train_agent(obs,acs,acs_mask,rewards,next_obs,terminals,lens)
            if self.logmetrics:
                self.perform_logging(itr, paths, policies, all_logs)
            if self.logchkpt:
                for i in range(n_agents):
                    self.agents[i].save('{}/agent_{}_itr_{}.pt'.format(self.params['logdir'],i,itr))
        if self.env_name == 'StarCraft2Env':
            self.env.save_replay()

    def train_agent(self,obs,acs,acs_mask,rewards,next_obs,terminals,lens):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            critic_logs = self.centralized_mixer.update(obs,next_obs,rewards,terminals, lens)
            actor_logs= OrderedDict()
            n_agents= self.env.num_agents()
            start_ob_idx = torch.zeros(n_agents,dtype = torch.int)
            end_ob_idx = torch.zeros_like(start_ob_idx)
            start_ac_idx = torch.zeros_like(start_ob_idx)
            end_ac_idx = torch.zeros_like(start_ob_idx)
            start_ac_m_idx = torch.zeros_like(start_ob_idx)
            end_ac_m_idx = torch.zeros_like(start_ob_idx)
            for i in range(self.env.num_agents()):
                end_ob_idx[i] = self.agents[i].actor.ob_dim
                end_ac_idx[i] = self.agents[i].actor.ac_dim
                end_ac_m_idx[i] = self.agents[i].actor.avail_ac_dim
            end_ob_idx = torch.cumsum(end_ob_idx,dim=0)
            end_ac_idx = torch.cumsum(end_ac_idx,dim=0)
            end_ac_m_idx = torch.cumsum(end_ac_m_idx,dim=0)
            start_ob_idx[1:n_agents] = end_ob_idx[:n_agents-1]
            start_ac_idx[1:n_agents] = end_ac_idx[:n_agents-1]
            start_ac_m_idx[1:n_agents] = end_ac_m_idx[:n_agents-1]

            for i in range(self.env.num_agents()):
                log = self.agents[i].train(
                obs[:,start_ob_idx[i]:end_ob_idx[i]], acs[:,start_ac_idx[i]:end_ac_idx[i]],acs_mask[:,start_ac_m_idx[i]:end_ac_m_idx[i]],
                rewards, next_obs[:,start_ob_idx[i]:end_ob_idx[i]], terminals, lens)
                log_with_index = OrderedDict([(k+"_%d"%i,v) for k,v in log.items()])
                actor_logs.update(log_with_index)
            critic_logs.update(actor_logs)
            all_logs.append([critic_logs])
        return all_logs
    def perform_logging(self, itr, paths, eval_policy, all_logs):
        import statistics
        last_log = all_logs[-1]
        #######################
        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])
        # save eval metrics
        n_agents = self.env.num_agents()
        if self.logmetrics:
            logs = OrderedDict()
            train_returns = [path["reward"].sum().item() for path in paths]
            eval_returns = [eval_path["reward"].sum().item() for eval_path in eval_paths]
            # episode lengths, for logging
            train_ep_lens = [path['length'] for path in paths]
            eval_ep_lens = [eval_path["length"] for eval_path in eval_paths]
            # decide what to log
            logs["Eval_AverageReturn"] = statistics.mean(eval_returns)
            logs["Eval_StdReturn"] = statistics.stdev(eval_returns)
            logs["Eval_MaxReturn"] = max(eval_returns)
            logs["Eval_MinReturn"] = min(eval_returns)
            logs["Eval_AverageEpLen"] = statistics.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = statistics.mean(train_returns)
            logs["Train_StdReturn"] = statistics.stdev(train_returns)
            logs["Train_MaxReturn"] = max(train_returns)
            logs["Train_MinReturn"] = min(train_returns)
            logs["Train_AverageEpLen"] = statistics.mean(train_ep_lens)
            logs["Total_steps"] = self.total_envsteps
            logs["Time"] = time.time() - self.start_time
            for log in last_log:
                logs.update(log)

            # perform the logging
            for key, value in logs.items():
                if isinstance(value,dict):
                    self.logger.log_scalars(value,key,itr)
                else:
                    self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')
            self.logger.flush()
