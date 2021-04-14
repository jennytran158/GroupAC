import numpy as np
import time
import copy
import torch

def sample_trajectory(env, policies, max_path_length):
    step_ob = env.reset()
    steps = 0
    n_agents = env.num_agents()
    acs = []
    obs = []
    next_obs = []
    acs_mask = []
    rewards = []
    terminals = []
    while True:
        ac = []
        ob = []
        ac_m = []
        for i in range(n_agents):  #per ob[i] = group[i] observation
            ac_m_i = step_ob[i]['action_mask']
            ob_i = step_ob[i]['obs']
            a_i = policies[i].get_action([ob_i],[ac_m_i],steps)
            ac.extend(a_i)
            ob.extend(ob_i)
            ac_m.extend(ac_m_i)
        obs.append(ob)
        acs.append(ac)
        acs_mask.append(ac_m)
        step_ob, rew, done, _ = env.step(ac)
        # save the observations after taking a step to next_obs
        next_ob = []
        for i in range(n_agents):
            next_ob.extend(step_ob[i]['obs'])
        next_obs.append(next_ob)
        steps += 1
        terminals.append(done[0])
        rewards.append(rew[0])
        if done["__all__"] or steps >= max_path_length:
            break
    return Path(obs, acs,acs_mask, rewards, next_obs, terminals,steps)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length):
    timesteps_this_batch = 0
    paths = []
    max_length = 0
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length)
        paths.append(path)
        #count steps
        timesteps_this_batch += get_pathlength(path)
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policies, ntraj, max_path_length):
    """
        Collect ntraj rollouts using policy
    """
    paths = []
    for i in range(ntraj):
        path = sample_trajectory(env, policies, max_path_length)
        paths.append(path)
    return paths

############################################
############################################

def Path(obs, acs,acs_mask, rewards, next_obs, terminals,length):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    p = {}
    p["observation"] = torch.tensor(obs,dtype= torch.float)
    p["action"] = torch.tensor(acs,dtype= torch.long)
    p["action_mask"] = torch.tensor(acs_mask,dtype= torch.float)
    p["next_observation"] = torch.tensor(next_obs,dtype= torch.float)
    p["reward"] = torch.tensor(rewards,dtype= torch.float)
    p["terminal"] = torch.tensor(terminals,dtype= torch.float)
    p['length'] = length
    return p


def convert_listofrollouts(paths,n_agents,normalize_reward=False):
    """
        Take a list of rollout dictionaries
        and return a tuple of tensors,
        where each element is a concatenation of that element from across the rollouts
    """
    paths.sort(key=lambda p: p['length'],reverse=True)
    observations = torch.cat([path["observation"] for path in paths])
    actions = torch.cat([path["action"] for path in paths])
    actions_mask = torch.cat([path["action_mask"] for path in paths])
    next_observations = torch.cat([path["next_observation"] for path in paths])
    terminals = torch.cat([path["terminal"] for path in paths])
    rewards = []
    for p_i in range(len(paths)):
        if normalize_reward:
             paths[p_i]["reward"] = (paths[p_i]["reward"] - torch.mean(paths[p_i]["reward"])) / (torch.std(paths[p_i]["reward"]) + 1e-8)
        rewards.append(paths[p_i]["reward"])
    rewards = torch.cat(rewards)
    lengths = [path["length"] for path in paths]
    return observations, actions,actions_mask, next_observations, terminals, rewards, lengths

############################################
############################################

def get_pathlength(path):
    return path['length']

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data
