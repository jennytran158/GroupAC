import os
import time
import yaml
import sys
import copy
import collections
from infrastructure.rl_trainer import GAC_Trainer
def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        filepath = os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name))
        with open(filepath, "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
def main(params):

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['log_path'])

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    print(params)
    grp_str = "_".join([str(g) for g in params['groups']])
    actor_rnn = 'ar' if params['agent_params']['actor']['rnn'] else 'a'
    critic_rnn = 'cr' if params['agent_params']['critic']['rnn'] else 'c'
    logdir = actor_rnn+"_"+critic_rnn+"_"+params['env'] + '_' + params['env_map'] + '_' + grp_str+'_seed{}_'.format(params['seed']) + '_'+ 'ntu'.format(params['agent_params']['critic']['num_target_updates'])+ '_'+ 'ngsptu'.format(params['agent_params']['critic']['num_grad_steps_per_target_update'])+'_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = GAC_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    params = copy.deepcopy(sys.argv)
    defaul_config_path = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
    with open(defaul_config_path, "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    main(config_dict)
