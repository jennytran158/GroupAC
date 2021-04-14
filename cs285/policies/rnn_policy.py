import abc
import itertools
from torch import nn
from torch.nn.utils import rnn
from torch.nn import functional as F
from torch import optim
from collections import OrderedDict
import torch
from torch import distributions
import time
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
class RNNPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, params):
        super().__init__()
        self.n_agents = params['n_agents']
        self.ac_dim = params['ac_dim']
        self.avail_ac_dim = params['avail_ac_dim']
        self.ob_dim = params['ob_dim']
        self.encoder_n_layers = params['encoder_n_layers']
        self.decoder_n_layers = params['decoder_n_layers']
        self.size = params['layer_size_per_agent'] * self.n_agents
        self.learning_rate = params['learning_rate']
        self.ent_coef = params['entropy_coefficient']
        self.debugging = params['debugging']
        self.rnn = params['rnn']
        self.max_grad_norm = params['max_grad_norm']
        self.encoder = ptu.build_mlp(input_size=self.ob_dim,
                                     output_size=self.size,
                                     n_layers=self.encoder_n_layers,
                                     size=self.size,output_activation= 'relu')
        if self.rnn:
            self.lstm = nn.LSTM(self.size,self.size,batch_first = True)
            self.hc = None
        else:
            self.lstm = None
        self.decoder = ptu.build_mlp(input_size=self.size,
                                     output_size=self.avail_ac_dim,
                                     n_layers=self.decoder_n_layers,
                                     size=self.size,output_activation= 'identity')
        self.encoder.to(ptu.device)
        if self.rnn:
            self.lstm.to(ptu.device)
        self.decoder.to(ptu.device)
        if self.debugging:
            print(self.ac_dim,self.avail_ac_dim)
        self.print_time = False
        self.t = 0
        # for name, param in self.named_parameters():
        #     print( name, param.shape)
        self.optimizer = optim.Adam(self.parameters(),self.learning_rate)

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs,action_mask,t):
        """ obs: a list of variable-length trajectories
                (batch_size, observations in a trajectory, observation elements)
            action_mask: a list of action masks
                (batch_size, action masks in a trajectory, action mask elements)
         """
        if self.debugging:
            print('----get_action RNNPolicy----')
        self.t = t
        length = [len(obs)]
        obs = torch.tensor(obs,device = ptu.device).float()
        action_mask = torch.tensor(action_mask,device = ptu.device).float()
        action_distributions = self(obs,action_mask,length)
        action = action_distributions.sample() # don't bother with rsample
        # print("action prob: ",action_distributions.probs)
        if self.debugging:
            print("obs ",obs.shape,obs)
            print("action ",action.shape,action)
            print("action_mask ",action_mask.shape,action_mask)
        self.t = 0
        return action[0].tolist() #remove out of batch size 1, and first observation

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    def forward(self, observation,action_mask,trajectory_len):
        """ observation: a padded tensor of shape
                (batch_size, largest observation length, observation elements)
            action_mask: a padded tensor of shape
                (batch_size, largest action mask length, action mask elements)
         """
        if self.debugging:
            print('----forward RNNPolicy----')

        if self.debugging:
            print("observation: ",observation,observation.shape)
            print("action_mask: ",action_mask,action_mask.shape)
            print("trajectory_len: ",trajectory_len)
        with torch.cuda.device(ptu.device if ptu.device.type == 'cuda' else None):
            torch.cuda.synchronize()
            tm = time.time()
            enc_obs = self.encoder(observation)
            if self.print_time:
                torch.cuda.synchronize()
                print("enc_obs time",time.time()-tm)
            if self.debugging:
                print("enc logit",enc_obs.shape,enc_obs)
            if self.print_time:
                torch.cuda.synchronize()
                tm = time.time()
            if self.rnn:
                padded_enc_obs,scatter_idx = ptu.flat_to_paddedsequences(enc_obs,trajectory_len,0)
                if self.print_time:
                    torch.cuda.synchronize()
                    print("padded_enc_obs time",time.time()-tm)
                if self.debugging:
                    print("padded_enc_obs ",padded_enc_obs.shape,padded_enc_obs)
                if self.print_time:
                    torch.cuda.synchronize()
                    tm = time.time()
                # synchronization happens here due to .cpu()
                packed_enc_obs = rnn.pack_padded_sequence(padded_enc_obs,trajectory_len,batch_first = True)
                if self.print_time:
                    torch.cuda.synchronize()
                    print("packed_enc_obs time",time.time()-tm)
                if self.debugging:
                    print("packed_padded_enc_obs ",packed_enc_obs, packed_enc_obs)
                if self.print_time:
                    torch.cuda.synchronize()
                    tm = time.time()
                if self.t > 0:
                    lstm_out,(ht, ct) = self.lstm(packed_enc_obs,self.hc)
                else:
                    lstm_out,(ht, ct) = self.lstm(packed_enc_obs)
                self.hc = (ht, ct)
                if self.print_time:
                    torch.cuda.synchronize()
                    print("lstm_out time",time.time()-tm)
                if self.debugging:
                    print("lstm_out : ",lstm_out)
                    print("hidden: ",ht)
                if self.print_time:
                    torch.cuda.synchronize()
                    tm = time.time()
                unpacked_lstm_out, unpacked_lens =  rnn.pad_packed_sequence(lstm_out, batch_first=True)
                if self.print_time:
                    torch.cuda.synchronize()
                    print("unpacked_lstm_out time",time.time()-tm)
                if self.debugging:
                    print("unpacked_lstm_out ", unpacked_lstm_out.shape,unpacked_lstm_out)
                if self.print_time:
                    torch.cuda.synchronize()
                    tm = time.time()
                flat_lstm_out = ptu.paddedsequences_to_flat(unpacked_lstm_out,scatter_idx)
                if self.print_time:
                    torch.cuda.synchronize()
                    print("flat_lstm_out time",time.time()-tm)
                if self.debugging:
                    print("flat_lstm_out ", flat_lstm_out.shape,flat_lstm_out)
                if self.print_time:
                    torch.cuda.synchronize()
                    tm = time.time()
                dec = self.decoder(flat_lstm_out)
            else:
                dec = self.decoder(enc_obs)
            if self.print_time:
                torch.cuda.synchronize()
                print("dec time",time.time()-tm)
            if self.debugging:
                print("dec ", dec.shape,dec)
                print("action_mask: ",action_mask.shape,action_mask)
            masked_dec = dec.masked_fill(action_mask == 0, float('-inf'))
            if self.debugging:
                print("masked_dec ",masked_dec.shape,masked_dec)
            masked_dec = masked_dec.view(-1,self.ac_dim,self.avail_ac_dim//self.ac_dim)
            if self.debugging:
                print("dist masked_dec ",masked_dec.shape,masked_dec)
            return distributions.Categorical(logits=masked_dec)

    def save(self, filepath,iteration = None):
        saved_dict = self.get_dict(iteration)
        torch.save(saved_dict, filepath)

    def get_dict(self,iteration=None):
        saved_dict = {
            'actor_model': self.state_dict(),
            'actor_optimizer': self.optimizer.state_dict(),
            'iter': iteration,
            'ac_dim': self.ac_dim,
            'ob_dim': self.ob_dim,
            'actor_encoder_n_layers': self.encoder_n_layers,
            'actor_decoder_n_layers': self.decoder_n_layers,
            'size': self.size
            }
        if iteration:
            saved_dict['iter'] = iteration
        return saved_dict
    def load(self,filepath=None,iteration=False,state_dict=None):
        if filepath:
            chkpt = torch.load(filepath, map_location=lambda storage, loc: storage)
            self.load_state_dict(chkpt['actor_model'])
            self.optimizer.load_state_dict(chkpt['actor_optimizer'])
        if state_dict:
            self.load_state_dict(state_dict)

#####################################################
#####################################################
class RNNPolicyAC(RNNPolicy):
    def update(self, observations, actions,action_mask, adv_n, lengths):
        """ observations: a tensor of observations of shape (batch_size,obs)
            action_mask: a tensor of action_mask of shape (batch_size,action_mask)
            actions: a tensor of actions of shape (batch_size,action_mask)
            adv_n: a tensor of adv_n of shape (batch_size,adv_n)
            lengths: a tensor contains length of trajectories of shape (trajectories,1)
         """
        if self.debugging:
            print('----update RNNPolicy----')
        observations = observations.to(ptu.device)
        actions = actions.to(ptu.device)
        action_mask = action_mask.to(ptu.device)
        adv_n = adv_n.to(ptu.device)
        action_distributions = self.forward(observations,action_mask,lengths)
        # actions = actions.view(actions.shape[0],self.n_agents,-1)
        logp = action_distributions.log_prob(actions)
        adv_n = adv_n[:,None]
        if self.debugging:
            print('action_distributions ',action_distributions)
            print('action_distributions probs ',action_distributions.probs.shape,action_distributions.probs)
            print('action_distributions log probs ',torch.log(action_distributions.probs))
            print('actions ',actions.shape,actions)
            print('logp ',logp.shape,logp)
            print('adv_n ',adv_n.shape,adv_n)
            print('logp*adv_n', logp*adv_n)
        policy_loss = -(logp*adv_n).mean()
        entropy_mean = action_distributions.entropy().mean()
        entropy_loss = - entropy_mean
        loss = policy_loss + self.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        log = OrderedDict()
        # log['Actor_layer_grad_norm_before_clipped'] = {}
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         log['Actor_layer_grad_norm_before_clipped'][name] = param.grad.norm()
        nn.utils.clip_grad_norm_(self.parameters(),self.max_grad_norm)
        self.optimizer.step()
        log['Actor Policy Loss'] = policy_loss.item()
        log['Actor Entropy'] = entropy_mean.item()
        # log['Advantage Distribution'] = {}
        # log['Advantage Distribution']['min'] = adv_n.min().item()
        # log['Advantage Distribution']['max'] = adv_n.max().item()
        # log['Advantage Distribution']['mean'] = adv_n.mean().item()
        # log['Advantage Distribution']['std'] = adv_n.std().item()
        # log['Actor Layer Grad Norm'] = {}
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         log['Actor Layer Grad Norm'][name] = param.grad.norm()
        return log
