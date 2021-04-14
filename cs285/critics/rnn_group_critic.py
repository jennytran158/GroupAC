from .base_critic import BaseCritic
from torch import nn
from torch import optim
import torch
from torch.nn.utils import rnn
from cs285.infrastructure import pytorch_util as ptu


class RNNGroupCritic(nn.Module, BaseCritic):
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
        self.lstm_size = hparams['lstm_layer_size_per_agent']* self.n_agents
        self.encoder_n_layers = hparams['encoder_n_layers']
        self.decoder_n_layers = hparams['decoder_n_layers']
        self.learning_rate = hparams['learning_rate']
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.debugging = hparams['debugging']
        self.rnn = hparams['rnn']
        if self.rnn:
            enc_output_size = self.lstm_size
            dec_input_size = self.lstm_size
        else:
            enc_output_size = self.size
            dec_input_size = self.size
        self.encoder = ptu.build_mlp(input_size=self.ob_dim,
                                     output_size=enc_output_size,
                                     n_layers=self.encoder_n_layers,
                                     size=self.size,activation="tanh",output_activation= 'tanh')
        self.lstm = nn.LSTM(self.lstm_size,self.lstm_size,batch_first = True)
        self.decoder = ptu.build_mlp(input_size=dec_input_size,
                                     output_size=1,
                                     n_layers=self.decoder_n_layers,
                                     size=self.size,activation="tanh",output_activation= 'identity')
        self.encoder.to(ptu.device)
        self.lstm.to(ptu.device)
        self.decoder.to(ptu.device)
    def forward(self, obs,trajectory_len):
        # print("obs.shape ",obs.shape," input dim ",self.ob_dim)
        # for name, param in self.named_parameters():
        #     print(name, param)
        # if self.debugging:
        #     print("******Forward group critic*****")
        #     print("obs ",obs.shape,obs)
        enc_obs = self.encoder(obs)
        # if self.debugging:
        #     print("enc logit",enc_obs.shape,enc_obs)
        if self.rnn:
            padded_enc_obs,scatter_idx = ptu.flat_to_paddedsequences(enc_obs,trajectory_len,0)
            # if self.debugging:
            #     print("padded_enc_obs ",padded_enc_obs.shape,padded_enc_obs)

            packed_enc_obs = rnn.pack_padded_sequence(padded_enc_obs,trajectory_len,batch_first = True)
            # if self.debugging:
            #     print("packed_padded_enc_obs ",packed_enc_obs, packed_enc_obs)

            lstm_out,(ht, ct) = self.lstm(packed_enc_obs)
            # if self.debugging:
            #     print("lstm_out : ",lstm_out)
            #     print("hidden: ",ht)
            unpacked_lstm_out, unpacked_lens =  rnn.pad_packed_sequence(lstm_out, batch_first=True)
            # if self.debugging:
            #     print("unpacked_lstm_out ", unpacked_lstm_out.shape,unpacked_lstm_out)

            flat_lstm_out = ptu.paddedsequences_to_flat(unpacked_lstm_out,scatter_idx)
            # if self.debugging:
            #     print("flat_lstm_out ", flat_lstm_out.shape,flat_lstm_out)
            dec = self.decoder(flat_lstm_out)
        else:
            dec = self.decoder(enc_obs)
        # if self.debugging:
        #     print("dec ", dec.shape,dec)
        dec = dec.squeeze(1)
        return dec

    def save(self, filepath,iteration = None):
        saved_dict = get_dict(iteration)
        torch.save(saved_dict, filepath)
    def get_dict(self,iteration = None):
        saved_dict =  {
            'critic_model': self.state_dict(),
            'critic_ob_dim': self.ob_dim,
            'critic_size': self.size,
            'critic_encoder_n_layers': self.encoder_n_layers,
            'critic_decoder_n_layers': self.decoder_n_layers
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
