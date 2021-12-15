from ruamel.yaml import YAML, dump, RoundTripDumper
from RaisimGymVecEnvLstm import RaisimGymVecEnv as Environment
from PPO2 import PPO2
from raisim_gym.archi.policies import LstmPolicy, ActorCriticPolicy
from raisim_gym.helper.raisim_gym_helper import ConfigurationSaver, TensorboardLauncher
from _raisim_gym import RaisimGymEnv
import os
import math
import tensorflow as tf
from dump_params import dump_params

from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
import numpy as np


class CustomLSTMPolicy(LstmPolicy, ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=[32, 32], reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, layer_norm=False, feature_extraction="mlp",
                 **kwargs):
        # super(ActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
        #                                         scale=(feature_extraction == "cnn"))
        ActorCriticPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                   scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("input", reuse=True):
            self.masks_ph = tf.placeholder(tf.float32, [n_batch], name="masks_ph")  # mask (done t-1)
            # n_lstm * 2 dim because of the cell and hidden states of the LSTM
            # self.states_ph = tf.placeholder(tf.float32, [self.n_env, n_lstm * 2], name="states_ph")  # states
            # self.states_ph = [tf.placeholder(tf.float32, [self.n_env, n_lstm * 2],
            #                                  name="states_ph{}".format(i))
            #                   for i in range(len(n_lstm))]
            self.states_ph = tf.placeholder(tf.float32, [self.n_env, sum(n_lstm) * 2 * 2], name="states_ph")
            size_splits = [k * 2 for k in (n_lstm + n_lstm)]  # create split number
            hidden_cell_collection = tf.split(self.states_ph, size_splits, 1)
            # put value's hidden and cell and policy's hidden and cell into one states_ph

        with tf.variable_scope("model", reuse=reuse):
            latent_pi = tf.layers.flatten(self.processed_obs)
            latent_v = tf.layers.flatten(self.processed_obs)

            self.snew_pi = []
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            for idx, lstm_num in enumerate(n_lstm):
                input_sequence = batch_to_seq(latent_pi, self.n_env, n_steps)
                rnn_output, temp = lstm(input_sequence, masks, hidden_cell_collection[idx],
                                        'lstm_pi{}'.format(idx), n_hidden=lstm_num,
                                        layer_norm=layer_norm)
                self.snew_pi.append(temp)
                latent_pi = seq_to_batch(rnn_output)

            self.snew_v = []
            for idx, lstm_num in enumerate(n_lstm):
                input_sequence = batch_to_seq(latent_v, self.n_env, n_steps)
                rnn_output, temp = lstm(input_sequence, masks, hidden_cell_collection[idx + len(n_lstm)],
                                        'lstm_v{}'.format(idx), n_hidden=lstm_num,
                                        layer_norm=layer_norm)
                self.snew_v.append(temp)
                latent_v = seq_to_batch(rnn_output)

            self.value_fn = linear(latent_v, 'vf', 1)
            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(latent_pi, latent_v)

        self.snew = tf.concat([s for s in self.snew_pi] + [s for s in self.snew_v], axis=1)
        self.initial_state = np.zeros((self.n_env, sum(n_lstm) * 2 * 2), dtype=np.float32)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self._value, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
        else:
            return self.sess.run([self.action, self._value, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})


def run(yaml_file, rsc_path):
    # configuration
    cfg = YAML().load(open(yaml_file, 'r'))
    total_timesteps = cfg['environment']['total_timesteps']

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + '/../model'
    saver = ConfigurationSaver(log_dir=log_dir + '/black_panther',
                            save_items=[rsg_root + '/../src/Environment.hpp', yaml_file])


    # create environment from the configuration file
    env = Environment(RaisimGymEnv(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper)), cfg)
    # env.seed(1)

    # Get algorithm
    model = PPO2(
        tensorboard_log=saver.data_dir,
        policy=CustomLSTMPolicy,
        # policy_kwargs=dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])]),
        policy_kwargs=dict(n_lstm=[64, 64]),
        env=env,
        gamma=0.99,
        n_steps=math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt']),
        ent_coef=0,
        learning_rate=cfg['learning_rate'],
        vf_coef=0.5,
        max_grad_norm=0.5,
        lam=0.998,
        nminibatches=1,
        noptepochs=10,
        cliprange=0.2,
        verbose=1,
    )

    # tensorboard
    # Make sure that your chrome browser is already on.
    TensorboardLauncher(saver.data_dir + '/PPO2_1')

    # PPO run
    model.learn(total_timesteps=total_timesteps, eval_every_n=200, log_dir=saver.data_dir, record_video=cfg['record_video'])

    # Need this line if you want to keep tensorflow alive after training
    # input("Press Enter to exit... Tensorboard will be closed after exit\n")

    pkls = [os.path.join(saver.data_dir, x_) for x_ in filter(lambda x: 'pkl' in x, os.listdir(saver.data_dir))]
    list(map(dump_params, pkls))


if __name__ == "__main__":
    rsc_path = 'rsc'
    cfg_file = os.path.join('cfg', 'default_cfg.yaml')
    run(cfg_file, rsc_path)
    # run('manual.yaml', rsc_path)
