from ruamel.yaml import YAML, dump, RoundTripDumper
from RaisimGymVecEnvLstm import RaisimGymVecEnv as Environment
from PPO2 import PPO2
from raisim_gym.archi.policies import MlpPolicy
from raisim_gym.helper.raisim_gym_helper import ConfigurationSaver, TensorboardLauncher
from _raisim_gym import RaisimGymEnv
import os
import math
import tensorflow as tf
from dump_params import dump_params

from commanded_locomotion_lstm import CustomLSTMPolicy


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
    model = PPO2.load(log_dir + '/black_panther/' + cfg['pretrained_model'])
    model.env = env
    model.tensorboard_log = saver.data_dir
    model.learning_rate = 1e-4

    # tensorboard
    # Make sure that your chrome browser is already on.
    TensorboardLauncher(saver.data_dir + '/PPO2_1')

    # PPO run
    model.learn(total_timesteps=total_timesteps, eval_every_n=50, log_dir=saver.data_dir, record_video=cfg['record_video'])

    # Need this line if you want to keep tensorflow alive after training
    # input("Press Enter to exit... Tensorboard will be closed after exit\n")

    pkls = [os.path.join(saver.data_dir, x_) for x_ in filter(lambda x: 'pkl' in x, os.listdir(saver.data_dir))]
    list(map(dump_params, pkls))


if __name__ == "__main__":
    rsc_path = 'rsc'
    cfg_file = os.path.join('cfg', 'retrain.yaml')
    run(cfg_file, rsc_path)
