import os
import psutil
import pathlib
import random
import numpy as np
import pickle
import cv2
import collections
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from dqn.dqn_egreedy import EpsilonGreedyDQN
from dqn.dqn_noisynets import NoisyNetsDQN

from run_statistics import Statistics

cv2.ocl.setUseOpenCL(False)
os.environ['TF_CPP_MIN_LONG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({'font.size': 14})

import gym_video_recorder

from behavioural_cloning.bc_base import BehaviouralCloning
from dqn.dqn_twin import DQNTwin

from constants.general import *
checkpoints_dir = "/home/wyq/uncertainty/deep-rl-action-advising/Runs/Checkpoints"

def eval(config, eval_env):
    env_info = ENV_INFO[config['env_key']]
    max_timesteps = env_info[8]
    config['env_type'] = env_info[1]
    config['env_obs_form'] = env_info[2]
    config['env_states_are_countable'] = env_info[3]
    config['rm_extra_content'] = ['source', 'state_id', 'state_id_next', 'expert_action', 'preserve']


    if config['env_type'] == ALE:
        config['env_obs_dims'] = eval_env.observation_space.shape
        config['env_n_actions'] = eval_env.action_space.n
        config['env_obs_dims'] = (84, 84, 4)  # If LazyFrames are enabled

    elif config['env_type'] == BOX2D:
        config['env_obs_dims'] = eval_env.observation_space.shape
        config['env_n_actions'] = eval_env.action_space.n

    elif config['env_type'] == GRIDWORLD:
        config['env_obs_dims'] = eval_env.obs_space.shape
        config['env_n_actions'] = eval_env.action_space.n

    elif config['env_type'] == MAPE:
        config['env_obs_dims'] = eval_env.observation_space[0].shape
        config['env_n_actions'] = eval_env.action_space[0].n

    elif config['env_type'] == MINATAR:
        config['env_obs_dims'] = eval_env.state_shape()
        config['env_n_actions'] = eval_env.num_actions()
    if config['use_gpu']:
            print('Using GPU.')
            session_config = tf.compat.v1.ConfigProto(
                # intra_op_parallelism_threads=1,
                # inter_op_parallelism_threads=1
            )
            session_config.gpu_options.allow_growth = True
    else:
        print('Using CPU.')
        session_config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            device_count={'CPU': 1, 'GPU': 0})

    session = tf.compat.v1.InteractiveSession(graph=tf.compat.v1.get_default_graph(), config=session_config)

    student_agent = EpsilonGreedyDQN(config['load_student'].split('/')[-2], config, session,
                                                  config['dqn_eps_start'],
                                                  config['dqn_eps_final'],
                                                  config['dqn_eps_steps'], stats=None,
                                                  demonstrations_datasets=None, n_heads=config['n_heads'])
    student_agent.restore(checkpoints_dir, config['load_student'], 5e6)
    eval_total_reward_real = 0.0
    eval_total_reward = 0.0
    eval_duration = 0

    eval_advices_reused = 0
    eval_advices_reused_correct = 0

    if config['env_type'] == ALE:
        eval_env.seed(config['env_evaluation_seed'])
    elif config['env_type'] == BOX2D:
        eval_env.seed(config['env_evaluation_seed'])
    elif config['env_type'] == GRIDWORLD:
        eval_env.set_random_state(config['env_evaluation_seed'])
    elif config['env_type'] == MAPE:
        eval_env.set_world_random_state(config['env_evaluation_seed'])
    elif config['env_type'] == MINATAR:
        eval_env.set_random_state(config['env_evaluation_seed'])

    for i_eval_trial in range(config['n_evaluation_trials']):
        eval_obs_images = []

        eval_obs = None

        if config['env_type'] == ALE:
            eval_obs = eval_env.reset()
        elif config['env_type'] == BOX2D:
            eval_obs = eval_env.reset()
        elif config['env_type'] == GRIDWORLD:
            eval_obs = eval_env.reset()
        elif config['env_type'] == MAPE:
            eval_obs = eval_env.reset()[0]
        elif config['env_type'] == MINATAR:
            eval_env.reset()
            eval_obs = eval_env.state().astype(dtype=np.float32)

        eval_state_id = eval_env.get_state_id() if config['env_type'] == GRIDWORLD else None

        eval_episode_reward_real = 0.0
        eval_episode_reward = 0.0
        eval_episode_duration = 0

        while True:

            eval_action = None
            eval_teacher_action = None


            eval_action = student_agent.get_greedy_action(eval_obs)
            # print(f"action is {eval_action}")

            eval_obs_next, eval_reward, eval_done = None, None, None

            if config['env_type'] == ALE:
                eval_obs_next, eval_reward, eval_done, eval_info, eval_real_reward \
                    = eval_env.step(eval_action)

            elif config['env_type'] == BOX2D:
                eval_obs_next, eval_reward, eval_done, eval_info = eval_env.step(eval_action)
                eval_real_reward = eval_reward

            elif config['env_type'] == GRIDWORLD:
                eval_obs_next, eval_reward, eval_done = eval_env.step(eval_action)
                eval_real_reward = eval_reward

            elif config['env_type'] == MAPE:
                eval_obs_next_n, eval_reward_n, eval_done_n, eval_info_n = eval_env.step([eval_action])
                eval_obs_next, eval_reward, eval_done = eval_obs_next_n[0], eval_reward_n[0], eval_done_n[0]
                eval_real_reward = eval_info_n['n'][0]

            elif config['env_type'] == MINATAR:
                eval_reward, eval_done = eval_env.act(eval_action)
                eval_obs_next = eval_env.state().astype(dtype=np.float32)
                eval_real_reward = eval_reward

            eval_episode_reward_real += eval_real_reward
            eval_episode_reward += eval_reward

            eval_duration += 1
            eval_episode_duration += 1
            eval_obs = eval_obs_next

            eval_state_id = eval_env.get_state_id() if config['env_type'] == GRIDWORLD else None

            eval_done = eval_done or eval_episode_duration >= max_timesteps

            if eval_done:

                eval_total_reward += eval_episode_reward
                eval_total_reward_real += eval_episode_reward_real
                break

    eval_mean_reward = eval_total_reward / float(config['n_evaluation_trials'])
    eval_mean_reward_real = eval_total_reward_real / float(config['n_evaluation_trials'])
    print('Evaluation @  {} & {}'.format(eval_mean_reward, eval_mean_reward_real))
    return eval_mean_reward, eval_mean_reward_real