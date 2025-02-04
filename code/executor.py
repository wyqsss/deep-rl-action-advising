import os
from turtle import distance
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
from dqn.byol import BYOL_

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
import time


class Executor:
    def __init__(self, config, env, eval_env) -> None:
        self.config = config
        self.env = env
        self.eval_env = eval_env

        self.stats = None

        self.student_agent = None
        self.teacher_agent = None

        self.steps_reward = 0.0
        self.steps_reward_real = 0.0
        self.episode_duration = 0
        self.episode_reward = 0.0
        self.episode_reward_real = 0.0

        self.process = None
        self.run_id = None

        self.video_recorder = None

        self.save_videos_path = None
        self.scripts_dir = None
        self.local_workspace_dir = None

        self.runs_local_dir = None
        self.summaries_dir = None
        self.checkpoints_dir = None
        self.copy_scripts_dir = None
        self.videos_dir = None
        self.obs_images_dir = None
        self.vis_images_dir = None

        self.save_summary_path = None
        self.save_model_path = None
        self.save_scripts_path = None
        self.save_videos_path = None

        self.session = None
        self.summary_writer = None
        self.saver = None

        # RND observation normalization - UNUSED
        # self.obs_running_mean = None
        # self.obs_running_std = None
        # self.obs_norm_n = 0
        # self.obs_norm_max_n = None  # Determined later

        # Rendering
        self.obs_images = []

        # Action advising
        self.action_advising_budget = self.config['advice_collection_budget']
        self.reuse_enabled = False
        self.bc_model = None
        self.initial_imitation_is_performed = False
        self.steps_since_imitation = 0
        self.samples_since_imitation = 0

        self.advices_reused_ep = 0
        self.advices_reused_ep_correct = 0

        self.advice_reuse_probability_decrement = 0
        self.advice_reuse_probability_decay_steps = \
            self.config['advice_reuse_probability_decay_end'] - self.config['advice_reuse_probability_decay_begin']
        if self.config['advice_reuse_probability_decay']:
            self.advice_reuse_probability_decrement = \
                (self.config['advice_reuse_probability'] - self.config['advice_reuse_probability_final']) / \
                self.advice_reuse_probability_decay_steps

        self.advice_reuse_probability = self.config['advice_reuse_probability']

        self.dqn_twin = None

        self.student_model_uc_values_buffer = None

        self.evaluation_scores_windows = [collections.deque(maxlen=self.config['advice_reuse_stopping_eval_window_size']),
                                          collections.deque(maxlen=self.config['advice_reuse_stopping_eval_window_size'])]

        self.visualisation_values = []

        # Advice lookup table for AIR-Simple
        self.advice_lookup_table = {}

        self.byol = BYOL_()
        
        self.pol_average_distance = None

    # ==================================================================================================================

    def render(self, env):
        if self.config['env_type'] == ALE:
            pass
        elif self.config['env_type'] == BOX2D:
            pass
        elif self.config['env_type'] == GRIDWORLD:
            return env.render()
        elif self.config['env_type'] == MAPE:
            return env.render('rgb_array')[0]
        elif self.config['env_type'] == MINATAR:
            return env.render_state()


    # ==================================================================================================================

    def run(self):
        self.process = psutil.Process(os.getpid())

        os.environ['PYTHONHASHSEED'] = str(self.config['seed'])
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        tf.compat.v1.set_random_seed(self.config['seed'])
        tf.random.set_seed(self.config['seed'])

        self.run_id = self.config['run_id']
        self.seed_id = str(self.config['seed'])

        print('Run ID: {}'.format(self.run_id))

        # --------------------------------------------------------------------------------------------------------------

        # Folder structure can be modified here to reflect the user's preference
        self.scripts_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_workspace_dir = os.path.join(str(pathlib.Path(self.scripts_dir).parent))

        print('{} (Scripts directory)'.format(self.scripts_dir))
        print('{} (Local Workspace directory)'.format(self.local_workspace_dir))

        self.runs_local_dir = os.path.join(self.local_workspace_dir, 'Runs')
        os.makedirs(self.runs_local_dir, exist_ok=True)

        self.summaries_dir = os.path.join(self.runs_local_dir, 'Summaries')
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.runs_local_dir, 'Checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.copy_scripts_dir = os.path.join(self.runs_local_dir, 'Scripts')
        os.makedirs(self.copy_scripts_dir, exist_ok=True)

        self.videos_dir = os.path.join(self.runs_local_dir, 'Videos')
        os.makedirs(self.videos_dir, exist_ok=True)

        self.obs_images_dir = os.path.join(self.runs_local_dir, 'Observations')
        os.makedirs(self.obs_images_dir, exist_ok=True)

        self.replay_memory_dir = os.path.join(self.runs_local_dir, 'ReplayMemory')
        os.makedirs(self.replay_memory_dir, exist_ok=True)

        self.vis_images_dir = os.path.join(self.runs_local_dir, 'Visualisations')
        os.makedirs(self.vis_images_dir, exist_ok=True)

        self.save_summary_path = os.path.join(self.summaries_dir, self.run_id, self.seed_id)
        self.save_model_path = os.path.join(self.checkpoints_dir, self.run_id, self.seed_id)
        self.save_scripts_path = os.path.join(self.copy_scripts_dir, self.run_id, self.seed_id)
        self.save_videos_path = os.path.join(self.videos_dir, self.run_id, self.seed_id)
        self.save_obs_real_images_path = os.path.join(self.obs_images_dir, self.run_id, self.seed_id, 'Real')
        self.save_obs_agent_images_path = os.path.join(self.obs_images_dir, self.run_id, self.seed_id, 'Agent')
        self.save_replay_memory_path = os.path.join(self.replay_memory_dir, self.run_id, self.seed_id)
        self.save_vis_images_path = os.path.join(self.vis_images_dir, self.run_id, self.seed_id)

        if self.config['save_models']:
            os.makedirs(self.save_model_path, exist_ok=True)

        if self.config['dump_replay_memory']:
            os.makedirs(self.save_replay_memory_path, exist_ok=True)

        os.makedirs(self.save_videos_path, exist_ok=True)

        if self.config['save_obs_images']:
            os.makedirs(self.save_obs_real_images_path, exist_ok=True)
            os.makedirs(self.save_obs_agent_images_path, exist_ok=True)

        os.makedirs(self.save_vis_images_path, exist_ok=True)

        # --------------------------------------------------------------------------------------------------------------

        if self.config['use_gpu']:
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

        self.session = tf.compat.v1.InteractiveSession(graph=tf.compat.v1.get_default_graph(), config=session_config)

        self.summary_writer = tf.compat.v1.summary.FileWriter(self.save_summary_path, self.session.graph)

        # --------------------------------------------------------------------------------------------------------------

        self.env_info = {}

        env_info = ENV_INFO[self.config['env_key']]
        self.env_info['max_timesteps'] = env_info[8]

        self.config['env_type'] = env_info[1]
        self.config['env_obs_form'] = env_info[2]
        self.config['env_states_are_countable'] = env_info[3]

        if self.config['env_type'] == ALE:
            self.config['env_obs_dims'] = self.env.observation_space.shape
            self.config['env_n_actions'] = self.env.action_space.n
            self.config['env_obs_dims'] = (84, 84, 4)  # If LazyFrames are enabled

        elif self.config['env_type'] == BOX2D:
            self.config['env_obs_dims'] = self.env.observation_space.shape
            self.config['env_n_actions'] = self.env.action_space.n

        elif self.config['env_type'] == GRIDWORLD:
            self.config['env_obs_dims'] = self.env.obs_space.shape
            self.config['env_n_actions'] = self.env.action_space.n

        elif self.config['env_type'] == MAPE:
            self.config['env_obs_dims'] = self.env.observation_space[0].shape
            self.config['env_n_actions'] = self.env.action_space[0].n

        elif self.config['env_type'] == MINATAR:
            self.config['env_obs_dims'] = self.env.state_shape()
            self.config['env_n_actions'] = self.env.num_actions()

        print('Environment')
        print('Key (name):', self.config['env_key'])
        print('Observation shape:', self.config['env_obs_dims'])
        print('# of actions:', self.config['env_n_actions'])

        self.config['rm_extra_content'] = ['source', 'state_id', 'state_id_next', 'expert_action', 'preserve']

        # --------------------------------------------------------------------------------------------------------------

        n_states = self.env.n_states if self.config['env_type'] == GRIDWORLD else 0

        self.stats = Statistics(self.summary_writer, self.session, n_states)
        self.teacher_stats = Statistics(self.summary_writer, self.session, n_states)

        # --------------------------------------------------------------------------------------------------------------

        if self.config['env_type'] == GRIDWORLD:
            height = self.env.height
            width =  self.env.width
            n_action = self.config['env_n_actions']

            self.visualisation_values.append(np.zeros((height, width), dtype=np.float64))  # uncertainty
            # self.visualisation_values.append(np.zeros((height, width), dtype=np.int))  # state visitation all
            self.visualisation_values.append(np.zeros((height, width), dtype=np.int))  # self
            self.visualisation_values.append(np.zeros((height, width), dtype=np.int))  # teacher
            self.visualisation_values.append(np.zeros((height, width), dtype=np.int))  # reuse

            self.visualisation_values.append(np.zeros((height, width), dtype=np.float64))  # td-error

        # --------------------------------------------------------------------------------------------------------------
        # Setup student agent
        self.config['student_id'] = self.run_id

        # --------------------------------------------------------------------------------------------------------------
        # Load demonstrations dataset into memory
        demonstrations_datasets = []
        if self.config['load_demonstrations_dataset']:
            dataset_info = DEMONSTRATIONS_DATASET[self.config['env_key']][self.config['demonstrator_level']]
            path = os.path.join(self.replay_memory_dir, dataset_info[0], dataset_info[1], dataset_info[0] + '.rm')
            demonstrations_datasets.append(pickle.load(open(path, 'rb')))
            print('Loaded demonstrations dataset(s) from disk.')
        else:
            print('NOT loaded any demonstrations dataset(s).')

        # --------------------------------------------------------------------------------------------------------------
        # Initialise the student agent
        if self.config['dqn_type'] == 'egreedy':
            self.student_agent = EpsilonGreedyDQN(self.config['student_id'], self.config, self.session,
                                                  self.config['dqn_eps_start'],
                                                  self.config['dqn_eps_final'],
                                                  self.config['dqn_eps_steps'], self.stats,
                                                  demonstrations_datasets=demonstrations_datasets, n_heads=self.config['n_heads'])
        elif self.config['dqn_type'] == 'noisy':
            self.student_agent = NoisyNetsDQN(self.config['student_id'], self.config, self.session, self.stats,
                                              demonstrations_datasets=demonstrations_datasets)

        self.config['student_id'] = self.student_agent.id

        self.student_agent.discrete_bcq_filtering = self.config['use_bcq_loss']
        self.student_agent.advice_lookup_table = self.advice_lookup_table

        print('Student ID: {}'.format(self.student_agent.id))

        # --------------------------------------------------------------------------------------------------------------
        # # Setup the student agent's RND if needed
        # if 'novelty' in self.config['action_advising_method']:
        #     self.student_agent_rnd = RND(self.config['student_id'], self.config, self.session,
        #                                self.config['rnd_learning_rate'])
        #
        #     if self.config['action_advising_method'] == 'state_novelty':
        #         self.student_agent.rnd_model = self.student_agent_rnd

        # --------------------------------------------------------------------------------------------------------------
        # Save experiment configuration to a text file
        save_config(self.config, os.path.join(self.save_summary_path, 'config.txt'))

        # --------------------------------------------------------------------------------------------------------------
        # Initialise the teacher agent

        if self.config['load_teacher']:
            if self.config['env_type'] != GRIDWORLD:
                print('Teacher Key:', self.config['env_key'] + '-' + str(self.config['teacher_level']))
                teacher_info = TEACHER[self.config['env_key'] + '-' + str(self.config['teacher_level'])]
                self.config['teacher_id'] = teacher_info[0]
                self.config['teacher_model_structure'] = teacher_info[3]

                # Teacher is assumed to be generated with epsilon greedy model
                import copy
                teacher_config = copy.deepcopy(self.config)
                teacher_config['dqn_type'] = 'egreedy'
                teacher_config['dqn_dropout'] = False
                self.teacher_agent = EpsilonGreedyDQN(self.config['teacher_id'], teacher_config, self.session,
                                                                 eps_start=0.0, eps_final=0.0, eps_steps=1,
                                                                 stats=self.stats,
                                                                 demonstrations_datasets=(),
                                                      network_naming_structure_v1=self.config['teacher_model_structure'] == 0)

        # --------------------------------------------------------------------------------------------------------------
        # Initialise the behavioural cloning module (for teacher imitation)

        if self.config['advice_imitation_method'] != 'none':
            print('Initialising behaviour cloning network...')
            self.bc_model = BehaviouralCloning('BHC', self.config, self.session, None)

            if self.config['load_demonstrations_dataset']:
                for demonstrations_dataset in demonstrations_datasets:
                    for i in range(demonstrations_dataset.__len__()):
                        data = demonstrations_dataset._storage[i]
                        self.bc_model.feedback_observe(data[0], data[1])

                print('Loaded {} samples into the behavioural cloner.'.format(self.bc_model.replay_memory.__len__()))

        # --------------------------------------------------------------------------------------------------------------
        if self.config['dqn_twin']:
            self.dqn_twin = DQNTwin('DQN_TWIN', self.config, self.session, None)

        # --------------------------------------------------------------------------------------------------------------

        if self.config['use_proportional_student_model_uc_th']:
            self.student_model_uc_values_buffer = collections.deque(
                maxlen=self.config['proportional_student_model_uc_th_window_size'])

        # --------------------------------------------------------------------------------------------------------------
        # Print the number of neural network parameters in the experiment setup

        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Number of parameters: {}'.format(total_parameters))

        # --------------------------------------------------------------------------------------------------------------

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)
        self.session.run(tf.compat.v1.global_variables_initializer())

        # --------------------------------------------------------------------------------------------------------------
        # Restore the teacher model from a saved checkpoint
        if self.config['load_teacher']:
            if self.config['env_type'] != GRIDWORLD:
                print('Restoring the teacher model...')
                teacher_info = TEACHER[self.config['env_key'] + '-' + str(self.config['teacher_level'])]
                self.teacher_agent.restore(self.checkpoints_dir, teacher_info[0] + '/' + teacher_info[1], teacher_info[2])

        # --------------------------------------------------------------------------------------------------------------

        print('Finalising')
        if not self.config['save_models']:
            tf.compat.v1.get_default_graph().finalize()

        # --------------------------------------------------------------------------------------------------------------
        # Perform pre-training, as in Deep Q-Learning from Demonstrations
        if self.config['n_pretraining_iterations'] > 0:
            print('Performing {} pre-training steps...'.format(self.config['n_pretraining_iterations']))
            for _ in range(self.config['n_pretraining_iterations']):
                self.student_agent.feedback_learn(force_learn=True)

        # --------------------------------------------------------------------------------------------------------------

        eval_score, eval_score_real = self.evaluate()
        print('Evaluation @ {} | {} & {}'.format(self.stats.n_env_steps, eval_score, eval_score_real))

        obs, render = self.reset_env()
        state_id = self.env.get_state_id() if self.config['env_type'] == GRIDWORLD else None
        state_id_next = None

        print('state_id', state_id)

        if self.config['save_obs_images']:
            self.save_obs_image(obs, self.stats.n_env_steps)

        reward_is_seen = False  # For debugging
        # episode_time = 0
        while True:
            # loop_start = time.time()
            # ----------------------------------------------------------------------------------------------------------
            # # RND observation normalisation - UNUSED
            # if self.config['rnd_compute_coeffs']:
            #     if self.obs_norm_n < self.config['rnd_normalisation_steps']:
            #         obs_mean = obs.mean(axis=(0, 1))
            #         obs_std = obs.std(axis=(0, 1))
            #         if self.obs_norm_n == 0:
            #             self.obs_running_mean = obs_mean
            #             self.obs_running_std = obs_std
            #         else:
            #             self.obs_running_mean = \
            #                 self.obs_running_mean + (obs_mean - self.obs_running_mean) / (self.obs_norm_n + 1)
            #             self.obs_running_std = \
            #                 self.obs_running_std + (obs_std - self.obs_running_std) / (self.obs_norm_n + 1)
            #         self.obs_norm_n += 1
            #
            #     if self.obs_norm_n == self.config['rnd_normalisation_steps']:
            #         print(repr(self.obs_running_mean))
            #         print(repr(self.obs_running_std))
            #         self.obs_norm_n += 1
            #         import sys
            #         sys.exit()

            # ----------------------------------------------------------------------------------------------------------

            self_action = None
            teacher_action = None
            action = None
            action_source = 0  # 0: self, 1: teacher, 2: teacher imitation

            if self.config['execute_teacher_policy']:
                if teacher_action is None:
                    if self.config['env_type'] == GRIDWORLD:
                        teacher_action = self.env.optimal_action()
                    else:
                        teacher_action = self.teacher_agent.get_greedy_action(obs)

                action = teacher_action
                action_is_explorative = False
            else:
                self_action, action_is_explorative = self.student_agent.get_action(obs)
            # print(f"obs shape is {obs.shape}")
            if action_is_explorative:
                self.stats.exploration_steps_taken += 1
                self.stats.exploration_steps_taken_episode += 1
                self.stats.exploration_steps_taken_cum += 1

            # ----------------------------------------------------------------------------------------------------------
            # Advice Collection

            reuse_advice = False
            advice_collection_occurred = False

            if action is None and \
                    self.config['advice_collection_method'] != 'none' and \
                    (self.action_advising_budget > 0 or self.config['advice_collection_method'] == 'dual_uc'):

                if self.config['advice_collection_method'] == 'early':
                    advice_collection_occurred = True
                
                if self.config['advice_collection_method'] == 'late':
                    if self.episode_duration > 250:
                        advice_collection_occurred = True

                elif self.config['advice_collection_method'] == 'random':
                    if random.random() < 0.5:
                        advice_collection_occurred = True

                elif self.config['advice_collection_method'] == 'tabular_lookup':
                    if state_id not in self.advice_lookup_table:
                        advice_collection_occurred = True
                elif self.config['advice_collection_method'] == 'rcmp':
                    ucertainty, _, _ = self.student_agent.get_uncertainty_rcmp(obs)
                    # print(f"uncertainty is {ucertainty}")
                    # (1) Adaptive threshold mode
                    if self.config['use_proportional_student_model_uc_th']:
                        # Always collect advice until the uc values buffer reach a minimum size
                        if len(self.student_model_uc_values_buffer) < \
                                self.config['proportional_student_model_uc_th_window_size_min']:
                            advice_collection_occurred = True
                        else:
                            sorted_values = sorted(self.student_model_uc_values_buffer)
                            percentile_th = np.percentile(sorted_values,
                                                        self.config['proportional_student_model_uc_th_percentile'])

                            if ucertainty > percentile_th:
                                # self.config['proportional_student_model_uc_th_percentile'] += 30/24800
                                advice_collection_occurred = True

                        self.student_model_uc_values_buffer.append(ucertainty)

                    # (2) Constant threshold mode
                    else:
                        if ucertainty > self.config['student_model_uc_th']:
                            advice_collection_occurred = True

                # Based on the "uncertainty" estimated by twin network
                elif self.config['advice_collection_method'] == 'student_model_uc' or \
                        self.config['advice_collection_method'] == 'dual_uc':

                    # If student model hasn't started learning, then don't measure/record the uncertainty values
                    # They will all be high and meaningless to compare between
                    if self.student_agent.replay_memory.__len__() < self.config['dqn_rm_init']:
                        advice_collection_occurred = True
                    else:
                        uc_value, _, _ = self.dqn_twin.get_uncertainty(obs)

                        is_uncertain = False

                        # (1) Adaptive threshold mode
                        if self.config['use_proportional_student_model_uc_th']:
                            # Always collect advice until the uc values buffer reach a minimum size
                            if len(self.student_model_uc_values_buffer) < \
                                    self.config['proportional_student_model_uc_th_window_size_min']:
                                advice_collection_occurred = True
                            else:
                                sorted_values = sorted(self.student_model_uc_values_buffer)
                                percentile_th = np.percentile(sorted_values,
                                                          self.config['proportional_student_model_uc_th_percentile'])

                                if uc_value > percentile_th:
                                    is_uncertain = True

                            self.student_model_uc_values_buffer.append(uc_value)

                        # (2) Constant threshold mode
                        else:
                            if uc_value > self.config['student_model_uc_th']:
                                is_uncertain = True

                        if is_uncertain:
                            if self.config['advice_collection_method'] == 'dual_uc':
                                if self.initial_imitation_is_performed:
                                    bc_uncertainty = self.bc_model.get_uncertainty(obs)
                                    if bc_uncertainty < self.config['teacher_model_uc_th']:

                                        if random.random() < self.advice_reuse_probability:
                                            reuse_advice = True

                                    elif self.action_advising_budget > 0:
                                        advice_collection_occurred = True
                                elif self.action_advising_budget > 0:
                                    advice_collection_occurred = True
                            else:
                                advice_collection_occurred = True

                elif self.config['advice_collection_method'] == 'teacher_model_uc':
                    if self.initial_imitation_is_performed:
                        bc_uncertainty = self.bc_model.get_uncertainty(obs)
                        if bc_uncertainty > self.config['teacher_model_uc_th']:
                            advice_collection_occurred = True
                    else:
                        advice_collection_occurred = True
                
                elif self.config['advice_collection_method'] == 'sample_efficency':   # 依据样本的多样性，请求advice
                    if self.student_agent.replay_memory.__len__() <= self.config['dqn_rm_init'] + 1:
                        advice_collection_occurred = True
                    else:
                        distance = self.byol.cal(obs)

                        # (1) Adaptive threshold mode
                        if self.config['use_proportional_student_model_uc_th']:
                            # Always collect advice until the uc values buffer reach a minimum size
                            if len(self.student_model_uc_values_buffer) < \
                                    self.config['proportional_student_model_uc_th_window_size_min']:
                                advice_collection_occurred = True
                            else:
                                sorted_values = sorted(self.student_model_uc_values_buffer)
                                percentile_th = np.percentile(sorted_values,
                                                          self.config['proportional_student_model_uc_th_percentile'])
                                if distance > percentile_th:
                                    advice_collection_occurred = True
                            self.student_model_uc_values_buffer.append(distance)
                        # (2) Constant threshold mode
                        else:
                            if distance > self.pol_average_distance:
                                advice_collection_occurred = True


            if advice_collection_occurred:
                # print("use advice")
                if self.config['env_type'] == GRIDWORLD:
                    teacher_action = self.env.optimal_action()
                else:
                    teacher_action = self.teacher_agent.get_greedy_action(obs)

                action = teacher_action
                action_source = 1

                if self.config['mistake_correction_mode'] and action == self_action:
                    pass
                else:
                    self.action_advising_budget -= 1
                    self.stats.advices_taken += 1
                    self.stats.advices_taken_cum += 1

                    if self.config['advice_imitation_method'] != 'none':
                        self.bc_model.feedback_observe(obs, action)
                        self.samples_since_imitation += 1

            self.steps_since_imitation += 1

            # ----------------------------------------------------------------------------------------------------------

            # RND - UNUSED
            # if self.config['action_advising_method'] == 'advice_novelty' and action is not None:
            #     self.student_agent_rnd.train_model(obs, loss_id=0, is_batch=False, normalize=True)

            # ----------------------------------------------------------------------------------------------------------
            # Imitation
            if self.config['advice_imitation_method'] == 'tabular_lookup':
                if advice_collection_occurred:
                    self.advice_lookup_table[state_id] = teacher_action
                    self.initial_imitation_is_performed = True

            elif self.config['advice_imitation_method'] == 'periodic':

                if (self.steps_since_imitation >= self.config['advice_imitation_period_steps'] and
                    self.samples_since_imitation >= (self.config['advice_imitation_period_samples'] / 2)) or \
                        self.samples_since_imitation >= self.config['advice_imitation_period_samples']:

                    print(self.steps_since_imitation, self.samples_since_imitation)

                    if not self.initial_imitation_is_performed:
                        train_behavioural_cloner(self.bc_model,
                                                 self.config['advice_imitation_training_iterations_init'])

                        print('Self evaluating model...')
                        uc_threshold, accuracy = evaluate_behavioural_cloner(self.bc_model)

                        if self.config['autoset_teacher_model_uc_th']:
                            self.config['teacher_model_uc_th'] = uc_threshold

                        self.initial_imitation_is_performed = True
                        self.steps_since_imitation = 0
                        self.samples_since_imitation = 0
                    else:
                        if self.bc_model.replay_memory.__len__() == self.config['advice_collection_budget']:
                            train_behavioural_cloner(self.bc_model,
                                                     self.config['advice_imitation_training_iterations_init'])
                        else:
                            train_behavioural_cloner(self.bc_model,
                                                     self.config['advice_imitation_training_iterations_periodic'])

                        print('Self evaluating model...')
                        uc_threshold, accuracy = evaluate_behavioural_cloner(self.bc_model)

                        if self.config['autoset_teacher_model_uc_th']:
                            print('setting uc threshold:', uc_threshold)
                            self.config['teacher_model_uc_th'] = uc_threshold

                        self.steps_since_imitation = 0
                        self.samples_since_imitation = 0

            # ----------------------------------------------------------------------------------------------------------
            # Reuse
            # reuse_start = time.time()
            reuse_model_action = None

            if self.config['evaluate_advice_reuse_model']:
                if self.config['advice_reuse_method'] != 'none' and \
                        self.initial_imitation_is_performed:
                    reuse_model_action = np.argmax(self.bc_model.get_action_probs(obs))

                    if teacher_action is None:
                        if self.config['env_type'] == GRIDWORLD:
                            teacher_action = self.env.optimal_action()
                        else:
                            teacher_action = self.teacher_agent.get_greedy_action(obs)

                    self.stats.advice_reuse_model_n_evaluations += 1
                    self.stats.advice_reuse_model_n_evaluations_cum += 1

                    if reuse_model_action == teacher_action:
                        self.stats.advice_reuse_model_is_correct += 1
                        self.stats.advice_reuse_model_is_correct_cum += 1

            if not advice_collection_occurred and self.config['advice_collection_method'] != 'dual_uc':
                if self.config['advice_reuse_method'] != 'none' and self.initial_imitation_is_performed:

                    if self.config['advice_reuse_method'] == 'extended' or \
                            (self.config['advice_reuse_method'] == 'restricted' and action_is_explorative):

                        if self.reuse_enabled:
                            if self.config['advice_imitation_method'] == 'tabular_lookup':
                                if state_id in self.advice_lookup_table:
                                    reuse_advice = True

                            bc_uncertainty = self.bc_model.get_uncertainty(obs)
                            if bc_uncertainty < self.config['teacher_model_uc_th']:
                                reuse_advice = True

            if reuse_advice:
                if self.config['advice_imitation_method'] == 'tabular_lookup':
                    action = self.advice_lookup_table[state_id]
                else:
                    if reuse_model_action is None:
                        reuse_model_action = np.argmax(self.bc_model.get_action_probs(obs))

                    action = reuse_model_action

                action_source = 2

                self.stats.advices_reused += 1
                self.stats.advices_reused_cum += 1

                self.advices_reused_ep += 1
                self.stats.advices_reused_ep_cum += 1

                if teacher_action is None:
                    if self.config['env_type'] == GRIDWORLD:
                        teacher_action = self.env.optimal_action()
                    else:
                        teacher_action = self.teacher_agent.get_greedy_action(obs)

                if action == teacher_action:
                    self.stats.advices_reused_correct += 1
                    self.stats.advices_reused_correct_cum += 1
                    self.advices_reused_ep_correct += 1
                    self.stats.advices_reused_ep_correct_cum += 1
            # reuse_end = time.time()
            # print('reuse time: %s Seconds'%(reuse_end - reuse_start))
            # ----------------------------------------------------------------------------------------------------------

            if action is None:
                if self.config['utilise_imitated_model'] and self.initial_imitation_is_performed:
                    if self.config['advice_imitation_method'] == 'tabular_lookup':
                        if state_id in self.advice_lookup_table:
                            action = self.advice_lookup_table[state_id]
                    else:
                        bc_uncertainty = self.bc_model.get_uncertainty(obs)
                        if bc_uncertainty < self.config['teacher_model_uc_th']:
                            action = np.argmax(self.bc_model.get_action_probs(obs))

            if action is None:
                action = self_action

            # ----------------------------------------------------------------------------------------------------------

            # if self.action_advising_method == 'state_novelty':
            #    self.student_agent_rnd.train_model(obs, loss_id=0, is_batch=False, normalize=True)

            # ----------------------------------------------------------------------------------------------------------

            # Record visualisation data
            if self.config['env_type'] == GRIDWORLD:
                pos = ((self.env.height - 1) - self.env.state.agent_pos[0], self.env.state.agent_pos[1])

                # Total
                # self.visualisation_values[0][pos[0]][pos[1]] += 1

                if action_source == 0:  # Self
                    self.visualisation_values[1][pos[0]][pos[1]] += 1
                elif action_source == 1:  # Teacher
                    self.visualisation_values[2][pos[0]][pos[1]] += 1
                elif action_source == 2:  # Teacher Imitation
                    self.visualisation_values[3][pos[0]][pos[1]] += 1

            # ----------------------------------------------------------------------------------------------------------
            # Execute action

            obs_next, reward, reward_real, done = None, None, None, None

            if self.config['env_type'] == ALE:
                obs_next, reward, done, info, reward_real = self.env.step(action)

            elif self.config['env_type'] == BOX2D:
                obs_next, reward, done, info = self.env.step(action)
                reward_real = reward

            elif self.config['env_type'] == GRIDWORLD:
                obs_next, reward, done = self.env.step(action)
                reward_real = reward

            elif self.config['env_type'] == MAPE:
                obs_next_n, reward_n, done_n, info_n = self.env.step([action])
                obs_next, reward, done = obs_next_n[0], reward_n[0], done_n[0]
                reward_real = info_n['n'][0]

            elif self.config['env_type'] == MINATAR:
                reward, done = self.env.act(action)
                reward_real = reward
                obs_next = self.env.state()

            state_id_next = self.env.get_state_id() if self.config['env_type'] == GRIDWORLD else None

            # ----------------------------------------------------------------------------------------------------------
            # self.config['generate_extra_visualisations'] and \
            if self.config['env_type'] == GRIDWORLD and \
                    self.stats.n_env_steps % 200 == -1:

                # self.config['dqn_twin'] and
                if self.student_agent.replay_memory.__len__() >= self.config['dqn_rm_init']:
                    height = self.env.height

                    for n in range(len(self.env.passage_positions[0][0])):
                        y = self.env.passage_positions[0][0][n]
                        x = self.env.passage_positions[0][1][n]

                        y_inv = (height - 1) - y
                        obs_sample = self.env.generate_obs(0, (y, x))
                        obs_uc = self.get_student_uncertainty(obs_sample)*100.0

                        self.visualisation_values[0][y_inv][x] = obs_uc


                generate_grid_visualisation(self.env, self.config, self.save_vis_images_path,
                                            self.stats.n_env_steps,  self.visualisation_values)

            transition = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'obs_next': obs_next,
                'done': done,
                'source': action_source,
                'state_id': state_id,
                'state_id_next': state_id_next,
                'expert_action': teacher_action,
                'preserve': advice_collection_occurred if self.config['preserve_collected_advice'] else False
            }

            if render:
                if self.config['env_type'] == ALE:
                    self.video_recorder.capture_frame()
                elif self.config['env_type'] == BOX2D:
                    self.video_recorder.capture_frame()
                elif self.config['env_type'] == GRIDWORLD:
                    self.obs_images.append(self.render(self.env))
                elif self.config['env_type'] == MAPE:
                    self.obs_images.append(self.env.render('rgb_array')[0])
                elif self.config['env_type'] == MINATAR:
                    self.obs_images.append(self.render(self.env))

            self.episode_reward += reward
            self.episode_reward_real += reward_real
            self.episode_duration += 1

            self.steps_reward += reward
            self.steps_reward_real += reward_real
            self.stats.n_env_steps += 1

            if reward > 0 and reward_is_seen is False:
                reward_is_seen = True
                print(">>> Reward is seen at ", self.stats.n_episodes, "|", self.episode_duration)

            if self.config['save_obs_images']:
                self.save_obs_image(obs_next, self.stats.n_env_steps)

            # ----------------------------------------------------------------------------------------------------------
            # Feedback
            self.student_agent.feedback_observe(transition)

            # Update collection statistics (for Gridworld)
            if self.config['env_type'] == GRIDWORLD:
                if action_source == 0:
                    self.stats.n_s1_transition_collected[state_id] += 1
                    self.stats.n_s1_transition_collected_total += 1
                elif action_source == 1:
                    self.stats.n_s2_transition_collected[state_id] += 1
                    self.stats.n_s2_transition_collected_total += 1
                elif action_source == 2:
                    self.stats.n_s3_transition_collected[state_id] += 1
                    self.stats.n_s3_transition_collected_total += 1

            # ----------------------------------------------------------------------------------------------------------

            td_error_batch, loss, ql_loss, ql_loss_weighted, lm_loss, lm_loss_weighted, l2_loss, l2_loss_weighted, \
            feed_dict, is_batch = self.student_agent.feedback_learn(self.stats)

            # Train the twin DQN if the original DQN has performed a learning step
            loss_twin = 0.0
            if self.config['dqn_twin'] and feed_dict is not None:
                loss_twin = self.dqn_twin.train_model_with_feed_dict(feed_dict, is_batch)

            if self.config['advice_collection_method'] == 'sample_efficency' and self.student_agent.replay_memory.__len__() >= self.config['dqn_rm_init'] \
                and self.student_agent.replay_memory.__len__() % self.config['cons_learning_inter'] == 0 and self.action_advising_budget > 0:
                print("begin to train constractive model")
                self.pol_average_distance = self.byol.train(self.student_agent.replay_memory, self.config['cons_learning_epoch']) * self.config['gamma']


            # Measure uncertainty values and reflect changes in the TensorFlow summary (for Gridworld)
            if self.config['env_type'] == GRIDWORLD and \
                    self.student_agent.replay_memory.__len__() >= self.config['dqn_rm_init']:
                if self.stats.n_env_steps % 1 == 500:  # Period
                    uncertainty_values = []
                    for i in range(self.env.n_states):
                        sample_obs = self.env.generate_obs_from_state(state_id)
                        uncertainty_values.append(self.get_student_uncertainty(sample_obs))

                    self.stats.update_summary_extra(uncertainty_values)
            # ----------------------------------------------------------------------------------------------------------

            if self.config['advice_reuse_probability_decay'] and \
                    self.stats.n_env_steps > self.config['advice_reuse_probability_decay_begin'] and \
                    self.advice_reuse_probability > self.config['advice_reuse_probability_final']:
                self.advice_reuse_probability -= self.advice_reuse_probability_decrement
                if self.advice_reuse_probability < self.config['advice_reuse_probability_final']:
                    self.advice_reuse_probability = self.config['advice_reuse_probability_final']

            self.stats.loss += loss
            self.stats.loss_twin += loss_twin

            obs = obs_next
            state_id = state_id_next
            done = done or self.episode_duration >= self.env_info['max_timesteps']

            if done:
                self.stats.n_episodes += 1
                self.stats.episode_reward_auc += np.trapz([self.stats.episode_reward_last, self.episode_reward])
                self.stats.episode_reward_last = self.episode_reward

                self.stats.episode_reward_real_auc += np.trapz([self.stats.episode_reward_real_last, self.episode_reward_real])
                self.stats.episode_reward_real_last = self.episode_reward_real

                self.stats.reuse_enabled_in_ep_cum += (1 if self.reuse_enabled != 0 else 0)

                self.stats.update_summary_episode(self.episode_reward, self.stats.episode_reward_auc,
                                                  self.episode_duration,
                                                  self.advices_reused_ep, self.advices_reused_ep_correct,
                                                  1 if self.reuse_enabled else 0,
                                                  self.episode_reward_real, self.stats.episode_reward_real_auc,)

                print('n_episodes : {}'.format(self.stats.n_episodes), end=' | ')
                print('episode_reward : {:.1f}'.format(self.episode_reward), end=' | ')
                print('episode_reward_real : {:.1f}'.format(self.episode_reward_real), end=' | ')
                print('episode_duration : {}'.format(self.episode_duration), end=' | ')
                print('n_env_steps : {}'.format(self.stats.n_env_steps), end=' | ')
                print('left advice : {}'.format(self.action_advising_budget))

                if render:
                    if self.config['env_type'] == ALE:
                        self.video_recorder.close()
                        self.video_recorder.enabled = False

                    elif self.config['env_type'] == BOX2D:
                        self.video_recorder.close()
                        self.video_recorder.enabled = False

                    elif self.config['env_type'] == GRIDWORLD:
                        write_video(self.obs_images, self.save_videos_path, '{}_{}'.format(
                            str(self.stats.n_episodes - 1), str(self.stats.n_env_steps - self.episode_duration)))

                    elif self.config['env_type'] == MAPE:
                        write_video(self.obs_images, self.save_videos_path, '{}_{}'.format(
                            str(self.stats.n_episodes - 1), str(self.stats.n_env_steps - self.episode_duration)))

                    elif self.config['env_type'] == MINATAR:
                        write_video(self.obs_images, self.save_videos_path, '{}_{}'.format(
                            str(self.stats.n_episodes - 1), str(self.stats.n_env_steps - self.episode_duration)))
                # episode_time = 0
                # print(' episode_time ---------- : %s Seconds'%(episode_time))
                obs, render = self.reset_env()
                state_id = self.env.get_state_id() if self.config['env_type'] == GRIDWORLD else None
                state_id_next = None

            # Per N steps summary update
            if self.stats.n_env_steps % self.stats.n_steps_per_update == 0:
                self.stats.steps_reward_auc += np.trapz([self.stats.steps_reward_last, self.steps_reward])
                self.stats.steps_reward_last = self.steps_reward
                self.stats.epsilon = self.student_agent.eps if self.student_agent.type == 'egreedy' else 0

                self.stats.steps_reward_real_auc += np.trapz([self.stats.steps_reward_real_last, self.steps_reward_real])
                self.stats.steps_reward_real_last = self.steps_reward_real

                self.stats.update_summary_steps(self.steps_reward, self.stats.steps_reward_auc,
                                                self.steps_reward_real, self.stats.steps_reward_real_auc)

                self.stats.exploration_steps_taken = 0

                self.stats.advices_taken = 0
                self.stats.advices_used = 0
                self.stats.advices_reused = 0
                self.stats.advices_reused_correct = 0

                self.stats.advice_reuse_model_n_evaluations = 0
                self.stats.advice_reuse_model_is_correct = 0

                self.steps_reward = 0.0
                self.steps_reward_real = 0.0

            if self.stats.n_env_steps % self.config['evaluation_period'] == 0:
                eval_score_a, eval_score_real_a = self.evaluate(self.config['utilise_imitated_model'], False)
                print('Evaluation @ {} | {} & {}'.format(self.stats.n_env_steps, eval_score_a, eval_score_real_a))

                # Evaluate (B) with the teacher model enabled (if appropriate)
                if self.config['advice_imitation_method'] != 'none':
                    eval_score_b, eval_score_real_b = self.evaluate(True, True)
                    print('Evaluation (B) @ {} | {} & {}'.format(self.stats.n_env_steps, eval_score_b, eval_score_real_b))

                    if self.stats.n_evaluations_b >= self.config['advice_reuse_stopping_eval_start']:
                        self.evaluation_scores_windows[0].append(eval_score_real_a)
                        self.evaluation_scores_windows[1].append(eval_score_real_b)

                    if self.config['advice_reuse_stopping'] and \
                            self.config['advice_reuse_probability_final'] > 0:  # Compare Evaluation A and Evaluation B scores

                        if len(self.evaluation_scores_windows[0]) == self.config['advice_reuse_stopping_eval_window_size'] \
                                and len(self.evaluation_scores_windows[1]) == self.config['advice_reuse_stopping_eval_window_size']:
                            average_a = np.mean(self.evaluation_scores_windows[0])
                            average_b = np.mean(self.evaluation_scores_windows[1])

                            if average_b < 0:
                                target_score = float(average_b) * (2.0 - self.config['advice_reuse_stopping_eval_proximity'])
                            else:
                                target_score = float(average_b) * self.config['advice_reuse_stopping_eval_proximity']

                            print('Average A: {} | B: {} | Target: {}'.format(average_a, average_b, target_score))

                            if average_a >= target_score:
                                print('>>> Stopping advice reuse!')
                                self.advice_reuse_probability = 0.0
                                self.config['advice_reuse_probability_final'] = 0.0


            if self.config['save_models'] and \
                    (self.stats.n_env_steps % self.config['model_save_period'] == 0 or
                     self.stats.n_env_steps >= self.config['n_training_frames']):
                self.save_model(self.save_model_path)
            # loop_end = time.time()
            # episode_time += loop_end - loop_start
            # print('loop running time: %s Seconds'%(loop_end - loop_start))
            if self.stats.n_env_steps >= self.config['n_training_frames']:
                # print(' episode_time ---------- : %s Seconds'%(episode_time))
                # episode_time = 0
                break

        print('Env steps: {}'.format(self.stats.n_env_steps))

        if self.config['dump_replay_memory']:
            pickle.dump(self.student_agent.replay_memory,
                        open(os.path.join(self.save_replay_memory_path, self.run_id + '.rm'), 'wb'))

        self.session.close()

    # ==================================================================================================================

    def reset_env(self):
        self.episode_duration = 0
        self.episode_reward = 0.0
        self.episode_reward_real = 0.0

        self.stats.advices_reused_episode = 0
        self.stats.advices_reused_correct_episode = 0
        self.stats.exploration_steps_taken_episode = 0

        self.advices_reused_ep = 0
        self.advices_reused_ep_correct = 0

        render = self.stats.n_episodes % self.config['visualization_period'] == 0 and self.config['visualize_videos']

        if render:
            if self.config['env_type'] == ALE:
                self.video_recorder = gym_video_recorder. \
                    VideoRecorder(self.env,
                                  base_path=os.path.join(self.save_videos_path, '{}_{}'.format(
                                      str(self.stats.n_episodes), str(self.stats.n_env_steps))))

            elif self.config['env_type'] == BOX2D:
                self.video_recorder = gym_video_recorder. \
                    VideoRecorder(self.env,
                                  base_path=os.path.join(self.save_videos_path, '{}_{}'.format(
                                      str(self.stats.n_episodes), str(self.stats.n_env_steps))))

            elif self.config['env_type'] == GRIDWORLD:
                pass
            elif self.config['env_type'] == MAPE:
                pass
            elif self.config['env_type'] == MINATAR:
                pass

        self.obs_images.clear()

        obs = None

        if self.config['env_type'] == ALE:
            obs = self.env.reset()
        elif self.config['env_type'] == BOX2D:
            obs = self.env.reset()
        elif self.config['env_type'] == GRIDWORLD:
            obs = self.env.reset()
        elif self.config['env_type'] == MAPE:
            obs = self.env.reset()[0]
        elif self.config['env_type'] == MINATAR:
            self.env.reset()
            obs = self.env.state()

        if render:
            if self.config['env_type'] == ALE:
                self.video_recorder.capture_frame()
            elif self.config['env_type'] == BOX2D:
                self.video_recorder.capture_frame()
            elif self.config['env_type'] == GRIDWORLD:
                self.obs_images.append(self.render(self.env))
            elif self.config['env_type'] == MAPE:
                self.obs_images.append(self.env.render('rgb_array', visible=False)[0])
            elif self.config['env_type'] == MINATAR:
                self.obs_images.append(self.render(self.env))

        if self.config['advice_reuse_method'] == 'none':
            self.reuse_enabled = False
        else:
            if self.config['advice_reuse_method'] == 'restricted' or \
                    self.config['advice_reuse_method'] == 'extended':
                # Enable/disable advice reuse in the next episode
                if random.random() < self.advice_reuse_probability:
                    self.reuse_enabled = True
                else:
                    self.reuse_enabled = False
            else:
                # Default: no valid method
                self.reuse_enabled = False

        return obs, render

    # ==================================================================================================================

    def evaluate(self, utilise_advice_reuse=False, log_as_B=False):
        eval_render = self.stats.n_evaluations % self.config['evaluation_visualization_period'] == 0 and \
                      self.config['visualize_videos']

        eval_total_reward_real = 0.0
        eval_total_reward = 0.0
        eval_duration = 0

        eval_advices_reused = 0
        eval_advices_reused_correct = 0

        if self.config['env_type'] == ALE:
            self.eval_env.seed(self.config['env_evaluation_seed'])
        elif self.config['env_type'] == BOX2D:
            self.eval_env.seed(self.config['env_evaluation_seed'])
        elif self.config['env_type'] == GRIDWORLD:
            self.eval_env.set_random_state(self.config['env_evaluation_seed'])
        elif self.config['env_type'] == MAPE:
            self.eval_env.set_world_random_state(self.config['env_evaluation_seed'])
        elif self.config['env_type'] == MINATAR:
            self.eval_env.set_random_state(self.config['env_evaluation_seed'])

        if eval_render:
            if self.config['env_type'] == ALE:
                video_capture_eval = gym_video_recorder. \
                    VideoRecorder(self.eval_env, base_path=
                os.path.join(self.save_videos_path,
                             'E_{}_{}'.format(str(self.stats.n_episodes), str(self.stats.n_env_steps))))

            elif self.config['env_type'] == BOX2D:
                video_capture_eval = gym_video_recorder. \
                    VideoRecorder(self.eval_env, base_path=
                os.path.join(self.save_videos_path,
                             'E_{}_{}'.format(str(self.stats.n_episodes), str(self.stats.n_env_steps))))

            elif self.config['env_type'] == GRIDWORLD:
                pass
            elif self.config['env_type'] == MAPE:
                pass
            elif self.config['env_type'] == MINATAR:
                pass

        for i_eval_trial in range(self.config['n_evaluation_trials']):
            eval_obs_images = []

            eval_obs = None

            if self.config['env_type'] == ALE:
                eval_obs = self.eval_env.reset()
            elif self.config['env_type'] == BOX2D:
                eval_obs = self.eval_env.reset()
            elif self.config['env_type'] == GRIDWORLD:
                eval_obs = self.eval_env.reset()
            elif self.config['env_type'] == MAPE:
                eval_obs = self.eval_env.reset()[0]
            elif self.config['env_type'] == MINATAR:
                self.eval_env.reset()
                eval_obs = self.eval_env.state().astype(dtype=np.float32)

            eval_state_id = self.eval_env.get_state_id() if self.config['env_type'] == GRIDWORLD else None

            eval_episode_reward_real = 0.0
            eval_episode_reward = 0.0
            eval_episode_duration = 0

            while True:
                if eval_render:
                    if self.config['env_type'] == ALE:
                        video_capture_eval.capture_frame()
                    elif self.config['env_type'] == BOX2D:
                        video_capture_eval.capture_frame()
                    elif self.config['env_type'] == GRIDWORLD:
                        eval_obs_images.append(self.render(self.eval_env))
                    elif self.config['env_type'] == MAPE:
                        eval_obs_images.append(self.eval_env.render('rgb_array', visible=False)[0])
                    elif self.config['env_type'] == MINATAR:
                        eval_obs_images.append(self.render(self.eval_env))

                eval_action = None
                eval_teacher_action = None

                if self.config['execute_teacher_policy']:
                    if eval_teacher_action is None:
                        if self.config['env_type'] == GRIDWORLD:
                            eval_teacher_action = self.env.optimal_action()
                        else:
                            eval_teacher_action = self.teacher_agent.get_greedy_action(eval_obs)

                    eval_action = eval_teacher_action
                else:
                    if utilise_advice_reuse:
                        eval_action = self.utilise_advice_reuse(eval_obs, eval_state_id)

                        if eval_action is not None:
                            eval_advices_reused += 1

                            if eval_teacher_action is None:
                                if self.config['env_type'] == GRIDWORLD:
                                    eval_teacher_action = self.env.optimal_action()
                                else:
                                    eval_teacher_action = self.teacher_agent.get_greedy_action(eval_obs)

                            if eval_action == eval_teacher_action:
                                eval_advices_reused_correct += 1

                    if not utilise_advice_reuse or eval_action is None:
                        eval_action = self.student_agent.get_greedy_action(eval_obs)
                        # print(f"action is {eval_action}")

                eval_obs_next, eval_reward, eval_done = None, None, None

                if self.config['env_type'] == ALE:
                    eval_obs_next, eval_reward, eval_done, eval_info, eval_real_reward \
                        = self.eval_env.step(eval_action)

                elif self.config['env_type'] == BOX2D:
                    eval_obs_next, eval_reward, eval_done, eval_info = self.eval_env.step(eval_action)
                    eval_real_reward = eval_reward

                elif self.config['env_type'] == GRIDWORLD:
                    eval_obs_next, eval_reward, eval_done = self.eval_env.step(eval_action)
                    eval_real_reward = eval_reward

                elif self.config['env_type'] == MAPE:
                    eval_obs_next_n, eval_reward_n, eval_done_n, eval_info_n = self.eval_env.step([eval_action])
                    eval_obs_next, eval_reward, eval_done = eval_obs_next_n[0], eval_reward_n[0], eval_done_n[0]
                    eval_real_reward = eval_info_n['n'][0]

                elif self.config['env_type'] == MINATAR:
                    eval_reward, eval_done = self.eval_env.act(eval_action)
                    eval_obs_next = self.eval_env.state().astype(dtype=np.float32)
                    eval_real_reward = eval_reward

                eval_episode_reward_real += eval_real_reward
                eval_episode_reward += eval_reward

                eval_duration += 1
                eval_episode_duration += 1
                eval_obs = eval_obs_next

                eval_state_id = self.eval_env.get_state_id() if self.config['env_type'] == GRIDWORLD else None

                eval_done = eval_done or eval_episode_duration >= self.env_info['max_timesteps']

                if eval_done:
                    if eval_render:
                        if self.config['env_type'] == ALE:
                            video_capture_eval.capture_frame()
                            video_capture_eval.close()
                            video_capture_eval.enabled = False

                        elif self.config['env_type'] == BOX2D:
                            video_capture_eval.capture_frame()
                            video_capture_eval.close()
                            video_capture_eval.enabled = False

                        elif self.config['env_type'] == GRIDWORLD:
                            eval_obs_images.append(self.render(self.eval_env))
                            write_video(eval_obs_images, self.save_videos_path,
                                             'E_{}_{}'.format(str(self.stats.n_episodes), str(self.stats.n_env_steps)))

                        elif self.config['env_type'] == MAPE:
                            eval_obs_images.append(self.render(self.eval_env))
                            write_video(eval_obs_images, self.save_videos_path,
                                             'E_{}_{}'.format(str(self.stats.n_episodes), str(self.stats.n_env_steps)))

                        elif self.config['env_type'] == MINATAR:
                            eval_obs_images.append(self.render(self.eval_env))
                            write_video(eval_obs_images, self.save_videos_path,
                                             'E_{}_{}'.format(str(self.stats.n_episodes), str(self.stats.n_env_steps)))

                        eval_render = False

                    eval_total_reward += eval_episode_reward
                    eval_total_reward_real += eval_episode_reward_real
                    break

        eval_mean_reward = eval_total_reward / float(self.config['n_evaluation_trials'])
        eval_mean_reward_real = eval_total_reward_real / float(self.config['n_evaluation_trials'])

        if not log_as_B:
            self.stats.n_evaluations += 1

            self.stats.evaluation_reward_auc += np.trapz([self.stats.evaluation_reward_last, eval_mean_reward])
            self.stats.evaluation_reward_last = eval_mean_reward

            self.stats.evaluation_reward_real_auc += np.trapz(
                [self.stats.evaluation_reward_real_last, eval_mean_reward_real])
            self.stats.evaluation_reward_real_last = eval_mean_reward_real

            self.stats.update_summary_evaluation(eval_mean_reward,
                                                 eval_duration,
                                                 self.stats.evaluation_reward_auc,
                                                 eval_mean_reward_real,
                                                 self.stats.evaluation_reward_real_auc)

        else:
            self.stats.n_evaluations_b += 1

            self.stats.evaluation_b_reward_auc += np.trapz([self.stats.evaluation_b_reward_last, eval_mean_reward])
            self.stats.evaluation_b_reward_last = eval_mean_reward

            self.stats.evaluation_b_reward_real_auc += np.trapz(
                [self.stats.evaluation_b_reward_real_last, eval_mean_reward_real])
            self.stats.evaluation_b_reward_real_last = eval_mean_reward_real

            self.stats.update_summary_evaluation_b(eval_mean_reward,
                                                 eval_duration,
                                                 self.stats.evaluation_b_reward_auc,
                                                 eval_mean_reward_real,
                                                 self.stats.evaluation_b_reward_real_auc,
                                                   eval_advices_reused,
                                                   eval_advices_reused_correct)

        return eval_mean_reward, eval_mean_reward_real

    # ==================================================================================================================

    def save_model(self, save_model_path):
        model_path = os.path.join(os.path.join(save_model_path), 'model-{}.ckpt').format(
            self.stats.n_env_steps)
        print('[{}] Saving model... {}'.format(self.stats.n_env_steps, model_path))
        self.saver.save(self.session, model_path)

    # ==================================================================================================================

    def save_obs_image(self, obs, t):
        if self.config['env_type'] == ALE:
            black_line = np.zeros((84, 1), dtype=np.uint8)
            rendered_frame = self.env.render(mode='rgb_array')
            cv2.imwrite(self.save_obs_real_images_path + '/' + str(t) + '.png', rendered_frame[:, :, ::-1])
            cv2.imwrite(self.save_obs_agent_images_path + '/' + str(t) + '.png', np.asarray(np.hstack((
            obs[0], black_line, obs[1], black_line, obs[2], black_line, obs[3]))))

    # ==================================================================================================================
    # Get uncertainty value by considering the available options

    def get_student_uncertainty(self, obs):
        if self.config['dqn_type'] == 'egreedy' and self.config['dqn_dropout']:
            return self.student_agent.get_uncertainty(obs)
        elif self.config['dqn_type'] == 'noisy':
            return self.student_agent.get_uncertainty(obs, True)
        elif self.config['dqn_twin']:
            uncertainty, _, _ = self.dqn_twin.get_uncertainty(obs)
            return uncertainty
        else:
            return 0.

    # ==================================================================================================================

    def utilise_advice_reuse(self, obs, state_id):
        reuse_advice = False
        if self.config['advice_collection_method'] == 'dual_uc':
            if self.student_agent.replay_memory.__len__() > self.config['dqn_rm_init']:
                uc_value, _, _ = self.dqn_twin.get_uncertainty(obs)
                is_uncertain = False

                # (1) Adaptive threshold mode
                if self.config['use_proportional_student_model_uc_th']:
                    if len(self.student_model_uc_values_buffer) < \
                            self.config['proportional_student_model_uc_th_window_size_min']:
                        pass
                    else:
                        sorted_values = sorted(self.student_model_uc_values_buffer)
                        percentile_th = np.percentile(sorted_values,
                                                      self.config['proportional_student_model_uc_th_percentile'])
                        if uc_value > percentile_th:
                            is_uncertain = True

                # (2) Constant threshold mode
                else:
                    if uc_value > self.config['student_model_uc_th']:
                        is_uncertain = True

                if is_uncertain:
                    if self.initial_imitation_is_performed:
                        bc_uncertainty = self.bc_model.get_uncertainty(obs)
                        if bc_uncertainty < self.config['teacher_model_uc_th']:
                            reuse_advice = True
        else:
            if self.config['advice_reuse_method'] != 'none' and \
                    self.initial_imitation_is_performed:

                if self.config['advice_imitation_method'] == 'tabular_lookup':
                    if state_id in self.advice_lookup_table:
                        reuse_advice = True
                else:
                    bc_uncertainty = self.bc_model.get_uncertainty(obs)
                    if bc_uncertainty < self.config['teacher_model_uc_th']:
                        reuse_advice = True

        if reuse_advice:
            if self.config['advice_imitation_method'] == 'tabular_lookup':
                return self.advice_lookup_table[state_id]
            else:
                return np.argmax(self.bc_model.get_action_probs(obs))

        return None

# ======================================================================================================================

def save_config(config, filepath):
    fo = open(filepath, "w")
    for k, v in config.items():
        fo.write(str(k) + '>> ' + str(v) + '\n')
    fo.close()

# ======================================================================================================================

def write_video(images, save_videos_path, filename):
    v_w = np.shape(images[0])[0]
    v_h = np.shape(images[0])[1]
    filename_full = os.path.join(save_videos_path, str(filename))
    video = cv2.VideoWriter(filename_full + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (v_h, v_w))
    i = 0
    for image in images:
        i += 1
        video.write(image)
    video.release()

# ======================================================================================================================

def train_behavioural_cloner(bc_model, n_iters):
    if bc_model.replay_memory.__len__() == 0:
        print('\nBehavioural cloner has 0 samples. Skipping training.')
    else:
        print('\nTraining behavioural cloner with {} samples for {} steps...'.format(
            bc_model.replay_memory.__len__(), n_iters))
        for _ in range(n_iters):
            bc_model.feedback_learn()


# ======================================================================================================================

def evaluate_behavioural_cloner(bc_model):
    n_samples = bc_model.replay_memory.__len__()

    if n_samples == 0:
        return np.inf, 0.01

    uc_values_all, uc_values_correct, uc_values_incorrect = [], [], []
    n_correct, n_incorrect = 0, 0

    for i in range(n_samples):
        bc_obs = bc_model.replay_memory._storage[i][0]
        bc_act = bc_model.replay_memory._storage[i][1]
        uncertainty = bc_model.get_uncertainty(bc_obs)
        prediction = np.argmax(bc_model.get_action_probs(bc_obs))

        uc_values_all.append(uncertainty)
        if bc_act == prediction:
            uc_values_correct.append(uncertainty)
            n_correct += 1
        else:
            uc_values_incorrect.append(uncertainty)
            n_incorrect += 1

    print('Post imitation analysis is completed.')
    print('All samples:')
    print('> Max:', np.max(uc_values_all))
    print('> Min:', np.min(uc_values_all))
    print('> Avg:', np.mean(uc_values_all))
    print('> 80%:', np.percentile(uc_values_all, 80))
    print('> 90%:', np.percentile(uc_values_all, 90))
    print('')
    if len(uc_values_correct) > 0:
        print('Correct samples:')
        print('> Max:', np.max(uc_values_correct))
        print('> Min:', np.min(uc_values_correct))
        print('> Avg:', np.mean(uc_values_correct))
        print('> 80%:', np.percentile(uc_values_correct, 80))
        print('> 90%:', np.percentile(uc_values_correct, 90))
    else:
        print('No correct samples.')
    print('')

    if len(uc_values_incorrect) > 0:
        print('Incorrect samples:')
        print('> Max:', np.max(uc_values_incorrect))
        print('> Min:', np.min(uc_values_incorrect))
        print('> Avg:', np.mean(uc_values_incorrect))
        print('> 80%:', np.percentile(uc_values_incorrect, 80))
        print('> 90%:', np.percentile(uc_values_incorrect, 90))
    else:
        print('No incorrect samples.')

    accuracy = (n_correct / (n_correct + n_incorrect)) * 100.0
    print('\nAccuracy:', accuracy)
    print('')

    return np.percentile(uc_values_correct, 90), accuracy

# ======================================================================================================================

def singlematrix(dims, middle_val, ax=None, triplotkw={}, tripcolorkw={}, vmin=None, vmax=None):
    if not ax:
        ax = plt.gca()
    n = middle_val.shape[0]
    m = middle_val.shape[1]
    a = np.array([[0, 0], [0, 1], [.5, .5], [1, 0], [1, 1]])
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])
    A = np.zeros((n * m * 5, 2))
    Tr = np.zeros((n * m * 4, 3))
    for i in range(n):
        for j in range(m):
            k = i * m + j
            A[k * 5:(k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
            Tr[k * 4:(k + 1) * 4, :] = tr + k * 5

    C = np.c_[middle_val.flatten(), middle_val.flatten(), middle_val.flatten(), middle_val.flatten()].flatten()

    if vmin is None:
        tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)
    else:
        tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, vmin=vmin, vmax=vmax, **tripcolorkw)

    height = dims[0]
    width = dims[1]
    for y in range(height):
        for x in range(width):
            start = 5 * (y * height + x)
            A[[start, start + 2], :] = None

    ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
    return tripcolor

# ======================================================================================================================

def quadruplematrix(dims, left_val, down_val, right_val, up_val, ax=None, triplotkw={}, tripcolorkw={}):
    if not ax:
        ax = plt.gca()
    n = left_val.shape[0]
    m = left_val.shape[1]
    a = np.array([[0, 0], [0, 1], [.5, .5], [1, 0], [1, 1]])
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])
    A = np.zeros((n * m * 5, 2))
    Tr = np.zeros((n * m * 4, 3))
    for i in range(n):
        for j in range(m):
            k = i * m + j
            A[k * 5:(k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
            Tr[k * 4:(k + 1) * 4, :] = tr + k * 5

    C = np.c_[left_val.flatten(), down_val.flatten(), right_val.flatten(), up_val.flatten()].flatten()

    tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)

    height = dims[0]
    width = dims[1]
    for y in range(height):
        #y_inv = (height - 1) - y
        for x in range(width):
            #if grid[y_inv][x] == 1:
            start = 5 * (y * height + x)
            A[[start, start + 2], :] = None

    ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
    return tripcolor

# ======================================================================================================================

def generate_grid_visualisation(env, config, save_path, step_number, values):
    height = env.height
    width = env.width

    fig, ax = plt.subplots(figsize=(28, 7), dpi=120, nrows=1, ncols=4)

    ax = np.expand_dims(ax, axis=0)

    ax[0][0].set_title('Uncertainty')
    ax[0][1].set_title('Student Actions')
    ax[0][2].set_title('Teacher Actions')
    ax[0][3].set_title('Teacher Clone Actions')

    ax_idx = [(0, 0), (0, 1), (0, 2), (0, 3)]

    for v in range(4):
        if v == 0:
            cmap_str = 'RdPu' # 'coolwarm'
        else:
            cmap_str = 'RdYlGn'

        if v == 0:
            vmin, vmax = 0.01, 0.75
        else:
            vmin, vmax = None, None

        singlematrix((height, width), values[v], ax=ax[ax_idx[v][0]][ax_idx[v][1]],
                     triplotkw={"color": "k", "lw": 1}, tripcolorkw={'cmap': cmap_str}, vmin=vmin, vmax=vmax)

        ax[ax_idx[v][0]][ax_idx[v][1]].margins(0)
        ax[ax_idx[v][0]][ax_idx[v][1]].set_aspect("equal")

        for y in range(height):
            for x in range(width):
                if y == 0 and x == 8:
                    continue
                y_inv = (height - 1) - y
                if env.grid[0][y][x] == 0:
                    if v == 0:  # Uncertainty
                        ax[ax_idx[v][0]][ax_idx[v][1]].text(x + 0.5, y_inv + 0.5,
                                                            "{0:.3f}".format(values[v][y_inv, x]),
                                                            ha="center", va="center", color="k")
                    else:
                        ax[ax_idx[v][0]][ax_idx[v][1]].text(x + 0.5, y_inv + 0.5,
                                                        "{}".format(values[v][y_inv, x]),
                                                        ha="center", va="center", color="k")

        for y in range(height):
            for x in range(width):
                y_inv = (height - 1) - y
                if env.grid[0][y][x] == 1:
                    ax[ax_idx[v][0]][ax_idx[v][1]].add_patch(
                        patches.Rectangle(
                            (x, y_inv),  # (x,y)
                            1.0,  # width
                            1.0,  # height
                            fill=True,
                            color='darkslategrey'
                        )
                    )

        # Goal position:
        ax[ax_idx[v][0]][ax_idx[v][1]].add_patch(
            patches.Rectangle(
                (8, 8),  # (x,y)
                1.0,  # width
                1.0,  # height
                fill=True,
                color='teal'
            )
        )

    for ax_i in range(1):
        for ax_j in range(4):
            ax[ax_i][ax_j].set_xticks([])
            ax[ax_i][ax_j].set_yticks([])

    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig(os.path.join(save_path, str(step_number)+'.jpg'))

    fig.clear()
    plt.close(fig)