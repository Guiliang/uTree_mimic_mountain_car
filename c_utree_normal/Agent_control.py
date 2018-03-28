import ast
import csv
import os
import pickle
import random
import unicodedata
import numpy as np
import scipy.io as sio
import scipy as sp
import scipy.stats
import C_UTree_Galen_control


def read_actions(game_directory, game_dir):
    actions = sio.loadmat(game_directory + game_dir + "/action_{0}.mat".format(game_dir))
    return actions["action"]


def read_states(game_directory, game_dir):
    states = sio.loadmat(game_directory + game_dir + "/state_{0}.mat".format(game_dir))
    return states["state"]


def read_rewards(game_directory, game_dir):
    rewards = sio.loadmat(game_directory + game_dir + "/reward_{0}.mat".format(game_dir))
    return rewards["reward"]


def read_train_info(game_directory, game_dir):
    train_info = sio.loadmat(game_directory + game_dir + "/training_data_dict_all_name.mat")
    return train_info['training_data_dict_all_name']


class CUTreeAgent:
    """
    Agent that implements McCallum's Sport-Analytic-U-Tree algorithm
    """

    def __init__(self, problem, max_hist, check_fringe_freq, is_episodic=0):

        self.utree = C_UTree_Galen_control.CUTree(gamma=problem.gamma, n_actions=len(problem.actions),
                                                  dim_sizes=problem.dimSizes, dim_names=problem.dimNames,
                                                  max_hist=max_hist, is_episodic=is_episodic, minSplitInstances=1)
        self.cff = check_fringe_freq
        self.problem = problem
        # self.TREE_PATH = "./csv_test/"
        self.SAVE_PATH = "/Local-Scratch/UTree model/mountaincar/model_control_normal_save/"
        self.epsilon_decay = .995
        self.max_episodes = 1000
        self.epsilon = 1

    def update(self, currentObs, nextObs, action, reward, terminal=0, check_fringe=0,
               beginning_flag=False):
        """
        update the tree
        :param nextObs: next observation
        :param action: action taken
        :param away_reward: reward get
        :param terminal: if end
        :param check_fringe: whether to check fringe
        :return:
        """
        t = self.utree.getTime()  # return the length of history
        i = C_UTree_Galen_control.Instance(t, currentObs, action, nextObs,
                                           reward)  # create a new instance T_t =[Time, I, a, I', r(I,a,I')]

        self.utree.updateCurrentNode(i, beginning_flag)
        self.utree.sweepLeaves()  # value iteration is performed here, every timestep might be too slow?

        if check_fringe:
            self.utree.testFringe()  # ks test is performed here

    def get_Q_values(self, currentObs, nextObs, action, home_reward, away_reward, home_identifier=None):
        inst = C_UTree_Galen_control.Instance(-1, currentObs, action, nextObs, home_reward,
                                              away_reward, home_identifier)

        node = self.utree.getAbsInstanceLeaf(inst)
        return node.utility(ishome=True), node.utility(ishome=False)

    def read_Utree(self, game_number):
        self.utree = pickle.load(open(self.SAVE_PATH + "pickle_Game_File_" + str(game_number) + ".p", 'rb'))

    def save_csv_q_values(self, q_values, filename):
        with open(filename, 'wb') as csvfile:
            fieldname = ['event_number', 'q_home', 'q_away']
            writer = csv.writer(csvfile)
            # writer.writerow(fieldname)
            event_counter = 1
            for q_tuple in q_values:
                writer.writerow([event_counter, q_tuple[0], q_tuple[1]])
                event_counter += 1

    def print_event_values(self):
        read_game_number = 250
        self.read_Utree(game_number=read_game_number)
        print "finishing read tree"

        game_directory = self.problem.games_directory
        game_dir_all = os.listdir(game_directory)

        game_to_print_list = [250]

        for game_number in game_to_print_list:
            q_values = []
            game_dir = game_dir_all[game_number - 1]

            states = read_states(game_directory, game_dir)
            actions = read_actions(game_directory, game_dir)
            rewards = (read_rewards(game_directory, game_dir))[0]
            training_information = read_train_info(game_directory, game_dir)
            assert states.shape[0] == actions.shape[0] and actions.shape[0] == rewards.shape[0] and rewards.shape[0] == \
                                                                                                    training_information.shape[
                                                                                                        0]

            event_number = len(states)

            for index in range(0, event_number):
                action_name = unicodedata.normalize('NFKD', actions[index]).encode('ascii', 'ignore').strip()
                action = self.problem.actions[action_name]
                currentObs = states[index]
                if index + 1 == event_number:
                    nextObs = states[index]
                else:
                    nextObs = states[index + 1]
                reward = rewards[index]

                calibrate_name_str = unicodedata.normalize('NFKD', training_information[index]).encode('ascii',
                                                                                                       'ignore')
                calibrate_name_dict = ast.literal_eval(calibrate_name_str)
                home_identifier = int(calibrate_name_dict.get('home'))

                if action_name == 'goal':  # goal is an absorbing state
                    nextObs = states[index]

                if reward < 0:
                    home_reward = 0
                    away_reward = 1
                elif reward > 0:
                    home_reward = 1
                    away_reward = 0
                else:
                    home_reward = 0
                    away_reward = 0

                Q_home, Q_away = self.get_Q_values(currentObs, nextObs, action, home_reward, away_reward,
                                                   home_identifier)
                q_values.append([Q_home, Q_away])

            self.save_csv_q_values(q_values,
                                   './q_values/Qvalues_game{0}_model_normal_r{1}'.format(game_number, read_game_number))

    # def episode(self, timeout=int(1e5), save_checkpoint_flag=1):
    #     """
    #     start to build the tree within an episode
    #     :param timeout: steps number?
    #     :return: None
    #     """
    #     # act_hist = np.zeros(len(self.problem.actions))  # record number of appearance of different actions
    #
    #     game_directory = self.problem.games_directory
    #
    #     game_dir_all = os.listdir(game_directory)
    #
    #     for game_number in range(0, len(game_dir_all)):
    #         game_dir = game_dir_all[game_number]
    #         beginning_flag = True
    #
    #         states = read_states(game_directory, game_dir)
    #         actions = read_actions(game_directory, game_dir)
    #         rewards = (read_rewards(game_directory, game_dir))[0]
    #         training_information = read_train_info(game_directory, game_dir)
    #         assert states.shape[0] == actions.shape[0] and actions.shape[0] == rewards.shape[0] and rewards.shape[0] == \
    #                                                                                                 training_information.shape[
    #                                                                                                     0]
    #
    #         event_number = len(states)
    #
    #         for index in range(0, event_number):
    #             action_name = unicodedata.normalize('NFKD', actions[index]).encode('ascii', 'ignore').strip()
    #             action = self.problem.actions[action_name]
    #             currentObs = states[index]
    #             nextObs = states[index + 1]
    #             reward = rewards[index]
    #
    #             calibrate_name_str = unicodedata.normalize('NFKD', training_information[index]).encode('ascii',
    #                                                                                                    'ignore')
    #             calibrate_name_dict = ast.literal_eval(calibrate_name_str)
    #             home_identifier = int(calibrate_name_dict.get('home'))
    #
    #             if action_name == 'goal':  # goal is an absorbing state
    #                 nextObs = states[index]
    #
    #             if reward < 0:
    #                 home_reward = 0
    #                 away_reward = 1
    #             elif reward > 0:
    #                 home_reward = 1
    #                 away_reward = 0
    #             else:
    #                 home_reward = 0
    #                 away_reward = 0
    #
    #             # act_hist[action] += 1  # action
    #             if self.problem.isEpisodic:  # what is episodic, guess we don't need to worry?
    #
    #                 if index == event_number - 2:  # the last states[event_number-1] won't have next transition, so terminate
    #                     terminal = True
    #                 else:
    #                     terminal = False
    #
    #                 if terminal:
    #                     # nextObs = [-1]
    #                     self.update(currentObs, nextObs, action, home_reward, away_reward)
    #                     break
    #                     # self.problem.reset()
    #                     # return index
    #
    #             if index % self.cff == 0:  # check fringe, check fringe after cff iterations
    #                 print "=============== update starts ==============="
    #                 self.update(currentObs, nextObs, action, home_reward, away_reward, check_fringe=1,
    #                             beginning_flag=beginning_flag, home_identifier=home_identifier)
    #                 print "=============== update finished ===============\n"
    #             else:
    #                 self.update(currentObs, nextObs, action, home_reward, away_reward, check_fringe=0,
    #                             beginning_flag=beginning_flag, home_identifier=home_identifier)
    #             beginning_flag = False
    #             # self.utree.print_tree(file_directory='tree-structure-till-{0}.txt'.format(game_dir))
    #             # break
    #
    #         if self.problem.isEpisodic:
    #             # print out tree info
    #             print "*** Writing Game File {0}***".format(str(game_number + 1))
    #             self.utree.print_tree_structure()
    #             if save_checkpoint_flag and (game_number + 1) % 10 == 0:
    #                 pickle.dump(self.utree,
    #                             open(self.SAVE_PATH + "pickle_Game_File_" + str(game_number + 1) + ".p", 'wb'))
    #                 self.utree.tocsvFile(self.TREE_PATH + "Game_File_" + str(game_number + 1) + ".csv")
    #     print "finish"

    def executePolicy(self, currentObs, epsilon):
        """
        epsilon-greedy policy basing on Q
        :param epsilon: epsilon
        :return: action to take
        """
        test = random.random()
        if test < epsilon:
            return random.choice(range(len(self.problem.actions)))
        return self.utree.getBestAction(currentObs, self.problem.actions.values())

    def reward_shaping(self, position):

        if position < -0.3:
            return (-0.3-position)/0.9*2-1
        else:
            return (position+0.3)/0.9*2-1

        # return (position + 1.2) / 1.8 * 2 - 1

    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
        return m, h

    def play(self):

        self.read_Utree(201)
        reward_total_list = []
        for game_number in range(200):
            reward_total = 0
            act_hist = np.zeros(len(self.problem.actions))  # record number of appearance of different actions
            currentObs = self.problem.env.reset()
            self.epsilon *= self.epsilon_decay
            for i in range(0, 1000):

                action = self.executePolicy(currentObs, 0.6)  # execute epsilon-greedy policy basing on Q

                act_hist[action] += 1  # action

                nextObs, reward, done, _ = self.problem.env.step(action)
                reward_total += reward

                self.problem.env.render()

                if done:
                    break

                currentObs = nextObs
            reward_total_list.append(reward_total)
        m, h = self.mean_confidence_interval(reward_total_list)

        print 'DRL mean:{0}, interval:{1}'.format(str(m), str(h))


    def episode(self, timeout=int(1e5), save_checkpoint_flag=1):
        """
        start to build the tree within an episode
        :param timeout: max time steps
        :return: None
        """
        beginning_flag = True
        for game_number in range(self.max_episodes):

            act_hist = np.zeros(len(self.problem.actions))  # record number of appearance of different actions

            currentObs = self.problem.env.reset()
            done, tran_start = False, False
            self.epsilon *= self.epsilon_decay
            for i in range(0, 1000):

                action = self.executePolicy(currentObs, self.epsilon)  # execute epsilon-greedy policy basing on Q

                act_hist[action] += 1  # action

                nextObs, reward, done, _ = self.problem.env.step(action)

                reward = self.reward_shaping(nextObs[0])

                if reward > 1 or reward <-1:
                    raise ValueError('reward problem')

                self.problem.env.render()

                if i == 999:
                    nextObs = [-1]
                    self.update(currentObs=currentObs, nextObs=nextObs, action=action, reward=reward,
                                check_fringe=1,
                                beginning_flag=beginning_flag)
                    break

                self.update(currentObs=currentObs, nextObs=nextObs, action=action, reward=reward, check_fringe=0,
                            beginning_flag=beginning_flag)
                beginning_flag = False
                currentObs = nextObs

            self.utree.print_tree_structure()
            print 'game number:{0}, with epsilon:{1}'.format(str(game_number), str(self.epsilon))
            if save_checkpoint_flag and game_number % 100 == 0:
                print "*** Writing Game File {0}***".format(str(game_number + 1))
                pickle.dump(self.utree,
                            open(self.SAVE_PATH + "pickle_Game_File_" + str(game_number + 1) + ".p", 'wb'))
