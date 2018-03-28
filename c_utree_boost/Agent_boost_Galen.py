# uncompyle6 version 2.14.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.4.3 (default, Nov 17 2016, 01:08:31) 
# [GCC 4.8.4]
# Embedded file name: /Local-Scratch/PycharmProjects/Sport-Analytic-U-Tree/continuous-U-Tree-ice-hockey/c_utree_oracle/Agent_boost_Galen.py
# Compiled at: 2018-01-03 14:44:40
import gc
import numpy as np, ast, scipy.io as sio, os, unicodedata, pickle, C_UTree_boost_Galen as C_UTree, csv
import tensorflow as tf
from scipy.stats import pearsonr

import linear_regression
import sys
# from tensorflow.python.framework import ops
from pympler.tracker import SummaryTracker

tracker = SummaryTracker()


class CUTreeAgent:
    """
      Agent that implements McCallum's Sport-Analytic-U-Tree algorithm
    """

    def __init__(self, problem, max_hist, check_fringe_freq, is_episodic=0, training_mode=''):
        self.utree = C_UTree.CUTree(gamma=problem.gamma, n_actions=len(problem.actions), dim_sizes=problem.dimSizes,
                                    dim_names=problem.dimNames, max_hist=max_hist, is_episodic=is_episodic,
                                    training_mode=training_mode)
        self.cff = check_fringe_freq
        self.valiter = 1
        self.problem = problem
        # self.TREE_PATH = './csv_oracle_linear_qsplit_test/'
        self.SAVE_PATH = '/Local-Scratch/UTree model/mountaincar/model_boost_linear_qsplit_noabs_save{0}/'.format(
            training_mode)
        self.SAVE_MODEL_TREE_PATH = '/Local-Scratch/UTree model/moutaincar/model_boost_add_linear_qsplit_save{0}/'.format(
            training_mode)
        self.PRINT_TREE_PATH = './print_tree_record/print_mountaincar_boost_linear_tree_split{0}.txt'.format(
            training_mode)
        self.training_mode = training_mode
        # print(tf.__version__)

    def update(self, currentObs, nextObs, action, reward, qValue, value_iter=0, check_fringe=0, home_identifier=None,
               beginflag=False):
        """
        update the tree
        :param currentObs: current observation
        :param nextObs: next observation
        :param action: action taken
        :param reward: reward get
        :param terminal: if end
        :param check_fringe: whether to check fringe
        :return:
        """
        t = self.utree.getTime()
        i = C_UTree.Instance(t, currentObs, action, nextObs, reward, qValue)
        self.utree.updateCurrentNode(i, beginflag)
        if check_fringe:
            self.utree.testFringe()

    def getQ(self, currentObs, nextObs, action, reward, home_identifier):
        """
        only insert instance to history
        :param currentObs:
        :param nextObs:
        :param action:
        :param reward:
        :return:
        """
        t = self.utree.getTime()
        i = C_UTree.Instance(t, currentObs, action, nextObs, reward, np.zeros(2))
        q_h, q_a = self.utree.getInstanceQvalues(i, reward)
        with open('csv_Q/Game1-5_oracle.csv', 'ab') as (csvfile):
            writer = csv.writer(csvfile)
            writer.writerow([t, q_h / (q_h + q_a), q_a / (q_h + q_a)])

    def read_csv_game_record(self, csv_dir):
        dict_all = []
        with open(csv_dir, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dict_all.append(row)
        return dict_all

    def get_Q_values(self, currentObs, nextObs, action, reward, home_identifier=None):
        inst = C_UTree.Instance(-1, currentObs, action, nextObs, reward, None)
        node = self.utree.getAbsInstanceLeaf(inst)
        return node.utility(home_identifier=True)

    def get_Q_values_linear_tree(self, currentObs, nextObs, action, reward, qValue,
                                 home_identifier=None, smooth_flag=None, merge_count=None):
        # ops.reset_default_graph()
        sess = tf.Session()
        inst = C_UTree.Instance(-1, currentObs, action, nextObs, reward, None)
        node = self.utree.getAbsInstanceLeaf(inst)
        LR = linear_regression.LinearRegression()
        LR.read_weights(weights=node.weight, bias=node.bias)
        LR.readout_linear_regression_model()
        sess.run(LR.init)
        temp = sess.run(LR.pred, feed_dict={LR.X: [inst.currentObs]}).tolist()
        LR.delete_para()
        sess.close()
        del LR
        del sess
        gc.collect()
        # print temp[0]
        # gc.collect()
        if smooth_flag:
            return_list = []
            tolertion_level = 0.15
            home_diff_abs = abs(qValue[0] - temp[0][0])
            return_list.append(node.qValues_home[action]) if home_diff_abs >= tolertion_level else return_list.append(
                temp[0][0])
            away_diff_abs = abs(qValue[1] - temp[0][1])
            return_list.append(node.qValues_away[action]) if away_diff_abs >= tolertion_level else return_list.append(
                temp[0][1])
            end_diff_abs = abs(qValue[2] - temp[0][2])
            return_list.append(node.qValues_end[action]) if end_diff_abs >= tolertion_level else return_list.append(
                temp[0][2])
            if home_diff_abs >= tolertion_level or away_diff_abs >= tolertion_level or end_diff_abs >= tolertion_level:
                merge_count += 1
            return return_list, merge_count
        else:
            return temp[0], merge_count

    def read_Utree(self, save_path, game_number):
        print >> sys.stderr, 'reading from {0}, starting at {1}'.format(self.SAVE_PATH, game_number)
        # temp = '{0}pickle_Game_File_{1}.p'.format(save_path, str(game_number))
        # print temp
        # temp1 = pickle.load(open(save_path + 'pickle_Game_File_' + str(game_number) + '.p', 'rb'))
        self.utree = pickle.load(open(save_path + 'pickle_Game_File_' + str(game_number) + '.p', 'rb'))
        self.utree.training_mode = self.training_mode
        self.utree.game_number = game_number + 1

    def save_csv_q_values(self, q_values, filename):
        with open(filename, 'wb') as (csvfile):
            fieldname = [
                'event_number', 'q_home', 'q_away', 'q_end']
            writer = csv.writer(csvfile)
            event_counter = 1
            for q_tuple in q_values:
                writer.writerow([event_counter, q_tuple[0], q_tuple[1], q_tuple[2]])
                event_counter += 1

    # def print_event_values(self, save_path):
    #     read_game_number = 240
    #     self.read_Utree(game_number=read_game_number, save_path=save_path)
    #     print 'finishing read tree'
    #     game_directory = self.problem.games_directory
    #     game_dir_all = os.listdir(game_directory)
    #     game_to_print_list = [
    #         1]
    #     for game_number in game_to_print_list:
    #         q_values = []
    #         game_dir = game_dir_all[game_number - 1]
    #         states = read_states(game_directory, game_dir)
    #         actions = read_actions(game_directory, game_dir)
    #         rewards = read_rewards(game_directory, game_dir)[0]
    #         training_information = read_train_info(game_directory, game_dir)
    #         assert states.shape[0] == actions.shape[0] and actions.shape[0] == rewards.shape[0] and rewards.shape[0] == \
    #                                                                                                 training_information.shape[
    #                                                                                                     0]
    #         event_number = len(states)
    #         for index in range(0, event_number):
    #             action_name = unicodedata.normalize('NFKD', actions[index]).encode('ascii', 'ignore').strip()
    #             action = self.problem.actions[action_name]
    #             currentObs = states[index]
    #             if index + 1 == event_number:
    #                 nextObs = states[index]
    #             else:
    #                 nextObs = states[index + 1]
    #             reward = rewards[index]
    #             calibrate_name_str = unicodedata.normalize('NFKD', training_information[index]).encode('ascii',
    #                                                                                                    'ignore')
    #             calibrate_name_dict = ast.literal_eval(calibrate_name_str)
    #             home_identifier = int(calibrate_name_dict.get('home'))
    #             if action_name == 'goal':
    #                 nextObs = states[index]
    #             if reward < 0:
    #                 home_reward = 0
    #                 away_reward = 1
    #             else:
    #                 if reward > 0:
    #                     home_reward = 1
    #                     away_reward = 0
    #                 else:
    #                     home_reward = 0
    #                     away_reward = 0
    #             Q_home, Q_away, Q_end = self.get_Q_values(currentObs, nextObs, action, home_reward, away_reward,
    #                                                       home_identifier)
    #             q_values.append([Q_home, Q_away, Q_end])
    #
    #         self.save_csv_q_values(q_values,
    #                                ('./oracle_qsplit_values/Qvalues_game{0}_model_normal_r{1}.csv').format(game_number,
    #                                                                                                        read_game_number))

    def normalization_q(self, q_list):
        sum_q = 0
        for index in range(0, len(q_list)):
            q_i = q_list[index]
            if q_i < 0:
                q_i = 0
                q_list[index] = q_i
            sum_q += q_i
        if sum_q != 0:
            q_norm_list = []
            for q_i in q_list:
                q_norm_list.append(float(q_i) / sum_q)
        else:
            q_norm_list = [0.5, 0.5, 0]
        print q_norm_list
        return q_norm_list

    def smooth_list(self, y, box_pts=10):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def smoothing_q(self, q_list):
        q_array = np.asarray(q_list)
        q_home = self.smooth_list(q_array[:, 0])
        q_away = self.smooth_list(q_array[:, 1])
        q_end = self.smooth_list(q_array[:, 2])

        return zip(q_home.tolist(), q_away.tolist(), q_end.tolist())

    def boost_tree_testing_performance(self, save_path, read_game_number, save_correlation_dir, save_mse_dir, save_mae_dir, save_rae_dir, save_rse_dir):
        print >> sys.stderr, 'starting from {0}'.format(read_game_number)
        self.utree = pickle.load(open(save_path + 'pickle_Game_File_' + str(read_game_number) + '.p', 'rb'))
        print >> sys.stderr, 'finishing read tree'
        game_directory = self.problem.games_directory

        game_testing_record_dict = {}

        game_to_print_list = range(1001, 1101)
        for game_number in game_to_print_list:

            game_record = self.read_csv_game_record(
                self.problem.games_directory + 'record_moutaincar_transition_game{0}.csv'.format(int(game_number)))

            event_number = len(game_record)

            for index in range(0, event_number):

                transition = game_record[index]
                currentObs = transition.get('observation').split('$')
                nextObs = transition.get('newObservation').split('$')
                reward = float(transition.get('reward'))
                action = float(transition.get('action'))
                qValue = float(transition.get('qValue'))

                inst = C_UTree.Instance(-1, currentObs, action, nextObs, reward, None)
                node = self.utree.getAbsInstanceLeaf(inst)

                if game_testing_record_dict.get(node) is None:
                    game_testing_record_dict.update({node: np.array([[currentObs, qValue, action]])})
                else:
                    node_record = game_testing_record_dict.get(node)
                    node_record = np.concatenate((node_record, [[currentObs, qValue, action]]), axis=0)
                    game_testing_record_dict.update({node: node_record})

        all_q_values_record = {'output_q': [],
                               'test_q': [],
                               'oracle_q': [],
                               'merge_q': []
                               }

        for node in game_testing_record_dict.keys():
            # print node.idx
            node_record = game_testing_record_dict.get(node)
            currentObs_node = node_record[:, 0]
            qValues_node = node_record[:, 1]
            actions = node_record[:, 2]

            test_q = all_q_values_record.get('test_q')
            test_append_q = [qValues_list for qValues_list in qValues_node]
            test_q += test_append_q

            sess = tf.Session()
            LR = linear_regression.LinearRegression()
            LR.read_weights(weights=node.weight, bias=node.bias)
            LR.readout_linear_regression_model()
            sess.run(LR.init)
            qValues_output = sess.run(LR.pred, feed_dict={LR.X: currentObs_node.tolist()}).tolist()

            output_q = all_q_values_record.get('output_q')
            output_append_q = [qValues_list[0] for qValues_list in qValues_output]
            output_q += output_append_q

            oracle_q = all_q_values_record.get('oracle_q')
            oracle_append_q = [node.qValues[int(action)] for action in actions]
            oracle_q += oracle_append_q

            merge_q = all_q_values_record.get('merge_q')
            merge_append_q = self.merge_oracle_linear_q(test_append_q, output_append_q,
                                                        oracle_append_q)
            merge_q += merge_append_q

        # self.compute_correlation(all_q_values_record, save_correlation_dir)
        # self.compute_mae(all_q_values_record, save_mae_dir)
        # self.compute_mse(all_q_values_record, save_mse_dir)
        self.compute_rae(all_q_values_record, save_rae_dir)
        self.compute_rse(all_q_values_record, save_rse_dir)

    def merge_oracle_linear_q(self, test_qs, linear_qs, oracle_qs):
        criteria = 0.7
        merge_qs = []
        for index in range(0, len(test_qs)):
            if abs(test_qs[index] - linear_qs[index]) > criteria:
                merge_qs.append(oracle_qs[index])
            else:
                merge_qs.append(linear_qs[index])
        return merge_qs

    def compute_correlation(self, all_q_values_record, save_correlation_dir):
        linear_correl = pearsonr(all_q_values_record.get('output_q'), all_q_values_record.get('test_q'))[0]
        oracle_correl = pearsonr(all_q_values_record.get('oracle_q'), all_q_values_record.get('test_q'))[0]
        merge_correl = pearsonr(all_q_values_record.get('merge_q'), all_q_values_record.get('test_q'))[0]

        text_file = open("./" + save_correlation_dir, "a")

        text_file.write('{linear_correl: ' + str(linear_correl) + '}\n')
        text_file.write('{oracle_correl: ' + str(oracle_correl) + '}\n')
        text_file.write('{merge_correl: ' + str(merge_correl) + '}\n')
        text_file.write('\n')
        text_file.close()

    def compute_rse(self, all_q_values_record, save_rse_dir):
        linear_rse = self.relative_square_error(all_q_values_record.get('output_q'),
                                                all_q_values_record.get('test_q'))

        oracle_rse = self.relative_square_error(all_q_values_record.get('oracle_q'),
                                                all_q_values_record.get('test_q'))

        merge_rse = self.relative_square_error(all_q_values_record.get('merge_q'),
                                               all_q_values_record.get('test_q'))

        text_file = open("./" + save_rse_dir, "a")

        text_file.write('{home_linear_rse: ' + str(linear_rse) + '}\n')
        text_file.write('{home_oracle_rse: ' + str(oracle_rse) + '}\n')
        text_file.write('{home_merge_rse: ' + str(merge_rse) + '}\n')
        text_file.write('\n')
        text_file.close()

    def compute_rae(self, all_q_values_record, save_rae_dir):
        linear_rae = self.relative_absolute_error(all_q_values_record.get('output_q'),
                                                  all_q_values_record.get('test_q'))

        oracle_rae = self.relative_absolute_error(all_q_values_record.get('oracle_q'),
                                                  all_q_values_record.get('test_q'))

        merge_rae = self.relative_absolute_error(all_q_values_record.get('merge_q'),
                                                 all_q_values_record.get('test_q'))

        text_file = open("./" + save_rae_dir, "a")

        text_file.write('{home_linear_rae: ' + str(linear_rae) + '}\n')
        text_file.write('{home_oracle_rae: ' + str(oracle_rae) + '}\n')
        text_file.write('{home_merge_rae: ' + str(merge_rae) + '}\n')
        text_file.write('\n')
        text_file.close()

    def compute_mse(self, all_q_values_record, save_mse_dir):
        linear_mse = self.mean_square_error(all_q_values_record.get('output_q'),
                                            all_q_values_record.get('test_q'))

        oracle_mse = self.mean_square_error(all_q_values_record.get('oracle_q'),
                                            all_q_values_record.get('test_q'))

        merge_mse = self.mean_square_error(all_q_values_record.get('merge_q'),
                                           all_q_values_record.get('test_q'))

        text_file = open("./" + save_mse_dir, "a")

        text_file.write('{home_linear_mse: ' + str(linear_mse) + '}\n')
        text_file.write('{home_oracle_mse: ' + str(oracle_mse) + '}\n')
        text_file.write('{home_merge_mse: ' + str(merge_mse) + '}\n')
        text_file.write('\n')
        text_file.close()

    def compute_mae(self, all_q_values_record, save_mae_dir):
        linear_mse = self.mean_abs_error(all_q_values_record.get('output_q'),
                                         all_q_values_record.get('test_q'))

        oracle_mse = self.mean_abs_error(all_q_values_record.get('oracle_q'),
                                         all_q_values_record.get('test_q'))

        merge_mse = self.mean_abs_error(all_q_values_record.get('merge_q'),
                                        all_q_values_record.get('test_q'))

        text_file = open("./" + save_mae_dir, "a")

        text_file.write('{home_linear_mse: ' + str(linear_mse) + '}\n')
        text_file.write('{home_oracle_mse: ' + str(oracle_mse) + '}\n')
        text_file.write('{home_merge_mse: ' + str(merge_mse) + '}\n')
        text_file.write('\n')
        text_file.close()

    def relative_square_error(self, test_qs, target_qs):
        sse = 0
        rse = 0
        test_qs = map(float, test_qs)
        target_qs = map(float, target_qs)
        tm = np.mean(target_qs)
        for index in range(0, len(test_qs)):
            sse += (test_qs[index] - target_qs[index]) ** 2
            rse += (tm - target_qs[index]) ** 2
        return sse / rse

    def relative_absolute_error(self, test_qs, target_qs):
        sae = 0
        rae = 0
        test_qs = map(float, test_qs)
        target_qs = map(float, target_qs)
        tm = np.mean(target_qs)
        for index in range(0, len(test_qs)):
            sae += abs(test_qs[index] - target_qs[index])
            rae += abs(tm - target_qs[index])
        return sae / rae

    def mean_square_error(self, test_qs, target_qs):
        sse = 0
        for index in range(0, len(test_qs)):
            sse += (float(test_qs[index]) - float(target_qs[index])) ** 2
        return sse / len(test_qs)

    def mean_abs_error(self, test_qs, target_qs):
        sse = 0
        for index in range(0, len(test_qs)):
            sse += abs(float(test_qs[index]) - float(target_qs[index]))
        return sse / len(test_qs)

    # def boosting_tree_print_event_values(self, save_path, read_game_number):
    #     merge_flag = '_merge'
    #     # read_game_number = 200
    #     print 'starting from {0}'.format(read_game_number)
    #     self.utree = pickle.load(open(save_path + 'pickle_Game_File_' + str(read_game_number) + '.p', 'rb'))
    #     print 'finishing read tree'
    #     game_directory = self.problem.games_directory
    #     # game_dir_all = os.listdir(game_directory)
    #     game_to_print_list = [1]
    #     for game_number in game_to_print_list:
    #         q_values = []
    #         merge_count = 0
    #         game_dir = 'game00{0:04}'.format(game_number)
    #         states = read_states(game_directory, game_dir)
    #         actions = read_actions(game_directory, game_dir)
    #         rewards = read_rewards(game_directory, game_dir)[0]
    #         qValues = read_qValue(game_directory, game_dir)
    #         training_information = read_train_info(game_directory, game_dir)
    #         assert states.shape[0] == actions.shape[0] and actions.shape[0] == rewards.shape[0] and rewards.shape[
    #                                                                                                     0] == \
    #                                                                                                 training_information.shape[
    #                                                                                                     0]
    #         event_number = len(states)
    #         for index in range(0, event_number):
    #             action_name = unicodedata.normalize('NFKD', actions[index]).encode('ascii', 'ignore').strip()
    #             action = self.problem.actions[action_name]
    #             currentObs = states[index]
    #             if index + 1 == event_number:
    #                 nextObs = states[index]
    #             else:
    #                 nextObs = states[index + 1]
    #             reward = rewards[index]
    #             qValue = qValues[index]
    #             q_end = 1 - qValue[0] - qValue[1]
    #             qValue = np.append(qValue, [q_end])
    #
    #             calibrate_name_str = unicodedata.normalize('NFKD', training_information[index]).encode('ascii',
    #                                                                                                    'ignore')
    #             calibrate_name_dict = ast.literal_eval(calibrate_name_str)
    #             home_identifier = int(calibrate_name_dict.get('home'))
    #             if action_name == 'goal':
    #                 nextObs = states[index]
    #             if reward < 0:
    #                 home_reward = 0
    #                 away_reward = 1
    #             else:
    #                 if reward > 0:
    #                     home_reward = 1
    #                     away_reward = 0
    #                 else:
    #                     home_reward = 0
    #                     away_reward = 0
    #
    #             Q_list, merge_count = self.get_Q_values_linear_tree(currentObs, nextObs, action, home_reward,
    #                                                                 away_reward, qValue,
    #                                                                 home_identifier, merge_flag, merge_count)
    #
    #             q_values.append(self.normalization_q(Q_list))
    #
    #         self.save_csv_q_values(q_values,
    #                                './boost_qsplit_values/Qvalues_game{0}_model_normal_r{1}{2}.csv'.format(
    #                                    game_number,
    #                                    read_game_number, merge_flag))
    #
    #         smooth_q_values = self.smoothing_q(q_values)
    #         self.save_csv_q_values(smooth_q_values,
    #                                './boost_qsplit_values/Qvalues_game{0}_model_normal_r{1}{2}_smooth.csv'.format(
    #                                    game_number,
    #                                    read_game_number, merge_flag))
    #         print merge_count

    def add_linear_regression(self):

        train_game_number = 200
        self.read_Utree(game_number=train_game_number, save_path=self.SAVE_PATH)

        # with tf.Session() as sess:
        self.utree.train_linear_regression_on_leaves(node=self.utree.root)

        print 'saving ' + self.SAVE_MODEL_TREE_PATH + 'Model_Tree_File_' + str(train_game_number) + '.p'
        pickle.dump(self.utree,
                    open(self.SAVE_MODEL_TREE_PATH + 'Model_Tree_File_' + str(train_game_number) + '.p', 'wb'))
        print 'finish saving ' + self.SAVE_MODEL_TREE_PATH + 'Model_Tree_File_' + str(train_game_number) + '.p'

    def feature_importance(self):
        self.read_Utree(game_number=100, save_path=self.SAVE_PATH)

        self.utree.feature_influence_dict = {'position': 0, 'velocity': 0, 'actions': 0}

        game_to_print_list = range(301, 401)
        for game_number in game_to_print_list:

            game_record = self.read_csv_game_record(
                self.problem.games_directory + 'record_moutaincar_transition_game{0}.csv'.format(int(game_number)))

            event_number = len(game_record)

            for index in range(0, event_number):
                transition = game_record[index]
                currentObs = transition.get('observation').split('$')
                nextObs = transition.get('newObservation').split('$')
                reward = float(transition.get('reward'))
                action = float(transition.get('action'))
                qValue = float(transition.get('qValue'))

                inst = C_UTree.Instance(-1, currentObs, action, nextObs, reward, qValue)
                self.utree.insertTestInstances(inst=inst, qValue=qValue)

        visit_count = self.utree.recursive_calculate_feature_importance(node=self.utree.root)


        print (self.utree.feature_influence_dict)

    def episode(self, game_number, timeout=int(100000.0), save_checkpoint_flag=1):
        """
        start to build the tree within an episode
        :param save_checkpoint_flag:
        :param timeout: no use here
        :return:
        """
        start_game = game_number
        self.utree.hard_code_flag = True
        if start_game > 0:
            self.read_Utree(game_number=start_game, save_path=self.SAVE_PATH)
        count = 0

        game_record = self.read_csv_game_record(
            self.problem.games_directory + 'record_moutaincar_transition_game{0}.csv'.format(int(game_number)))
        event_number = len(game_record)
        beginflag = True
        count += 1
        for index in range(0, event_number):

            if self.problem.isEpisodic:
                transition = game_record[index]
                currentObs = transition.get('observation').split('$')
                nextObs = transition.get('newObservation').split('$')
                reward = float(transition.get('reward'))
                action = float(transition.get('action'))
                qValue = float(transition.get('qValue'))

                if index == event_number - 1:  # game ending
                    print >> sys.stderr, '=============== update starts ==============='
                    # tracker.print_diff()
                    self.update(currentObs, nextObs, action, reward, qValue, value_iter=1, check_fringe=1,
                                beginflag=beginflag)
                    # tracker.print_diff()
                    print >> sys.stderr, '=============== update finished ===============\n'
                else:
                    self.update(currentObs, nextObs, action, reward, qValue,
                                beginflag=beginflag)
                    # else:
                    # currentObs = states[index]
                    # reward = rewards[index]
                    # self.getQ(currentObs, [], action, reward, home_identifier)
                beginflag = False

        if self.problem.isEpisodic:
            # print 'Game File ' + str(count)
            print '*** Writing Game File {0}***\n'.format(str(game_number + 1))
            self.utree.print_tree_structure(self.PRINT_TREE_PATH)
            if save_checkpoint_flag and (game_number + 1) % 1 == 0:
                pickle.dump(self.utree,
                            open(self.SAVE_PATH + 'pickle_Game_File_' + str(game_number + 1) + '.p', 'wb'))
                # self.utree.tocsvFile(self.TREE_PATH + 'Game_File_' + str(game_number + 1) + '.csv')
