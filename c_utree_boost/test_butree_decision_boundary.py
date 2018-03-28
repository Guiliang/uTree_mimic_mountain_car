import Agent_boost_Galen as Agent
import C_UTree_boost_Galen
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import linear_regression
import Problem_moutaincar
import matplotlib as mpl

ACTION_LIST = [0, 1, 2]
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size


def save_decision_csv(decision_all):
    with open('./decision_all.csv', 'wb') as file:
        for decision_positions in decision_all:
            write_text = ''
            for decision_position in decision_positions:
                write_text += str(decision_position) + ','
                write_text = write_text[:-1]
            file.write(write_text)
            file.write('\n')


def generate_similar_lmu_tree_one_way_decision(input_all):
    decision_all = []
    train_game_number = 200
    ice_hockey_problem = Problem_moutaincar.MoutainCar()
    CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=3000,
                                    check_fringe_freq=1200, is_episodic=0, training_mode='_linear_epoch_decay_lr')
    CUTreeAgent.read_Utree(game_number=train_game_number,
                           save_path='/Local-Scratch/UTree model/mountaincar/model_boost_linear_qsplit_noabs_save_linear_epoch_decay_lr/')

    for index in range(0, len(input_all)):
        input_observation = input_all[index]

        action = None
        top_actions = []
        Q_value = 0
        decision_action_all = []
        for action_test in ACTION_LIST:
            inst = C_UTree_boost_Galen.Instance(-1, input_observation, action_test, input_observation, None,
                                                None)  # leaf is located by the current observation
            node = CUTreeAgent.utree.getAbsInstanceLeaf(inst)
            Q_value = 0
            min_mse = 999
            mse_criterion = 0.2

            for instance in node.instances:
                instance_observation = instance.currentObs
                mse = compute_mse(np.asarray(input_observation), np.asarray(instance_observation))
                # mse = ((np.asarray(input_observation) - np.asarray(instance_observation)) ** 2).mean()
                if mse < min_mse:
                    min_mse = mse
                    Q_value = instance.qValue
                    action = action_test
                if mse < mse_criterion:
                    top_actions.append(action_test)

                    # if len(top_actions) >= 3:
                    #     done = True
                    #     a = np.asarray(top_actions)
                    #     counts = np.bincount(a)
                    #     action_most = np.argmax(counts)
                    #     # if action != action_most:
                    #     # print 'catch you'
                    #     action = action_most

            decision_action_all.append(Q_value)
        decision_all.append(decision_action_all)
    return decision_all


def generate_similar_lmu_tree_two_way_decision(input_all, action):
    column_length = len(input_all[0])
    row_length = len(input_all)
    decision_all = np.full((row_length, column_length), np.inf)
    train_game_number = 200
    ice_hockey_problem = Problem_moutaincar.MoutainCar()
    CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=3000,
                                    check_fringe_freq=1200, is_episodic=0, training_mode='_linear_epoch_decay_lr')
    CUTreeAgent.read_Utree(game_number=train_game_number,
                           save_path='/Local-Scratch/UTree model/mountaincar/model_boost_linear_qsplit_noabs_save_linear_epoch_decay_lr/')

    for input_positions_index in range(0, len(input_all)):
        input_positions = input_all[input_positions_index]

        for input_observation_index in range(0, len(input_positions)):
            input_observation = input_positions[input_observation_index]

            min_mse = 999
            mse_criterion = 0.2
            # action = None
            top_actions = []
            Q_value = 0

            for action_test in [action]:
                inst = C_UTree_boost_Galen.Instance(-1, input_observation, action_test, input_observation, None,
                                                    None)  # leaf is located by the current observation
                node = CUTreeAgent.utree.getAbsInstanceLeaf(inst)

                for instance in node.instances:
                    instance_observation = instance.currentObs
                    mse = compute_mse(np.asarray(input_observation), np.asarray(instance_observation))
                    # mse = ((np.asarray(input_observation) - np.asarray(instance_observation)) ** 2).mean()
                    if mse < min_mse:
                        min_mse = mse
                        Q_value = instance.qValue
                        # action = action_test
                    if mse < mse_criterion:
                        top_actions.append(action_test)

                        # if len(top_actions) >= 3:
                        #     done = True
                        #     a = np.asarray(top_actions)
                        #     counts = np.bincount(a)
                        #     action_most = np.argmax(counts)
                        #     # if action != action_most:
                        #     # print 'catch you'
                        #     action = action_most

            decision_all[input_positions_index, input_observation_index] = Q_value

    return decision_all


def compute_mse(input_observation, instance_observation, scale_number=12.85):
    input_observation[1] = input_observation[1] * scale_number
    instance_observation[1] = instance_observation[1] * scale_number

    mse = ((np.asarray(input_observation) - np.asarray(instance_observation)) ** 2).mean()

    return mse


def generate_linear_b_u_tree_two_way_decision(input_all, action):
    game_testing_record_dict = {}
    train_game_number = 200
    ice_hockey_problem = Problem_moutaincar.MoutainCar()
    CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=3000,
                                    check_fringe_freq=1200, is_episodic=0, training_mode='_linear_epoch_decay_lr')
    CUTreeAgent.read_Utree(game_number=train_game_number,
                           save_path='/Local-Scratch/UTree model/mountaincar/model_boost_linear_qsplit_noabs_save_linear_epoch_decay_lr/')

    index_number = 0

    for input_positions in input_all:

        for input in input_positions:

            inst_aleft = C_UTree_boost_Galen.Instance(-1, input, 0, input, None,
                                                      None)  # next observation is not important
            inst_amiddle = C_UTree_boost_Galen.Instance(-1, input, 1, input, None, None)
            inst_aright = C_UTree_boost_Galen.Instance(-1, input, 2, input, None, None)
            node_aleft = CUTreeAgent.utree.getAbsInstanceLeaf(inst_aleft)
            node_amiddle = CUTreeAgent.utree.getAbsInstanceLeaf(inst_amiddle)
            node_aright = CUTreeAgent.utree.getAbsInstanceLeaf(inst_aright)

            if game_testing_record_dict.get(node_aleft) is None:
                game_testing_record_dict.update({node_aleft: np.array([[input, 0, index_number]])})
            else:
                node_record = game_testing_record_dict.get(node_aleft)
                node_record = np.concatenate((node_record, [[input, 0, index_number]]), axis=0)
                game_testing_record_dict.update({node_aleft: node_record})

            if game_testing_record_dict.get(node_amiddle) is None:
                game_testing_record_dict.update({node_amiddle: np.array([[input, 1, index_number]])})
            else:
                node_record = game_testing_record_dict.get(node_amiddle)
                node_record = np.concatenate((node_record, [[input, 1, index_number]]), axis=0)
                game_testing_record_dict.update({node_amiddle: node_record})

            if game_testing_record_dict.get(node_aright) is None:
                game_testing_record_dict.update({node_aright: np.array([[input, 2, index_number]])})
            else:
                node_record = game_testing_record_dict.get(node_aright)
                node_record = np.concatenate((node_record, [[input, 2, index_number]]), axis=0)
                game_testing_record_dict.update({node_aright: node_record})

            index_number += 1

    index_qvalue_record = {}

    for node in game_testing_record_dict.keys():
        node_record = game_testing_record_dict.get(node)
        currentObs_node = node_record[:, 0]
        actions = node_record[:, 1]
        index_numbers = node_record[:, 2]

        # for i in range(0, len(index_numbers)):
        #     min_mse = 999999
        #
        #     currentObs = currentObs_node[i]
        #     for instance in node.instances:
        #         instance_observation = instance.currentObs
        #         mse = ((np.asarray(currentObs) - np.asarray(instance_observation)) ** 2).mean()
        #         if mse < min_mse:
        #             min_mse = mse
        #             Q_value = instance.qValue
        #
        #     if index_qvalue_record.get(index_numbers[i]) is not None:
        #         index_record_dict = index_qvalue_record.get(index_numbers[i])
        #         index_record_dict.update({actions[i]: Q_value})
        #     else:
        #         index_qvalue_record.update({index_numbers[i]: {actions[i]: Q_value}})

        sess = tf.Session()
        LR = linear_regression.LinearRegression()
        LR.read_weights(weights=node.weight, bias=node.bias)
        LR.readout_linear_regression_model()
        sess.run(LR.init)
        qValues_output = sess.run(LR.pred, feed_dict={LR.X: currentObs_node.tolist()})

        for i in range(0, len(index_numbers)):
            if index_qvalue_record.get(index_numbers[i]) is not None:
                index_record_dict = index_qvalue_record.get(index_numbers[i])
                index_record_dict.update({actions[i]: qValues_output[i]})
            else:
                index_qvalue_record.update({index_numbers[i]: {actions[i]: qValues_output[i]}})

    column_length = len(input_all[0])
    row_length = len(input_all)
    decision_all = np.full((row_length, column_length), np.inf)

    for i in index_qvalue_record:
        index_record_dict = index_qvalue_record.get(i)
        q_left = index_record_dict.get(0)
        q_middle = index_record_dict.get(1)
        q_right = index_record_dict.get(2)
        qValues = [q_left, q_middle, q_right]

        max_action = qValues.index(max(qValues))

        row_number = i / column_length
        column_number = i % column_length

        decision_all[row_number, column_number] = qValues[action]

    return decision_all


def generate_linear_b_u_tree_one_way_decision(input_all):
    game_testing_record_dict = {}
    train_game_number = 200
    ice_hockey_problem = Problem_moutaincar.MoutainCar()
    CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=3000,
                                    check_fringe_freq=1200, is_episodic=0, training_mode='_linear_epoch_decay_lr')
    CUTreeAgent.read_Utree(game_number=train_game_number,
                           save_path='/Local-Scratch/UTree model/mountaincar/model_boost_linear_qsplit_noabs_save_linear_epoch_decay_lr/')

    index_number = 0

    for input in input_all:

        # for input in input_positions:

        inst_aleft = C_UTree_boost_Galen.Instance(-1, input, 0, input, None,
                                                  None)  # next observation is not important
        inst_amiddle = C_UTree_boost_Galen.Instance(-1, input, 1, input, None, None)
        inst_aright = C_UTree_boost_Galen.Instance(-1, input, 2, input, None, None)
        node_aleft = CUTreeAgent.utree.getAbsInstanceLeaf(inst_aleft)
        node_amiddle = CUTreeAgent.utree.getAbsInstanceLeaf(inst_amiddle)
        node_aright = CUTreeAgent.utree.getAbsInstanceLeaf(inst_aright)

        if game_testing_record_dict.get(node_aleft) is None:
            game_testing_record_dict.update({node_aleft: np.array([[input, 0, index_number]])})
        else:
            node_record = game_testing_record_dict.get(node_aleft)
            node_record = np.concatenate((node_record, [[input, 0, index_number]]), axis=0)
            game_testing_record_dict.update({node_aleft: node_record})

        if game_testing_record_dict.get(node_amiddle) is None:
            game_testing_record_dict.update({node_amiddle: np.array([[input, 1, index_number]])})
        else:
            node_record = game_testing_record_dict.get(node_amiddle)
            node_record = np.concatenate((node_record, [[input, 1, index_number]]), axis=0)
            game_testing_record_dict.update({node_amiddle: node_record})

        if game_testing_record_dict.get(node_aright) is None:
            game_testing_record_dict.update({node_aright: np.array([[input, 2, index_number]])})
        else:
            node_record = game_testing_record_dict.get(node_aright)
            node_record = np.concatenate((node_record, [[input, 2, index_number]]), axis=0)
            game_testing_record_dict.update({node_aright: node_record})

        index_number += 1

    index_qvalue_record = {}

    for node in game_testing_record_dict.keys():
        node_record = game_testing_record_dict.get(node)
        currentObs_node = node_record[:, 0]
        actions = node_record[:, 1]
        index_numbers = node_record[:, 2]

        # for i in range(0, len(index_numbers)):
        #     min_mse = 999999
        #
        #     currentObs = currentObs_node[i]
        #     for instance in node.instances:
        #         instance_observation = instance.currentObs
        #         mse = ((np.asarray(currentObs) - np.asarray(instance_observation)) ** 2).mean()
        #         if mse < min_mse:
        #             min_mse = mse
        #             Q_value = instance.qValue
        #
        #     if index_qvalue_record.get(index_numbers[i]) is not None:
        #         index_record_dict = index_qvalue_record.get(index_numbers[i])
        #         index_record_dict.update({actions[i]: Q_value})
        #     else:
        #         index_qvalue_record.update({index_numbers[i]: {actions[i]: Q_value}})

        sess = tf.Session()
        LR = linear_regression.LinearRegression()
        LR.read_weights(weights=node.weight, bias=node.bias)
        LR.readout_linear_regression_model()
        sess.run(LR.init)
        qValues_output = sess.run(LR.pred, feed_dict={LR.X: currentObs_node.tolist()})

        for i in range(0, len(index_numbers)):
            if index_qvalue_record.get(index_numbers[i]) is not None:
                index_record_dict = index_qvalue_record.get(index_numbers[i])
                index_record_dict.update({actions[i]: qValues_output[i]})
            else:
                index_qvalue_record.update({index_numbers[i]: {actions[i]: qValues_output[i]}})

    length = len(input_all)
    decision_all = []

    for i in index_qvalue_record:
        index_record_dict = index_qvalue_record.get(i)
        q_left = index_record_dict.get(0)
        q_middle = index_record_dict.get(1)
        q_right = index_record_dict.get(2)
        qValues = [q_left[0], q_middle[0], q_right[0]]

        max_action = qValues.index(max(qValues))

        decision_all.append(qValues)

    return decision_all


def generate_two_way_data():
    position_interval = [-1.2, 0.6]
    velocity_interval = [-0.07, 0.07]
    position_all = np.arange(position_interval[0], position_interval[1], 0.01)
    velocity_all = np.arange(velocity_interval[0], velocity_interval[1], 0.001)

    input_all = []

    for i in range(len(velocity_all) - 1, -1, -1):
        input_position_list = []
        for j in range(0, len(position_all)):
            input_position_list.append([position_all[j], velocity_all[i]])
        input_all.append(input_position_list)

    return input_all


def generate_one_way_data(feature_target):
    position_interval = [-1.2, 0.6]
    mean_position = (position_interval[0] + position_interval[1]) / 2
    velocity_interval = [-0.07, 0.07]
    mean_velocity = (velocity_interval[0] + velocity_interval[1]) / 2

    position_all = np.arange(position_interval[0], position_interval[1], 0.05)
    velocity_all = np.arange(velocity_interval[0], velocity_interval[1], 0.005)

    if feature_target == 'velocity':
        input_velocity_list = []
        for i in range(0, len(velocity_all)):
            input_velocity_list.append([mean_position, velocity_all[i]])
        return input_velocity_list
    elif feature_target == 'position':
        input_position_list = []
        for i in range(0, len(position_all)):
            input_position_list.append([position_all[i], mean_velocity])
        return input_position_list


def visualize_two_way_decision(decision_all, action):
    plt.figure(figsize=(4.5, 4.5))
    action_name = {0: 'push left', 1: 'no push', 2: 'push right'}
    # plt.figure(figsize=(15, 6))
    sns.set(font_scale=1.6)
    ax = sns.heatmap(decision_all, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r")
    # vmin=vmin_set,
    # vmax=vmax_set)
    plt.xlabel('Position', fontsize=18)
    plt.ylabel('Velocity', fontsize=18)
    plt.title('Action={0}'.format(action_name.get(action)))
    plt.savefig('/home/gla68/Desktop/U-Tree/paper image/mc-two-way-{0}.png'.format(action))
    plt.show()


def plot_one_way_decision(input_all, decision_all, feature_target):
    plt.figure(figsize=(5, 6))
    # sns.set(font_scale=1.6)
    action_name = {0: 'push left', 1: 'no push', 2: 'push right'}
    # problem_color_dict = {0: 'g', 1: 'b', 2: 'o'}
    problem_line_style_dict = {0: '-', 1: '--', 2: '-.'}

    for decision_action_index in range(0, decision_all.shape[1]):
        plt.plot(input_all, decision_all[:, decision_action_index], linewidth=4,
                 label=action_name.get(decision_action_index),
                 linestyle=problem_line_style_dict.get(decision_action_index))
    # vmin=vmin_set,
    # vmax=vmax_set)
    plt.xlabel(feature_target, fontsize=20)
    plt.ylabel('Q value', fontsize=20)
    plt.legend(loc='best', prop={'size': 18})
    plt.savefig('/home/gla68/Desktop/U-Tree/paper image/mc-one-way-{0}.png'.format(feature_target))
    plt.show()


def two_way_partial_dependency_graph():
    input_all = generate_two_way_data()
    # decision_all = generate_similar_lmu_tree_two_way_decision(input_all)
    for action in ACTION_LIST:
        decision_all = generate_similar_lmu_tree_two_way_decision(input_all, action=action)
        # decision_all = generate_linear_b_u_tree_two_way_decision(input_all, action=action)
        # save_decision_csv(decision_all=decision_all)
        visualize_two_way_decision(decision_all, action=action)


def one_way_partial_dependency_graph():
    feature_target = 'position'
    input_all = generate_one_way_data(feature_target)
    # temp = np.asarray(input_all)[:, 1]
    # decision_all = generate_similar_lmu_tree_one_way_decision(input_all)
    decision_all = generate_linear_b_u_tree_one_way_decision(input_all)
    plot_one_way_decision(np.asarray(input_all)[:, 0], np.asarray(decision_all), feature_target)


if __name__ == "__main__":
    two_way_partial_dependency_graph()
    # one_way_partial_dependency_graph()
