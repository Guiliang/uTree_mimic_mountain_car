import Agent_boost_Galen as Agent
import C_UTree_boost_Galen
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import linear_regression
import Problem_moutaincar

ACTION_LIST = [0, 1, 2]


def save_decision_csv(decision_all):
    with open('./decision_all.csv', 'wb') as file:
        for decision_positions in decision_all:
            write_text = ''
            for decision_position in decision_positions:
                write_text += str(decision_position) + ','
                write_text = write_text[:-1]
            file.write(write_text)
            file.write('\n')


def generate_similar_b_u_tree_decision(input_all):
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
            action = None
            top_actions = []
            Q_value = 0

            for action_test in ACTION_LIST:
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

            decision_all[input_positions_index, input_observation_index] = Q_value

    return decision_all


def compute_mse(input_observation, instance_observation, scale_number=12.85):
    input_observation[1] = input_observation[1] * scale_number
    instance_observation[1] = instance_observation[1] * scale_number

    mse = ((np.asarray(input_observation) - np.asarray(instance_observation)) ** 2).mean()

    return mse


def generate_linear_b_u_tree_decision(input_all):
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

        decision_all[row_number, column_number] = max_action

    return decision_all


def generate_data():
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


def visualize_decision(decision_all):
    # plt.figure(figsize=(15, 6))
    sns.set(font_scale=1.6)
    ax = sns.heatmap(decision_all, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r")
    # vmin=vmin_set,
    # vmax=vmax_set)
    plt.xlabel('Position', fontsize=18)
    plt.ylabel('Velocity', fontsize=18)
    plt.show()


if __name__ == "__main__":
    input_all = generate_data()
    decision_all = generate_similar_b_u_tree_decision(input_all)
    # save_decision_csv(decision_all=decision_all)
    visualize_decision(decision_all)
