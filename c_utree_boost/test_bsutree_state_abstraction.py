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


def generate_similar_b_u_tree_cluster(input_all):
    decision_region = [-10, -15, -20, -25, -30, -35]

    column_length = len(input_all[0])
    row_length = len(input_all)
    decision_all = np.full((row_length, column_length), np.inf)
    train_game_number = 10
    ice_hockey_problem = Problem_moutaincar.MoutainCar()
    CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=3000,
                                    check_fringe_freq=1200, is_episodic=0, training_mode='_linear_epoch_decay_lr')
    CUTreeAgent.read_Utree(game_number=train_game_number,
                           save_path='/Local-Scratch/UTree model/mountaincar/model_boost_linear_qsplit_noabs_save_linear_epoch_decay_lr/')

    for input_positions_index in range(0, len(input_all)):
        input_positions = input_all[input_positions_index]

        for input_observation_index in range(0, len(input_positions)):
            input_observation = input_positions[input_observation_index]

            # for action_test in ACTION_LIST[0]:
            inst = C_UTree_boost_Galen.Instance(-1, input_observation, ACTION_LIST[2], input_observation, None,
                                                None)  # leaf is located by the current observation
            node = CUTreeAgent.utree.getAbsInstanceLeaf(inst)

            type = len(decision_region)
            for decision_cri in decision_region:
                if node.qValues[ACTION_LIST[2]] > decision_cri:
                    type = decision_region.index(decision_cri)
                    break

            decision_all[input_positions_index, input_observation_index] = type

    return decision_all


def compute_mse(input_observation, instance_observation, scale_number=12.85):
    input_observation[1] = input_observation[1] * scale_number
    instance_observation[1] = instance_observation[1] * scale_number

    mse = ((np.asarray(input_observation) - np.asarray(instance_observation)) ** 2).mean()

    return mse


def generate_data():
    position_interval = [-1.2, 0.6]
    velocity_interval = [-0.07, 0.07]
    position_all = np.arange(position_interval[0], position_interval[1], 0.001)
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
    decision_all = generate_similar_b_u_tree_cluster(input_all)
    # save_decision_csv(decision_all=decision_all)
    visualize_decision(decision_all)
