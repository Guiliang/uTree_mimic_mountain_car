import brain
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

environment = gym.make('MountainCar-v0')
learning_rate = 1e-3
dqn_model = brain.Model(environment, learning_rate, None)


def save_decision_csv(decision_all):
    with open('./decision_all.csv', 'wb') as file:
        for decision_positions in decision_all:
            write_text = ''
            for decision_position in decision_positions:
                write_text += str(decision_position) + ','
                write_text = write_text[:-1]
            file.write(write_text)
            file.write('\n')


def generate_model_decision(input_all):
    dqn_model.read_model()
    decision_all = []

    for input_positions in input_all:

        qValues_list = dqn_model.get_qvalues(input_positions)

        decision_list = []

        for qValues in qValues_list:
            qValues = qValues.tolist()
            max_index = qValues.index(max(qValues))
            decision_list.append(max(qValues))

        decision_all.append(decision_list)
    return decision_all


def generate_data():
    position_interval = [-1.2, 0.6]
    velocity_interval = [-0.07, 0.07]
    position_all = np.arange(position_interval[0], position_interval[1], 0.01)
    velocity_all = np.arange(velocity_interval[0], velocity_interval[1], 0.01)

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
    decision_all = generate_model_decision(input_all)
    # save_decision_csv(decision_all=decision_all)
    visualize_decision(decision_all)
