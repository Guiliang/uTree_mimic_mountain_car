import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(color_codes=True)
sns.set(font_scale=2)


def read_result_csv(csv_dir):
    home_linear_correl = []
    home_oracle_correl = []
    home_merge_correl = []

    away_linear_correl = []
    away_oracle_correl = []
    away_merge_correl = []

    end_linear_correl = []
    end_oracle_correl = []
    end_merge_correl = []

    read_counter = 0

    with open(csv_dir, 'rb') as file:
        rows = file.readlines()
        for row in rows:
            if row == '\n':
                continue
            else:
                # print row
                row = row.replace('{', '').replace('}', '').replace('\n', '')
                correlations = row.split(',')
                read_counter += 1
                if read_counter % 3 == 1:
                    home_linear_correl.append(float(correlations[0].strip().split(':')[1]))
                elif read_counter % 3 == 2:
                    home_oracle_correl.append(float(correlations[0].strip().split(':')[1]))
                elif read_counter % 3 == 0:
                    home_merge_correl.append(float(correlations[0].strip().split(':')[1]))
    return home_linear_correl, home_oracle_correl, home_merge_correl


def dealing_plot_data(correl, transition_length_list, root=False):
    length = len(correl)
    game_numbers = range(1, length + 1, 1)
    transition_numbers = [sum(transition_length_list[:game_number - 1]) + transition_length_list[game_number - 1] for
                          game_number in game_numbers]
    # game_numbers = [math.log(game_number,2)for game_number in game_numbers]

    x_plot_list = []
    y_plot_list = []
    for transition_number_index in range(0, len(transition_numbers)):
        if transition_numbers[transition_number_index] <= 30000:

            x_plot_list.append(transition_numbers[transition_number_index])
            if root:
                y_plot_list.append(np.sqrt(correl[transition_number_index]))
            else:
                y_plot_list.append(correl[transition_number_index])
    x_plot_list.append(30000)
    y_plot_list.append(y_plot_list[-1])
    return x_plot_list, y_plot_list


def plot_tree_result(x_plot_list, y_plot_list):
    x_plot_list = [float(number) / 1000 for number in x_plot_list]
    plt.plot(x_plot_list, y_plot_list, label='correlation')
    # plt.plot(game_numbers, away_correl, label='correlation away')
    # plt.plot(game_numbers, end_correl)
    plt.xlabel('game_numbers')
    plt.ylabel('correlated coefficient')
    plt.legend(loc='best')
    plt.show()


def plot_tree_shadow_result(x_plot_array, y_plot_array, name):
    x_plot_array = [float(number) / 1000 for number in x_plot_array]
    plt.figure(figsize=(6, 6.5))
    ax = sns.tsplot(y_plot_array, x_plot_array, condition=name, legend=True)
    ax.ticklabel_format(axis='x', style='sci')
    plt.xlabel('Transition Numbers (by thousands)')
    # plt.ylabel(name)
    # plt.legend(loc='best')
    plt.show()


def record_transition_length():
    game_number = range(1, 201)
    transition_length_list = []

    for i in game_number:
        game_file_dir = '../save_all_transition/record_moutaincar_transition_game{0}.csv'.format(str(i))

        with open(game_file_dir, 'rb') as file:
            rows = file.readlines()
            transition_length_list.append(len(rows) - 1)

    return transition_length_list


def plot_simple_graph():
    home_linear_correl, home_oracle_correl, home_merge_correl = read_result_csv(
        csv_dir='./result/result-correlation-all-linear-epoch-decay-lr-st0-500')
    transition_length_list = record_transition_length()
    x_plot_list, y_plot_list = dealing_plot_data(home_linear_correl, transition_length_list)
    plot_tree_result(x_plot_list, y_plot_list)


def plot_mse_shadow_graph():
    csv_dir_list = ['./result/result-mae-all-linear-epoch-decay-lr-st0-600',
                    './result/result-mae-all-linear-epoch-decay-lr-st0-700',
                    './result/result-mae-all-linear-epoch-decay-lr-st0-800',
                    './result/result-mae-all-linear-epoch-decay-lr-st0-900',
                    './result/result-mae-all-linear-epoch-decay-lr-st0-1000'
                    ]
    x_plot_array = None
    y_plot_array = None
    transition_length_list = record_transition_length()
    for csv_dir in csv_dir_list:
        home_linear_correl, home_oracle_correl, home_merge_correl = read_result_csv(
            csv_dir=csv_dir)
        x_plot_list, y_plot_list = dealing_plot_data(home_merge_correl, transition_length_list)
        x_plot_array = np.asarray(x_plot_list)
        if y_plot_array is None:
            y_plot_array = np.asarray([y_plot_list])
        else:
            y_plot_array = np.concatenate((y_plot_array, np.asarray([y_plot_list])), axis=0)

    plot_tree_shadow_result(x_plot_array, y_plot_array, "MAE")


def plot_correl_shadow_graph():
    csv_dir_list = ['./result/result-correlation-all-linear-epoch-decay-lr-st0-600',
                    './result/result-correlation-all-linear-epoch-decay-lr-st0-700',
                    './result/result-correlation-all-linear-epoch-decay-lr-st0-800',
                    './result/result-correlation-all-linear-epoch-decay-lr-st0-900',
                    './result/result-correlation-all-linear-epoch-decay-lr-st0-1000']
    x_plot_array = None
    y_plot_array = None
    transition_length_list = record_transition_length()
    for csv_dir in csv_dir_list:
        home_linear_correl, home_oracle_correl, home_merge_correl = read_result_csv(
            csv_dir=csv_dir)
        x_plot_list, y_plot_list = dealing_plot_data(home_linear_correl, transition_length_list)
        x_plot_array = np.asarray(x_plot_list)
        if y_plot_array is None:
            y_plot_array = np.asarray([y_plot_list])
        else:
            y_plot_array = np.concatenate((y_plot_array, np.asarray([y_plot_list])), axis=0)

    plot_tree_shadow_result(x_plot_array, y_plot_array, "Correlation")


if __name__ == "__main__":
    # plot_mse_shadow_graph()
    plot_correl_shadow_graph()
    # plot_simple_graph()
