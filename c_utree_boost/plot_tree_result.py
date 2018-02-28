import math
import matplotlib.pyplot as plt


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


def plot_tree_correlation(correl, transition_length_list):
    length = len(correl)
    game_numbers = range(1, length + 1, 1)
    transition_numbers = [sum(transition_length_list[:game_number-1]) + transition_length_list[game_number-1] for game_number in game_numbers]
    # game_numbers = [math.log(game_number,2)for game_number in game_numbers]
    x_plot_list = []
    y_plot_list = []
    for transition_number_index in range(0, len(transition_numbers)):
        if transition_numbers[transition_number_index] < 20000:
            x_plot_list.append(transition_numbers[transition_number_index])
            y_plot_list.append(correl[transition_number_index])

    plt.plot(x_plot_list, y_plot_list, label='correlation')
    # plt.plot(game_numbers, away_correl, label='correlation away')
    # plt.plot(game_numbers, end_correl)
    plt.xlabel('game_numbers')
    plt.ylabel('correlated coefficient')
    plt.legend(loc='best')
    plt.show()


def record_transition_length():
    game_number = range(1, 200)
    transition_length_list = []

    for i in game_number:
        game_file_dir = '../save_all_transition/record_moutaincar_transition_game{0}.csv'.format(str(i))

        with open(game_file_dir, 'rb') as file:
            rows = file.readlines()
            transition_length_list.append(len(rows)-1)

    return transition_length_list


if __name__ == "__main__":
    home_linear_correl, home_oracle_correl, home_merge_correl = read_result_csv(
        csv_dir='./result-mse-all-linear-epoch-decay-lr-st0-900')
    transition_length_list = record_transition_length()
    plot_tree_correlation(home_linear_correl, transition_length_list)
    # plot_tree_correlation(home_oracle_correl)
    # plot_tree_correlation(home_merge_correl)
