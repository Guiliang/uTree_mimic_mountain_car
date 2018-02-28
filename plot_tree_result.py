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


def plot_tree_correlation(home_correl):
    length = len(home_correl)
    game_numbers = range(1, length + 1, 1)
    # game_numbers = [math.log(game_number,2)for game_number in game_numbers]
    plt.plot(game_numbers, home_correl, label='correlation')
    # plt.plot(game_numbers, away_correl, label='correlation away')
    # plt.plot(game_numbers, end_correl)
    plt.xlabel('game_numbers')
    plt.ylabel('correlated coefficient')
    plt.legend(loc = 'best')
    plt.show()


if __name__ == "__main__":
    home_linear_correl, home_oracle_correl, home_merge_correl = read_result_csv(
        csv_dir='./bak-result-correlation-all-linear-epoch-decay-lr-st0-900')
    plot_tree_correlation(home_linear_correl)
    plot_tree_correlation(home_oracle_correl)
    plot_tree_correlation(home_merge_correl)
