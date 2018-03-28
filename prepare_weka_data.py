import csv
import copy
import brain
import numpy as np


def read_csv_game_record(csv_dir):
    dict_all = []
    with open(csv_dir, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dict_all.append(row)
    return dict_all


def save_csv_record(csv_dir, data):
    with open(csv_dir, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in data:
            writer.writerow(val)


def generate_data():
    data_list = []
    for game_number in range(900, 1000):
        game_record = read_csv_game_record(
            './save_all_transition/record_moutaincar_transition_game{0}.csv'.format(int(game_number)))

        event_number = len(game_record)

        for index in range(0, event_number):
            transition = game_record[index]
            currentObs = transition.get('observation').split('$')
            nextObs = transition.get('newObservation').split('$')
            reward = float(transition.get('reward'))
            action = float(transition.get('action'))
            qValue = float(transition.get('qValue'))
            data_row = copy.copy(currentObs)
            if data_row[1] == '0.0':
                data_row[1] = '0.001'
            data_row.append(action)
            data_row.append(qValue)
            data_list.append(data_row)
    save_csv_record('moutaincar_testing_data.csv', data_list)


def read_csv_game_record_list(csv_dir):
    dict_all = []
    with open(csv_dir, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            dict_all.append(row)
    return dict_all


def generate_training_record_data():
    dict_all = read_csv_game_record_list('./record_training_observations.csv')

    agent = brain.Agent()
    agent.model.read_model()

    for training_info_record in dict_all:
        observation = map(float, training_info_record[:-1])
        action = int(training_info_record[-1])

        _, qValues = agent.model.get_action(observation, 0)

        training_info_record.append(str(qValues[action]))

    save_csv_record('mountaincar_dataset.csv', dict_all)


if __name__ == "__main__":
    # generate_data()
    generate_training_record_data()