import Agent_boost_Galen as Agent
import C_UTree_boost_Galen
import tensorflow as tf
import Problem_moutaincar
import sys

data_game_number = [600]
node_error_all_dict = {}


def distribute_instance(node, instance):
    if node.extra_Intance is not None:
        node.extra_Instance.append(instance)
    else:
        node.extra_Instance = []

    while node.nodeType != C_UTree_boost_Galen.NodeLeaf:
        child = node.applyInstanceDistinction(instance)
        node = node.children[child]
        distribute_instance(child, instance)

    return


def compute_split_error(node):
    dict_name = node.distinction.dimension_name

    square_error_list = []

    for child in node.children:
        for instance in child.extra_Instance():
            square_error_list.append((instance.qValue - child.qValues) ** 2)

    mse = sum(square_error_list) / len(square_error_list)

    if node_error_all_dict.get(dict_name) is not None:
        node_error_all_dict.update({dict_name: square_error_list})
    else:
        dict_square_error_list = node_error_all_dict.get(dict_name)
        dict_square_error_list += square_error_list


def compute_regression_error_upon_all_nodes():
    game_testing_record_dict = {}
    train_game_number = 200
    ice_hockey_problem = Problem_moutaincar.MoutainCar()
    CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=3000,
                                    check_fringe_freq=1200, is_episodic=0, training_mode='_linear_epoch_decay_lr')
    CUTreeAgent.read_Utree(game_number=train_game_number,
                           save_path='/Local-Scratch/UTree model/mountaincar/model_boost_linear_qsplit_noabs_save_linear_epoch_decay_lr/')

    index_number = 0
    root_node = CUTreeAgent.utree.root
    for game_number in data_game_number:

        game_record = CUTreeAgent.read_csv_game_record(
            CUTreeAgent.problem.games_directory + 'record_moutaincar_transition_game{0}.csv'.format(int(game_number)))
        event_number = len(game_record)
        beginflag = True
        for index in range(0, event_number):
            transition = game_record[index]
            currentObs = transition.get('observation').split('$')
            nextObs = transition.get('newObservation').split('$')
            reward = float(transition.get('reward'))
            action = float(transition.get('action'))
            qValue = float(transition.get('qValue'))

            instance = C_UTree_boost_Galen.Instance(None, currentObs, action, nextObs, reward, qValue)

            distribute_instance(root_node, instance)

    compute_split_error(root_node)


def compute_dict_mse():
    for dict_name in node_error_all_dict.keys():
        se_list = node_error_all_dict.get(dict_name)

        print '{0}:{1}'.format(dict_name, float(sum(se_list)) / len(se_list))
