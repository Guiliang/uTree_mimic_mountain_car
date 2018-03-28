import optparse
import Problem_moutaincar
import pickle
import Agent_boost_Galen as Agent

optparser = optparse.OptionParser()
optparser.add_option("-m", "--max_node_hist", dest="MAX_NODE_HIST", default=3000,
                     help="max number of instance in every node (default = 10000)")
optparser.add_option("-c", "--check_fringe_freq", dest="CHECK_FRINGE_FREQ", default=1200,
                     help="check fringe frequent (default = 100)")
optparser.add_option("-d", "--directory_of_games", dest="GAME_DIRECTORY", default="",
                     help="games dir of all the games")
optparser.add_option("-g", "--game number to test", dest="GAME_NUMBER", default=100,
                     help="which game to test")
optparser.add_option("-a", "--result correlation dir", dest="SAVE_CORRELATION_DIR", default=None,
                     help="the dir correlation result")
optparser.add_option("-j", "--result relative absolute error dir", dest="SAVE_RAE_DIR", default=None,
                     help="the dir relative absolute error result")
optparser.add_option("-i", "--result relative square error dir", dest="SAVE_RSE_DIR", default=None,
                     help="the dir relative square error result")
optparser.add_option("-b", "--result mean square error dir", dest="SAVE_MSE_DIR", default=None,
                     help="the dir mean square error result")
optparser.add_option("-f", "--result mean absolute error dir", dest="SAVE_MAE_DIR", default=None,
                     help="the dir mean absolute error result")
optparser.add_option("-e", "--training mode", dest="TRAINING_MODE", default='_linear_epoch_decay_lr',
                     help="training mode")

opts = optparser.parse_args()[0]


def recursive_find_path(node, target_idx):
    current_node_idx = node.idx

    if current_node_idx == target_idx:
        return True, "Q value = {0}".format(node.qValues)
    else:
        for c_index in range(0, len(node.children)):
            child = node.children[c_index]
            find_flag, path = recursive_find_path(child, target_idx)

            if find_flag:
                if c_index == 0:
                    mark = '<'
                else:
                    mark = '>'
                if node.distinction.continuous_divide_value is None:
                    return True, "{0} = {1}, {2}".format(str(node.distinction.dimension_name),
                                                         str(c_index), path, mark)
                else:
                    return True, "{0} {3} {1}, {2}".format(str(node.distinction.dimension_name),
                                                           str(node.distinction.continuous_divide_value), path, mark)
        return False, ''


def find_idx_path(idx):
    mountaincar = Problem_moutaincar.MoutainCar()
    CUTreeAgent = Agent.CUTreeAgent(problem=mountaincar, max_hist=opts.MAX_NODE_HIST,
                                    check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0,
                                    training_mode=opts.TRAINING_MODE)
    CUTreeAgent.read_Utree(game_number=200, save_path=CUTreeAgent.SAVE_PATH)
    utree = CUTreeAgent.utree
    # utree.print_tree_structure(CUTreeAgent.PRINT_TREE_PATH)

    flag, path = recursive_find_path(utree.root, idx)
    path_list = path.split(',')
    feature_value_dict = {}
    for path_section in path_list[:-1]:
        path_section = path_section.strip()
        path_section_list = path_section.split(' ')
        feature_name = path_section_list[0] + path_section_list[1]
        value = float(path_section_list[2])

        if feature_value_dict.get(feature_name) is not None:
            feature_value = feature_value_dict.get(feature_name)

            if path_section_list[1] == '<':
                feature_value = feature_value if feature_value < value else value
            elif path_section_list[1] == '>':
                feature_value = feature_value if value < feature_value else value
            feature_value_dict.update({feature_name: feature_value})
        else:
            feature_value_dict.update({feature_name: value})

    # CUTreeAgent.feature_importance()
    print feature_value_dict
    print 'path_length is {0}'.format(len(path_list[:-1]))
    print '{0}'.format(path_list[-1])


if __name__ == "__main__":
    # test()
    # idx = 360
    idx = 848
    # idx = 2832
    find_idx_path(idx)
