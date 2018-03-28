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
optparser.add_option("-g", "--game number to test", dest="GAME_NUMBER", default=40,
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


def test():
    ice_hockey_problem = Problem_moutaincar.MoutainCar()
    CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=opts.MAX_NODE_HIST,
                                    check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0)

    CUTreeAgent.boost_tree_testing_performance(
        save_path='/Local-Scratch/UTree model/mountaincar/model_boost_linear_qsplit_noabs_save{0}/'.format(opts.TRAINING_MODE),
        read_game_number=opts.GAME_NUMBER, save_correlation_dir = opts.SAVE_CORRELATION_DIR,
        save_mse_dir =opts.SAVE_MSE_DIR, save_mae_dir =opts.SAVE_MAE_DIR, save_rae_dir=opts.SAVE_RAE_DIR,
        save_rse_dir=opts.SAVE_RSE_DIR)


def test_sh():
    print "hello"


if __name__ == "__main__":
    test()
