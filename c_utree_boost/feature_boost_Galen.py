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


def feature_importance():
    mountaincar = Problem_moutaincar.MoutainCar()
    CUTreeAgent = Agent.CUTreeAgent(problem=mountaincar, max_hist=opts.MAX_NODE_HIST,
                                    check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0, training_mode=opts.TRAINING_MODE)
    CUTreeAgent.feature_importance()
    print "hello"


if __name__ == "__main__":
    # test()
    feature_importance()
