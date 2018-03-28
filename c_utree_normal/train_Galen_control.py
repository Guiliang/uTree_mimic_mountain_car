import optparse

import Agent_control
import Problem_moutaincar_control

optparser = optparse.OptionParser()
optparser.add_option("-m", "--max_node_hist", dest="MAX_NODE_HIST", default=3000,
                     help="max number of instance in every node (default = 3000)")
optparser.add_option("-c", "--check_fringe_freq", dest="CHECK_FRINGE_FREQ", default=1200,
                     help="check fringe frequent (default = 100)")
optparser.add_option("-d", "--directory_of_games", dest="GAME_DIRECTORY", default="",
                     help="games dir of all the games")

opts = optparser.parse_args()[0]


def train_model():
    ice_hockey_problem = Problem_moutaincar_control.MoutainCar(games_directory=opts.GAME_DIRECTORY)
    CUTreeAgent = Agent_control.CUTreeAgent(problem=ice_hockey_problem, max_hist=opts.MAX_NODE_HIST,
                                            check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0)
    CUTreeAgent.episode()


# def test_model():
#     ice_hockey_problem = Problem_moutaincar.MoutainCar(games_directory=opts.GAME_DIRECTORY)
#     CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=opts.MAX_NODE_HIST,
#                                     check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0)
#     CUTreeAgent.print_event_values()


if __name__ == "__main__":
    train_model()
