import scipy as sp
import scipy.stats
import Agent_boost_Galen as Agent
import Problem_moutaincar
import C_UTree_boost_Galen as C_UTree
import gym
import numpy as np
import tensorflow as tf
import linear_regression

env = gym.make('MountainCar-v0')
ACTION_LIST = [0, 1, 2]


def get_action_linear_regression(observation, CUTreeAgent):
    Q_list = []
    Q_number = []
    for action_test in ACTION_LIST:
        sess = tf.Session()
        inst = C_UTree.Instance(-1, observation, action_test, observation, None,
                                None)  # leaf is located by the current observation
        node = CUTreeAgent.utree.getAbsInstanceLeaf(inst)
        LR = linear_regression.LinearRegression()
        LR.read_weights(weights=node.weight, bias=node.bias)
        LR.readout_linear_regression_model()
        sess.run(LR.init)
        temp = sess.run(LR.pred, feed_dict={LR.X: [inst.currentObs]}).tolist()
        Q_list.append(temp)
        Q_number.append(len(node.instances))
    return ACTION_LIST[Q_list.index(max(Q_list))]


import numpy as np
import scipy as sp
import scipy.stats


def get_action_similar_instance(observation, CUTreeAgent):
    min_mse = 10000
    Q_value = 0
    action = None
    for action_test in ACTION_LIST:
        inst = C_UTree.Instance(-1, observation, action_test, observation, None,
                                None)  # leaf is located by the current observation
        node = CUTreeAgent.utree.getAbsInstanceLeaf(inst)

        for instance in node.instances:
            instance_observation = instance.currentObs
            mse = ((np.asarray(observation) - np.asarray(instance_observation)) ** 2).mean()
            if mse < min_mse:
                min_mse = mse
                Q_value = instance.qValue
                action = action_test
    return action


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def test():
    ice_hockey_problem = Problem_moutaincar.MoutainCar(games_directory='../save_all_transition/')
    CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=3000,
                                    check_fringe_freq=1200, is_episodic=0, training_mode='_linear_epoch_decay_lr')
    CUTreeAgent.read_Utree(game_number=5, save_path=CUTreeAgent.SAVE_PATH)

    reward_list = []
    game_number = 100
    for i in range(game_number):
        observation = env.reset()
        done = False
        count = 0
        total_reward = 0
        for i in range(0, 500):

            if done and i != 200:
                break

            action = get_action_similar_instance(observation.tolist(), CUTreeAgent)
            # action = 0
            newObservation, reward, done, _ = env.step(action)

            # observation_str = ''
            # for feature in observation:
            #     observation_str += str(feature) + '$'
            # observation_str = observation_str[:-1]
            #
            # newObservation_str = ''
            # for feature in newObservation:
            #     newObservation_str += str(feature) + '$'
            # newObservation_str = newObservation_str[:-1]

            observation = newObservation
            total_reward += reward
            count += 1
            # print('U-tree: The episode ' + str(i) + ' lasted for ' + str(
            #     count) + ' time steps' + ' with action ' + str(action))
        reward_list.append(total_reward)

    # length2divide = float(len(reward_list)) / 1000
    mean, h = mean_confidence_interval(reward_list)
    print 'mean:{0}+-{1}'.format(str(mean), str(h))


if __name__ == "__main__":
    test()
