import csv

import scipy as sp
import scipy.stats
import tensorflow as tf
import gym
import numpy as np
from collections import deque
import random


class Model:
    def __init__(self, env, learning_rate, memory):
        self.env = env
        self.num_actions = env.action_space.n
        self.num_env_space = len(env.observation_space.high)
        self.memory = memory

        self.input = tf.placeholder(tf.float32, shape=[None, self.num_env_space], name='input')
        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')

        init = tf.truncated_normal_initializer()

        # create the network
        net = self.input
        net = tf.layers.dense(inputs=net, units=100, activation=tf.nn.relu, kernel_initializer=init, name='dense1')
        net = tf.layers.dense(inputs=net, units=100, activation=tf.nn.relu, kernel_initializer=init, name='dense2')
        net = tf.layers.dense(inputs=net, units=100, activation=tf.nn.relu, kernel_initializer=init, name='dense3')
        net = tf.layers.dense(inputs=net, units=self.num_actions, activation=None, kernel_initializer=init)

        self.output = net

        q_reward = tf.reduce_sum(tf.multiply(self.output, self.actions), 1)
        loss = tf.reduce_mean(tf.squared_difference(self.rewards, q_reward))
        self.optimiser = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def train(self):
        if len(self.memory.memory) < self.memory.batch_size:
            return
        states, actions, rewards = self.memory.get_batch(self, self.memory.batch_size)
        self.session.run(self.optimiser, feed_dict={self.input: states, self.actions: actions, self.rewards: rewards})

    def save_model(self, game_number):
        print 'saving model'
        self.saver.save(self.session, "./model_save_mc_v0/mountaincar-v0-game-", global_step=game_number)

    def read_model(self):
        checkpoint = tf.train.get_checkpoint_state('./model_save_mc_v0/')
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = self.saver
            sess = self.session
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def get_qvalues(self, state_list):
        qValues = self.session.run(self.output, feed_dict={self.input: state_list})

        return qValues

    def get_action(self, state, epsilon):
        qValues = self.session.run(self.output, feed_dict={self.input: [state]})[0]
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(qValues)

        return action, qValues

    def get_batch_action(self, states):
        return self.session.run(self.output, feed_dict={self.input: states})


class ReplayMemory:
    def __init__(self, env, batch_size, max_memory_size, gamma):
        self.env = env
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = gamma

    def add(self, state, action, reward, done, next_state):
        actions = np.zeros(self.env.action_space.n)
        actions[action] = 1
        self.memory.append([state, actions, reward, done, next_state])

    def get_batch(self, model, batch_size=50):
        mini_batch = random.sample(self.memory, batch_size)
        states = [item[0] for item in mini_batch]
        actions = [item[1] for item in mini_batch]
        rewards = [item[2] for item in mini_batch]
        done = [item[3] for item in mini_batch]
        next_states = [item[4] for item in mini_batch]

        q_values = model.get_batch_action(next_states)
        y_batch = []

        for i in range(batch_size):
            if done[i]:
                y_batch.append(rewards[i])
            else:
                y_batch.append(rewards[i] + self.gamma * np.max(q_values[i]))

        return states, actions, y_batch


class Agent:
    def __init__(self):
        # Hyper-parameters
        self.gamma = 0.97
        self.learning_rate = 1e-3
        self.epsilon = 1.
        self.final_epsilon = .05
        self.epsilon_decay = .995

        # Memory parameters
        self.batch_size = 50
        self.max_memory_size = 10000

        self.env = gym.make('MountainCar-v0')

        self.memory = ReplayMemory(self.env, self.batch_size, self.max_memory_size, self.gamma)
        self.model = Model(self.env, self.learning_rate, self.memory)

        self.max_episodes = 50000
        self.render = True

    def train(self):

        for i in range(self.max_episodes):
            current_state = self.env.reset()
            done = False
            count = 0
            total_reward = 0

            while not done:
                if self.render:
                    self.env.render()

                action, _ = self.model.get_action(current_state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self.memory.add(current_state, action, reward, done, next_state)
                current_state = next_state
                total_reward += reward
                self.model.train()
                count += 1
            print('TRAIN: The episode ' + str(i) + ' lasted for ' + str(
                count) + ' time steps with epsilon ' + str(self.epsilon))

            # if i > 200:
            if self.epsilon > self.final_epsilon:
                self.epsilon *= self.epsilon_decay
            self.model.save_model(i)

    def test(self):

        self.model.read_model()
        test_epsilon = 0.1

        for i in range(100):
            observation = self.env.reset()
            done = False
            count = 0
            total_reward = 0
            total_reward_list = []

            record_transition = []

            while not done:
                if self.render:
                    self.env.render()

                action, qValues = self.model.get_action(observation, test_epsilon)
                newObservation, reward, done, _ = self.env.step(action)

                observation_str = ''
                for feature in observation:
                    observation_str += str(feature) + '$'
                observation_str = observation_str[:-1]

                newObservation_str = ''
                for feature in newObservation:
                    newObservation_str += str(feature) + '$'
                newObservation_str = newObservation_str[:-1]

                record_transition.append(
                    {'observation': observation_str, 'action': action, 'qValue': qValues[action], 'reward': reward,
                     'newObservation': newObservation_str})

                observation = newObservation
                total_reward += reward
                count += 1
            print('TRAIN: The episode ' + str(i) + ' lasted for ' + str(
                count) + ' time steps with epsilon ' + str(test_epsilon))

            self.mean_confidence_interval(total_reward_list)

            # self.save_csv_all_correlations(record_transition,
            #                                './save_all_transition_e{1}/record_moutaincar_transition_game{0}.csv'.format(
            #                                    int(i), str(test_epsilon)))

    def save_csv_all_correlations(self, record_transition, csv_name):
        with open(csv_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=record_transition[0].keys())
            writer.writeheader()

            for row_dict in record_transition:
                writer.writerow(row_dict)

    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
        return m, h


if __name__ == "__main__":
    agent = Agent()
    # agent.train()
    agent.test()
