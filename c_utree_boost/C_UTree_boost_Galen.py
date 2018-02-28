# uncompyle6 version 2.14.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.4.3 (default, Nov 17 2016, 01:08:31) 
# [GCC 4.8.4]
# Embedded file name: /Local-Scratch/PycharmProjects/Sport-Analytic-U-Tree/continuous-U-Tree-ice-hockey/c_utree_oracle/C_UTree_boost_Galen.py
# Compiled at: 2018-01-03 14:44:40
import random, numpy as np, optparse, sys, csv
import gc
import math
from collections import defaultdict
import linear_regression
from scipy.stats import ks_2samp
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

NodeSplit = 0
NodeLeaf = 1
NodeFringe = 2
ActionDimension = -1
AbsorbAction = 5
HOME = 0
AWAY = 1
MAX_DEPTH = 20


class CUTree:
    def __init__(self, gamma, n_actions, dim_sizes, dim_names, max_hist, max_back_depth=1, minSplitInstances=50,
                 significance_level=0.0005, is_episodic=0, hard_code_flag=True, training_mode=''):

        # LR = linear_regression.LinearRegression()
        # weight = LR.weight_initialization()
        # bias = LR.bias_initialization()

        self.node_id_count = 0
        self.root = UNode(self.genId(), NodeLeaf, None, n_actions, 1)
        self.n_actions = n_actions
        self.max_hist = max_hist
        self.max_back_depth = max_back_depth
        self.gamma = gamma
        self.history = []
        self.n_dim = len(dim_sizes)
        self.dim_sizes = dim_sizes
        self.dim_names = dim_names
        self.minSplitInstances = minSplitInstances
        self.significanceLevel = significance_level
        self.nodes = {self.root.idx: self.root}
        self.term = UNode(self.genId(), NodeLeaf, None, 1, 1)
        self.start = UNode(self.genId(), NodeLeaf, None, 1, 1)
        self.nodes[self.term.idx] = self.term
        self.nodes[self.start.idx] = self.start
        self.hard_code_flag = hard_code_flag
        self.training_mode = training_mode
        self.game_number = None

        # self.root.weight = weight
        # self.root.bias = bias
        # self.term.weight = weight
        # self.term.bias = bias
        # self.start.weight = weight
        # self.start.bias = bias

        return

    def tocsvFile(self, filename):
        """
        Store a record of U-Tree in file, make it easier to rebuild tree
        :param filename: the path of file to store the record
        :return:
        """
        with open(filename, 'wb') as (csvfile):
            fieldname = [
                'idx', 'dis', 'dis_value', 'par', 'q_home', 'q_away']
            writer = csv.writer(csvfile)
            writer.writerow(fieldname)
            for i, node in self.nodes.items():
                if node.nodeType == NodeSplit:
                    writer.writerow([node.idx,
                                     node.distinction.dimension,
                                     node.distinction.continuous_divide_value if node.distinction.continuous_divide_value else None,
                                     node.parent.idx if node.parent else None,
                                     None,
                                     None])
                else:
                    writer.writerow([node.idx,
                                     None,
                                     None,
                                     node.parent.idx if node.parent else None,
                                     node.qValues_home,
                                     node.qValues_away])

        return

    def getAbsInstanceLeaf(self, inst, ntype=NodeLeaf):
        node = self.root
        while node.nodeType != ntype:
            child = node.applyInstanceDistinction(inst)
            node = node.children[child]

        return node

    def tocsvFileComplete(self, filename):
        """
        Store a record of U-Tree in file including the instances in each leaf node,
        make it easier to rebuild tree
        :param filename: the path of file to store record
        :return:
        """
        with open(filename, 'wb') as (csvfile):
            fieldnamecomplete = [
                'idx', 'dis', 'dis_value', 'par', 'q_home', 'q_away', 'instances']
            writer = csv.writer(csvfile)
            writer.writerow(fieldnamecomplete)
            for i, node in self.nodes.items():
                if node.nodeType == NodeSplit:
                    writer.writerow([node.idx,
                                     node.distinction.dimension,
                                     node.distinction.continuous_divide_value if node.distinction.continuous_divide_value else None,
                                     node.parent.idx if node.parent else None,
                                     None,
                                     None,
                                     None,
                                     None])
                else:
                    writer.writerow([node.idx,
                                     None,
                                     None,
                                     node.parent.idx if node.parent else None,
                                     node.qValues_home,
                                     node.qValues_away,
                                     node.qValues_end,
                                     [inst.timestep for inst in node.instances]])

        return

    def fromcsvFile(self, filename):
        """
        Load U-Tree structure from csv file
        :param filename: the path of file to load record
        :return:
        """
        with open(filename, 'rb') as (csvfile):
            fieldname = [
                'idx', 'dis', 'dis_value', 'par', 'q_home', 'q_away']
            reader = csv.reader(csvfile)
            self.node_id_count = 0
            for record in reader:
                if record[0] == fieldname[0]:
                    continue
                if not record[4]:
                    node = UNode(int(record[0]), NodeSplit, self.nodes[int(record[3])] if record[3] else None,
                                 self.n_actions, self.nodes[int(record[3])].depth + 1 if record[3] else 1)
                    node.distinction = Distinction(dimension=int(record[1]), back_idx=0,
                                                   dimension_name=self.dim_names[int(record[1])],
                                                   iscontinuous=True if record[2] else False,
                                                   continuous_divide_value=float(record[2]) if record[2] else None)
                else:
                    node = UNode(int(record[0]), NodeLeaf, self.nodes[int(record[3])] if record[3] else None,
                                 self.n_actions, self.nodes[int(record[3])].depth + 1 if record[3] else 1)
                    node.qValues_home = np.array(map(float, record[4][1:-1].split()))
                    node.qValues_away = np.array(map(float, record[5][1:-1].split()))
                    node.qValues_end = np.array(map(float, record[6][1:-1].split()))
                if node.parent:
                    self.nodes[int(record[3])].children.append(node)
                if node.idx == 1:
                    self.root = node
                else:
                    if node.idx == 2:
                        self.term = node
                self.nodes[int(node.idx)] = node
                self.node_id_count += 1

        return

    def print_tree(self):
        """
        print U tree
        :return:
        """
        self.print_tree_recursive('', self.root)

    def print_tree_structure(self, file_directory):
        root = self.root
        with open(file_directory, 'w') as (f):
            tree_structure = self.recursive_print_tree_structure(root, 0)
            print >> f, tree_structure

    def recursive_print_tree_structure(self, node, layer):
        tree_structure = ''
        for i in range(0, layer):
            tree_structure += '\t'

        if node.nodeType == NodeSplit:
            tree_structure += ('idx{2}(Q:{0}, distinct_name:{1}, dictinctin_value:{4}, par:{3})').format(
                format(node.utility(), '.4f'), node.distinction.dimension_name,
                node.idx, node.parent.idx if node.parent is not None else None,
                node.distinction.continuous_divide_value
            )
            child_string = ''
            for child in node.children:
                child_string += '\n' + self.recursive_print_tree_structure(child, layer + 1)

            tree_structure += child_string
        else:
            if node.nodeType == NodeLeaf:
                tree_structure += 'idx{1}(Q:{0}, par:{2})'.format(
                    format(node.utility(), '.4f'),
                    node.idx, node.parent.idx if node.parent is not None else None
                )
            else:
                raise ValueError(('Unsupported tree nodeType:{0}').format(node.nodeType))
        return tree_structure

    def print_tree_recursive(self, blank, node):
        """
        recursively print tree from root to leaves
        :param node: the node to be expand
        :return:
        """
        if node.nodeType == NodeSplit:
            print blank + ('idx={}, dis={}, par={}').format(node.idx, node.distinction.dimension,
                                                            node.parent.idx if node.parent else None)
            for child in node.children:
                self.print_tree_recursive(blank + ' ', child)

        else:
            print blank + ('idx={}, t_h_h={}, t_h_a={}, t_a_h={}, t_a_a={}, q_h={}, q_a={}, par={}').format(node.idx,
                                                                                                            node.transitions_home_home,
                                                                                                            node.transitions_home_away,
                                                                                                            node.transitions_away_home,
                                                                                                            node.transitions_away_away,
                                                                                                            node.qValues_home,
                                                                                                            node.qValues_away,
                                                                                                            node.parent.idx if node.parent else None)
        return

    def getInstanceQvalues(self, instance, reward):
        """
        get the Q-value from instance, q(I,a)
        :return: state's maximum Q
        """
        self.insertInstance(instance)
        if instance.action == AbsorbAction:
            if reward == 1:
                return (1, 0)
            return (0, 1)
        else:
            next_state = self.getInstanceLeaf(instance)
        return (next_state.utility(home_identifier=True),
                next_state.utility(home_identifier=False))

    def getTime(self):
        """
        :return: length of history
        """
        return len(self.history)

    def updateCurrentNode(self, instance, beginflag):
        """
        add the new instance ot LeafNode
        :param instance: instance to add
        :return:
        """
        old_state = self.getLeaf(previous=1)
        self.insertInstance(instance)
        new_state = self.getLeaf()
        self.updateParents(new_state)
        new_state.addInstance(instance, self.max_hist)
        if not beginflag:
            old_state.updateModel(new_state=new_state.idx, action=self.history[-2].action,
                                  qValue=self.history[-2].qValue)
        if instance.nextObs[0] == -1 or instance.action == AbsorbAction:
            new_state.updateModel(new_state=self.start.idx, action=instance.action, qValue=instance.qValue)

    def sweepLeaves(self):
        """
        Serve as a public function calls sweepRecursive
        :return:
        """
        return self.sweepRecursive(self.root, self.gamma)

    def sweepRecursive(self, node, gamma):
        """
        Apply single step of value iteration to leaf node
        or recursively to children if it is a split node
        :param node: target node
        :param gamma: gamma in dynamic programming
        :return:
        """
        if node.nodeType == NodeLeaf:
            for action, reward in enumerate(node.rewards_home):
                c = float(node.count_home[action])
                if c == 0:
                    continue
                exp = 0
                for node_to, t_h in node.transitions_home_home[action].items():
                    t_a = node.transitions_home_away[action][node_to]
                    if reward[node_to] > 0:
                        exp += reward[node_to] / c
                    if node.idx != node_to:
                        exp += gamma * (
                            self.nodes[node_to].utility(True) * t_h + self.nodes[node_to].utility(False) * t_a) / c

                node.qValues_home[action] = exp

            for action, reward in enumerate(node.rewards_away):
                c = float(node.count_away[action])
                if c == 0:
                    continue
                exp = 0
                for node_to, t_h in node.transitions_away_home[action].items():
                    t_a = node.transitions_away_away[action][node_to]
                    if reward[node_to] > 0:
                        exp += reward[node_to] / c
                    if node.idx != node_to:
                        exp += gamma * (
                            self.nodes[node_to].utility(True) * t_h + self.nodes[node_to].utility(False) * t_a) / c

                node.qValues_away[action] = exp

        for c in node.children:
            self.sweepRecursive(c, gamma)

    def insertInstance(self, instance):
        """
        append new instance to history
        :param instance: current instance
        :return:
        """
        self.history.append(instance)

    def nextInstance(self, instance):
        """
        get the next instance
        :param instance: current instance
        :return: the next instance
        """
        return self.history[instance.timestep + 1]

    def transFromInstances(self, node, n_id, action):
        """
        compute transition probability from current node to n_id node when perform action
        Formula (7) in U tree paper
        :param node: current node
        :param n_id: target node
        :param action: action to perform
        :return: transition probability
        """
        count = 0
        total = 0
        for inst in node.instances:
            if inst.action == action:
                leaf_to = self.getInstanceLeaf(inst, previous=1)
                if leaf_to.idx == n_id:
                    count += 1
                total += 1

        if total:
            return count / total
        return 0

    def rewardFromInstances(self, node, action):
        """
        compute reward of perform action on current node
        Formula (6) in U tree paper
        :param node: current node
        :param action: action to perform
        :return: reward computed
        """
        rtotal = 0
        total = 0
        for inst in node.instances:
            if inst.action == action:
                rtotal += inst.reward
                total += 1

        if total:
            return rtotal / total
        return 0

    def modelFromInstances(self, node):
        """
        rebuild model for leaf node, with newly added instance
        :param node:
        :return:
        """
        node.count = np.zeros(self.n_actions)
        node.transitions = [{} for i in range(self.n_actions)]
        for inst in node.instances:
            leaf_to = self.getInstanceLeaf(inst, previous=1)
            if leaf_to != self.term:
                node.updateModel(leaf_to.idx, inst.action, inst.qValue)
            else:
                node.updateModel(leaf_to.idx, inst.action, inst.qValue)

    def getLeaf(self, previous=0):
        """
        Get leaf corresponding to current history
        :param previous: 0 is not check goal, 1 is check it
        :return:
        """
        idx = len(self.history) - 1
        node = self.root
        if previous == 1:
            if idx == -1 or self.history[idx].nextObs[0] == -1:
                return self.start
        while node.nodeType != NodeLeaf:
            child = node.applyDistinction(self.history, idx)
            node = node.children[child]

        return node

    def updateParents(self, new_state):
        node = self.root
        idx = len(self.history) - 1
        instance = self.history[idx]
        action = instance.action
        qValue = instance.qValue
        while node.nodeType != NodeLeaf:
            node.qValues[action] = (node.qValues[action] * node.count[action] + qValue) / (node.count[action] + 1)
            node.count[action] += 1
            transition_count = node.transitions[action].get(new_state.idx)
            if transition_count is not None:
                transition_count += 1
                node.transitions[action].update({new_state.idx: transition_count})
            else:
                node.transitions[action].update({new_state.idx: 1})

            child = node.applyDistinction(self.history, idx)
            node = node.children[child]

    def getInstanceLeaf(self, inst, ntype=NodeLeaf, previous=0):
        """
        Get leaf that inst records a transition from
        previous=0 indicates transition_from, previous=1 indicates transition_to
        :param inst: target instance
        :param ntype: target node type
        :param previous: previous=0 indicates present inst, previous=1 indicates next inst
        :return:
        """
        idx = inst.timestep + previous
        if previous == 1:
            if idx >= len(self.history):
                return self.term
            if inst.nextObs[0] == -1 or inst.action == AbsorbAction:
                return self.start
        node = self.root
        while node.nodeType != ntype:
            child = node.applyDistinction(self.history, idx)
            node = node.children[child]

        return node

    def genId(self):
        """
        :return: a new ID for node
        """
        self.node_id_count += 1
        return self.node_id_count

    def reduceId(self, count):
        """
        After splitFringe(maybe something else), reduce to normal
        :param count: the reduce number
        :return:
        """
        self.node_id_count -= count

    def split(self, node, distinction):
        """
        split decision tree on nodes
        :param node: node to split
        :param distinction: distinction to split
        :return:
        """
        node.nodeType = NodeSplit
        node.distinction = distinction
        if distinction.dimension == ActionDimension:
            for i in range(self.n_actions):
                idx = self.genId()
                n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
                self.nodes[idx] = n
                node.children.append(n)
                n.weight = node.weight
                n.bias = node.bias

        else:
            if not distinction.iscontinuous:
                for i in range(self.dim_sizes[distinction.dimension]):
                    idx = self.genId()
                    n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
                    self.nodes[idx] = n
                    node.children.append(n)
                    n.weight = node.weight
                    n.bias = node.bias

            else:
                for i in range(2):
                    idx = self.genId()
                    n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
                    self.nodes[idx] = n
                    node.children.append(n)
                    n.weight = node.weight
                    n.bias = node.bias

        for inst in node.instances:
            n = self.getInstanceLeaf(inst, previous=0)
            n.addInstance(inst, self.max_hist)

        for i, n in self.nodes.items():
            if n.nodeType == NodeLeaf:
                self.modelFromInstances(n)

        node.instances = []

    def splitToFringe(self, node, distinction):
        """
        Create fringe nodes instead of leaf nodes after splitting; these nodes
        aren't used in the agent's model
        :param node: node to split
        :param distinction: distinction used for splitting
        :return:
        """
        node.distinction = distinction
        if distinction.dimension == ActionDimension:
            for i in range(self.n_actions):
                idx = self.genId()
                fringe_node = UNode(idx, NodeFringe, node, self.n_actions, node.depth + 1)
                node.children.append(fringe_node)

        else:
            if not distinction.iscontinuous:
                for i in range(self.dim_sizes[distinction.dimension]):
                    idx = self.genId()
                    fringe_node = UNode(idx, NodeFringe, node, self.n_actions, node.depth + 1)
                    node.children.append(fringe_node)

            else:
                for i in range(2):
                    idx = self.genId()
                    fringe_node = UNode(idx, NodeFringe, node, self.n_actions, node.depth + 1)
                    node.children.append(fringe_node)

        for inst in node.instances:
            n = self.getInstanceLeaf(inst, ntype=NodeFringe, previous=0)
            n.addInstance(inst, self.max_hist)

    def unsplit(self, node):
        """
        Unsplit node
        :param node: the node to unsplit
        :return:
        """
        node.distinction = None
        self.reduceId(len(node.children))
        if node.nodeType == NodeSplit:
            node.nodeType = NodeLeaf
            for c in node.children:
                del self.nodes[c.idx]

            for i, n in self.nodes.items():
                if n.nodeType == NodeLeaf:
                    self.modelFromInstances(n)

        node.children = []
        return

    def testFringe(self):
        """
        Tests fringe nodes for viable splits, splits nodes if they're found
        :return: how many real splits it takes
        """

        # if self.hard_code_flag:
        #     result_flag = self.hard_code_split()
        #     self.hard_code_flag = result_flag

        return self.testFringeRecursive(self.root)

    def testFringeRecursive(self, node):
        """
        recursively perform test in fringe, until return total number of split
        :param node: node to test
        :return: number of splits
        """
        if node.depth >= MAX_DEPTH:
            return 0
        if node.nodeType == NodeLeaf:
            # if self.game_number <= 50:
            #     self.train_linear_regression_on_leaves(node)
            # if self.game_number > 50 and self.game_number % 5 == 0:
            #     self.train_linear_regression_on_leaves(node)
            self.train_linear_regression_on_leaves(node)
            d = self.getUtileDistinction(node)
            if d:
                self.split(node, d)
                if self.hard_code_flag:
                    result_flag = self.hard_code_split()
                    self.hard_code_flag = result_flag

                return 1 + self.testFringeRecursive(node)
            return 0
        total = 0
        for c in node.children:
            total += self.testFringeRecursive(c)

        return total

    def getUtileDistinction(self, node):
        """
        Different kinds of tests are performed here
        1. find all the possible distinction
        2. try to split node according to distinction and get expected future discounted returns
        3. perform test until find the proper distinction, otherwise, return None
        """
        if len(node.instances) < self.minSplitInstances:
            return None
        cds = self.getCandidateDistinctions(node)
        return self.ksTestonQ(node, cds)

    def ksTest(self, node, cds, p_significanceLevel=float(0.54)):
        """
        KS test is performed here
        1. find all the possible distinction
        2. try to split node according to distinction and get expected future discounted returns
        3. perform ks test until find the proper distinction, otherwise, return None
        :param p_significanceLevel:
        :param node:
        :return:
        """
        assert node.nodeType == NodeLeaf
        root_utils_home, root_utils_away = self.getEFDRs(node)
        child_utils = []
        p_min = float(1)
        cd_min = None
        for cd in cds:
            self.splitToFringe(node, cd)
            for c in node.children:
                child_utils.append(self.getEFDRs(c))

            self.unsplit(node)
            for i, cu in enumerate(child_utils):
                k_home, p_home = ks_2samp(root_utils_home, cu[0])
                k_away, p_away = ks_2samp(root_utils_away, cu[1])
                p = p_home if p_home < p_away else p_away
                if p < p_significanceLevel and p < p_min:
                    p_min = p
                    cd_min = cd
                    print ('KS passed, p={}, d = {}, back={}').format(p, cd.dimension, cd.back_idx)

        if cd_min:
            print ('Will be split, p={}, d={}, back={}').format(p_min, cd_min.dimension_name, cd_min.back_idx)
            return cd_min
        else:
            return cd_min

    def getEFDRs(self, node):
        """
        Get all expected future discounted returns for all instances in a node
        (q-value is just the average EFDRs)
        """
        efdrs_home = np.zeros(len(node.instances))
        efdrs_away = np.zeros(len(node.instances))
        for i, inst in enumerate(node.instances):
            next_state = self.getInstanceLeaf(inst, previous=1)
            if next_state == 'exceed':
                efdrs_home[i] = inst.reward
                efdrs_away[i] = -inst.reward
            elif inst.action == 5:
                efdrs_home[i] = inst.reward
                efdrs_away[i] = -inst.reward
            elif node.parent is None:
                efdrs_home[i] = inst.reward
                efdrs_away[i] = -inst.reward
            elif node.parent.idx == next_state.idx:
                efdrs_home[i] = inst.reward
                efdrs_away[i] = -inst.reward
            else:
                next_home_state_util = next_state.utility(True)
                efdrs_home[i] = inst.reward + self.gamma * next_home_state_util
                next_away_state_util = next_state.utility(False)
                efdrs_away[i] = -inst.reward + self.gamma * next_away_state_util

        return [efdrs_home, efdrs_away]

    def splitQs(self, node, cd):

        if cd.iscontinuous:
            Q_value_list = []
            for i in range(0, 2):
                Q_value_list.append([])

            for inst in node.instances:

                if inst.currentObs[cd.dimension] <= cd.continuous_divide_value:
                    Q_value_list[0].append(inst.qValue)
                else:
                    Q_value_list[1].append(inst.qValue)

        else:
            Q_value_list = []
            for i in range(0, self.n_actions):
                Q_value_list.append([])

            for inst in node.instances:
                Q_value_list[int(inst.action)].append(inst.qValue)

        return Q_value_list

    def ksTestonQ(self, node, cds, diff_significanceLevel=float(0.01)):
        """
        KS test is performed here
        1. find all the possible distinction
        2. try to split node according to distinction and get expected future discounted returns
        3. perform ks test until find the proper distinction, otherwise, return None
        :param diff_significanceLevel:
        :param node:
        :return:
        """
        assert node.nodeType == NodeLeaf
        root_utils = self.getQs(node)
        variance = np.var(root_utils)
        diff_max = float(0)
        cd_split = None
        for cd in cds:
            child_qs = self.splitQs(node, cd)
            for i, cq in enumerate(child_qs):

                if len(cq) == 0:
                    continue
                else:
                    variance_child = np.var(cq)

                    diff = variance - variance_child
                    if diff > diff_significanceLevel and diff > diff_max:
                        diff_max = diff
                        cd_split = cd
                        print >> sys.stderr, 'variance test passed, diff={}, d = {}, back={}'.format(diff, cd.dimension,
                                                                                                     cd.back_idx)

        if cd_split:
            print >> sys.stderr, 'Will be split, p={}, d={}, back={}'.format(diff_max, cd_split.dimension_name,
                                                                             cd_split.back_idx)
            return cd_split
        else:
            return cd_split

    def varDiff(self, listA=[], listB=[], diff=0):
        if len(listA) == 0 or len(listB) == 0:
            return diff - 1
        mean_a = sum(listA) / len(listA)
        var_a = float(0)
        for number_a in listA:
            var_a += (number_a - mean_a) ** 2

        mean_b = sum(listB) / len(listB)
        var_b = float(0)
        for number_b in listB:
            var_b += (number_b - mean_b) ** 2

        return abs(var_a / len(listA) - var_b / len(listB))

    def getQs(self, node):
        """
        Get all expected future discounted returns for all instances in a node
        (q-value is just the average EFDRs)
        """
        efdrs = np.zeros(len(node.instances))
        for i, inst in enumerate(node.instances):
            efdrs[i] = inst.qValue
        return [efdrs]

    def hard_code_split(self):
        root = self.root
        if len(root.children) == 0 and len(self.history) >= 100:
            print "\nHard Coding\n"
            d = Distinction(dimension=-1, back_idx=0, dimension_name='actions')
            self.split(root, d)
            return True
        elif len(root.children) == 13:
            print "\nHard Coding\n"
            d = Distinction(dimension=9, back_idx=0, dimension_name=self.dim_names[9], iscontinuous=True,
                            continuous_divide_value=float(0))  # 0 is good enough
            self.split(root.children[5], d)
            return False
        else:
            return True

    def train_linear_regression_on_leaves(self, node):
        leaves_number = 0
        if node.nodeType != NodeLeaf:
            for child in node.children:
                leaves_number += self.train_linear_regression_on_leaves(child)
            return leaves_number
        else:
            train_x = []
            train_y = []
            # before = defaultdict(int)
            # after = defaultdict(int)
            # for i in gc.get_objects():
            #     before[type(i)] += 1
            for instance in node.instances:
                train_x.append(instance.currentObs)
                train_y.append([instance.qValue])
            if len(train_x) != 0 and len(train_y) != 0:
                sess = tf.InteractiveSession(config=config)
                if self.training_mode == '_epoch_linear':
                    training_epochs = len(node.instances)
                    # if self.game_number > 50:
                    #     training_epochs = training_epochs * 5
                    LR = linear_regression.LinearRegression(training_epochs=training_epochs)
                elif self.training_mode == '_linear_epoch_decay_lr':
                    node.update_times += 1
                    times = node.update_times
                    lr = 0.1 * float(1) / (1 + 0.0225 * times) * math.pow(0.977, len(node.instances) / 30)
                    # lr = 0.05*math.pow(0.02, float(len(node.instances))/float(self.max_hist))
                    training_epochs = len(node.instances) if len(node.instances) > 50 else 50
                    # if self.game_number > 50:
                    #     training_epochs = training_epochs * 5
                    #     lr = float(lr) / 5
                    LR = linear_regression.LinearRegression(training_epochs=training_epochs, learning_rate=lr)
                elif len(self.training_mode) == 0:
                    LR = linear_regression.LinearRegression()
                else:
                    raise ValueError("undefined training mode")
                if node.weight is None or node.bias is None:
                    LR.read_weights()
                else:
                    LR.read_weights(node.weight, node.bias)
                LR.linear_regression_model()
                trained_weights, trained_bias, average_diff = LR.gradient_descent(sess=sess, train_X=train_x, train_Y=train_y)
                print >> sys.stderr, 'node index is {0}'.format(node.idx)
                LR.delete_para()
                node.weight = None
                node.bias = None
                node.weight = trained_weights
                node.bias = trained_bias
                node.average_diff = average_diff
                trained_weights = None
                trained_bias = None
                train_x = None
                train_y = None
                del LR
                sess.close()
                del sess
                gc.collect()

                # for i in gc.get_objects():
                #     after[type(i)] += 1
                # for k in after:
                #     if after[k] - before[k]:
                #         print (k, after[k] - before[k])

            return 1

    def getCandidateDistinctions(self, node, select_interval=100):
        """
        construct all candidate distinctions
        :param node: target nodes
        :return: all candidate distinctions
        """
        p = node.parent
        anc_distinctions = []
        while p:
            anc_distinctions.append(p.distinction)
            p = p.parent

        candidates = []
        for i in range(self.max_back_depth):
            for j in range(-1, self.n_dim):
                if j > -1 and self.dim_sizes[j] == 'continuous':
                    count = 0
                    for inst in sorted(node.instances, key=lambda inst: inst.currentObs[j]):
                        count += 1
                        if count % select_interval != 0:
                            continue
                        d = Distinction(dimension=j, back_idx=i, dimension_name=self.dim_names[j], iscontinuous=True,
                                        continuous_divide_value=inst.currentObs[j])
                        if d in anc_distinctions:
                            continue
                        else:
                            candidates.append(d)

                else:
                    d = Distinction(dimension=j, back_idx=i, dimension_name=self.dim_names[j] if j > -1 else 'actions')
                    if d in anc_distinctions:
                        continue
                    else:
                        candidates.append(d)

        return candidates


class UNode:
    def __init__(self, idx, nodeType, parent, n_actions, depth):
        self.idx = idx
        self.nodeType = nodeType
        self.parent = parent
        self.children = []
        self.count = np.zeros(n_actions)
        self.transitions = [{} for i in range(n_actions)]
        self.qValues = np.zeros(n_actions)
        self.distinction = None
        self.instances = []
        self.depth = depth
        self.weight = None
        self.bias = None
        self.average_diff = None
        self.update_times = 0

        # LR = linear_regression.LinearRegression()
        # self.weight = LR.weight_initialization()
        # self.bias = LR.bias_initialization()
        # return

    def utility(self):
        """
        :param: index: if index is HOME, return Q_home, else return Q_away
        :return: maximum Q value
        """
        qValues_cp = np.copy(self.qValues)
        qValues_cp = qValues_cp[qValues_cp != float(0)]
        return max(qValues_cp)

    def addInstance(self, instance, max_hist):
        """
        add new instance to node instance list
        if instance length exceed maximum history length, select most recent history
        :param instance:
        :param max_hist:
        :return:
        """
        self.instances.append(instance)
        if len(self.instances) > max_hist:
            self.instances = self.instances[1:]

    def updateModel(self, new_state, action, qValue):
        """
        1. add action reward
        2. add action count
        3. record transition states
        :param new_state: new transition state
        :param action: new action
        :param reward: reward of action
        :param home_identifier: identify home and away
        :return:
        """
        self.qValues[action] = (self.qValues[action] * self.count[action] +
                                qValue) / (self.count[action] + 1)

        self.count[action] += 1
        if new_state not in self.transitions[action]:

            self.transitions[action][new_state] = 1
        else:
            self.transitions[action][new_state] += 1

    def applyDistinction(self, history, idx, previous=0):
        """
        :param history: history of instances
        :param idx: the idx of instance to apply distinction
        :return: the index of children
        """
        inst = history[idx - self.distinction.back_idx]
        if self.distinction.dimension == ActionDimension:
            return inst.action
        if previous == 0:
            if self.distinction.iscontinuous:
                if inst.currentObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
                    return 0
                else:
                    return 1
            else:
                return int(inst.currentObs[self.distinction.dimension])
        else:
            if self.distinction.iscontinuous:
                if inst.nextObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
                    return 0
                return 1
            else:
                return int(inst.nextObs[self.distinction.dimension])

    def applyInstanceDistinction(self, inst):
        if self.distinction.dimension == ActionDimension:
            return inst.action
        if self.distinction.iscontinuous:
            if inst.currentObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
                return 0
            return 1
        else:
            return int(inst.currentObs[self.distinction.dimension])


class Instance:
    """
    records the transition as an instance
    """

    def __init__(self, timestep, currentObs, action, nextObs, reward, qValue):
        self.timestep = int(timestep)
        self.action = int(action)
        self.nextObs = map(float, nextObs)
        self.currentObs = map(float, currentObs)
        self.reward = reward
        self.qValue = qValue


class Distinction:
    """
    For split node
    """

    def __init__(self, dimension, back_idx, dimension_name='unknown', iscontinuous=False, continuous_divide_value=None):
        """
        initialize distinction
        :param dimension: split of the node is based on the dimension
        :param back_idx: history index, how many time steps backward from the current time this feature will be examined
        :param dimension_name: the name of dimension
        :param iscontinuous: continuous or not
        :param continuous_divide_value: the value of continuous division
        """
        self.dimension = dimension
        self.back_idx = back_idx
        self.dimension_name = dimension_name
        self.iscontinuous = iscontinuous
        self.continuous_divide_value = continuous_divide_value

    def __eq__(self, distinction):
        return self.dimension == distinction.dimension and self.back_idx == distinction.back_idx and self.continuous_divide_value == distinction.continuous_divide_value
