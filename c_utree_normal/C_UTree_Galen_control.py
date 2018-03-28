import csv
import os
import random

import numpy as np
import sys
import os.path

from scipy.stats import ks_2samp

NodeSplit = 0
NodeLeaf = 1
NodeFringe = 2
ActionDimension = -1


class CUTree:
    def __init__(self, gamma, n_actions, dim_sizes, dim_names, max_hist, max_back_depth=1, minSplitInstances=50,
                 significance_level=50, is_episodic=0, max_depth=20):

        self.node_id_count = 0
        self.root = UNode(self.genId(), NodeLeaf, None, n_actions, 0)
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
        self.max_tree_depth = max_depth

        self.nodes = {self.root.idx: self.root}  # root_id:root_node

        self.term = UNode(self.genId(), NodeLeaf, None, 1, None)  # dummy terminal node with 0 value
        self.nodes[self.term.idx] = self.term  # term_id:term_node
        self.pre_split_flag = True

    def getTime(self):
        """
        :return: length of history
        """
        return len(self.history)

    def updateCurrentNode(self, instance, beginning_flag):
        """
        add the new instance ot LeafNode
        :param instance: instance to add
        :return:
        """
        old_state = self.getLeaf()
        if old_state.idx == self.term.idx:  # if leaf is the dummy terminal node
            return
        self.insertInstance(instance)
        new_state = self.getLeaf()
        new_state.addInstance(instance, self.max_hist)  # add the new instance to leaf node
        if not beginning_flag:
            # if self.history[-2].action == 5:
            #     print "catch goal"
            #     old_state.updateModel(new_state=old_state.idx, action=self.history[-2].action,
            #                           home_reward=self.history[-2].reward[0], away_reward=self.history[-2].reward[
            #             1], home_identifier=self.history[-2].home_identifier)
            # else:
            old_state.updateModel(new_state=new_state.idx, action=self.history[-2].action,
                                  reward=self.history[-2].reward)
        else:
            print "New game update begins"

    def sweepLeaves(self):
        return self.sweepRecursive(self.root, self.gamma)

    def getBestAction(self,currentObs, actions):
        """
        :return: the best action corresponding to Q value
        """
        # node = self.getLeaf()
        q_values = []
        for action in actions:
            instance_test = Instance(-100, currentObs, action, currentObs,
                                           None)
            node = self.getAbsInstanceLeaf(instance_test)
            q_values.append(node.qValues[action])
        # return random.choice(np.where(q_values== max(q_values))[0])
        return q_values.index(max(q_values))

    def sweepRecursive(self, node, gamma):
        """
        Apply single step of value iteration to leaf node
        or recursively to children if it is a split node
        :param node: target node
        :param gamma: gamma in dynamic programming
        :return:
        """
        tot = 0
        if node.nodeType == NodeLeaf:
            for action, reward in enumerate(node.rewards):
                c = float(node.count[action])  # action count
                if c == 0: continue

                exp = 0
                for node_to, t in node.transitions[action].items():

                    if node.idx == node_to:
                        exp += reward[node_to] / c
                    else:
                        exp += gamma * self.nodes[node_to].utility() * t / c + reward[node_to] / c

                if exp < -1:
                    ValueError("exp too small")
                node.qValues[action] = exp
            return 1

        assert node.nodeType == NodeSplit

        for c in node.children:
            tot += self.sweepRecursive(c, gamma)
        return tot

    def insertInstance(self, instance):
        """
        append new instance to history
        :param instance: current instance
        :return:
        """
        self.history.append(instance)
        # if len(self.history)>self.max_hist:
        #    self.history = self.history[1:]

    def fromcsvFile(self, filename):
        '''
        Load U-Tree structure from csv file
        :param filename: the path of file to load record
        :return:
        '''
        with open(filename, 'rb') as csvfile:
            fieldname = ['idx', 'dis', 'dis_value', 'par', 'q_home', 'q_away']
            reader = csv.reader(csvfile)
            self.node_id_count = 0
            for record in reader:
                if record[0] == fieldname[0]:  # idx determines header or not
                    continue
                if not record[4]:  # qValues determines NodeSplit or NodeLeaf
                    node = UNode(int(record[0]), NodeSplit, self.nodes[int(record[3])] if record[3] else None,
                                 self.n_actions, self.nodes[int(record[3])].depth + 1 if record[3] else 1)
                    node.distinction = Distinction(dimension=int(record[1]),
                                                   back_idx=0,
                                                   dimension_name=self.dim_names[int(record[1])],
                                                   iscontinuous=True if record[2] else False,
                                                   continuous_divide_value=float(record[2]) if record[
                                                       2] else None)  # default back_idx is 0
                else:
                    node = UNode(int(record[0]), NodeLeaf, self.nodes[int(record[3])] if record[3] else None,
                                 self.n_actions, self.nodes[int(record[3])].depth + 1 if record[3] else 1)
                    node.qValues_home = np.array(map(float, record[4][1:-1].split()))
                    node.qValues_away = np.array(map(float, record[5][1:-1].split()))
                if node.parent:
                    self.nodes[int(record[3])].children.append(node)
                if node.idx == 1:
                    self.root = node
                elif node.idx == 2:
                    self.term = node
                self.nodes[int(node.idx)] = node
                self.node_id_count += 1

    def tocsvFile(self, filename):
        '''
        Store a record of U-Tree in file, make it easier to rebuild tree
        :param filename: the path of file to store the record
        :return:
        '''

        with open(filename, 'wb') as csvfile:
            fieldname = ['idx', 'dis', 'dis_value', 'par', 'q_home', 'q_away']
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
                                     node.home_qValues,
                                     node.away_qValues])

    def nextInstance(self, instance):
        """
        get the next instance
        :param instance: current instance
        :return: the next instance
        """
        assert instance.timestep + 1 < len(self.history)
        return self.history[instance.timestep + 1]

    # def transFromInstances(self, node, n_id, action):
    #     """
    #     compute transition probability from current node to n_id node when perform action
    #     Formula (7) in U tree paper
    #     :param node: current node
    #     :param n_id: target node
    #     :param action: action to perform
    #     :return: transition probability
    #     """
    #
    #     count = 0
    #     total = 0
    #
    #     for inst in node.instances:
    #         if inst.action == action:
    #             # leaf_to = getInstanceLeaf(ninst, previous=0) // TODO:origin code syntax wrong
    #             leaf_to = self.getInstanceLeaf(inst, previous=0)
    #             if leaf_to.idx == n_id:
    #                 # c += 1 // TODO:origin code syntax wrong
    #                 count += 1
    #
    #             total += 1
    #
    #     if total:
    #         return count / total
    #     else:
    #         return 0

    # def rewardFromInstances(self, node, action):
    #     """
    #     compute reward of perform action on current node
    #     Formula (6) in U tree paper
    #     :param node: current node
    #     :param action: action to perform
    #     :return: reward computed
    #     """
    #
    #     rtotal = 0
    #     total = 0
    #
    #     for inst in node.instances:
    #         if inst.action == action:
    #             # rew_total += inst.reward // TODO:origin code syntax wrong
    #             rtotal += inst.reward
    #             total += 1
    #     if total:
    #         # return rew_total / total // TODO:origin code syntax wrong
    #         return rtotal / total
    #     else:
    #         return 0

    def modelFromInstances(self, node):
        """
        rebuild model for leaf node, with newly added instance
        :param node:
        :return:
        """

        assert node.nodeType == NodeLeaf

        node.rewards = [{} for i in range(self.n_actions)]  # r(s, a, s')
        node.count = np.zeros(self.n_actions)  # re-initialize count
        node.transitions = [{} for i in range(self.n_actions)]  # re-initialize transition

        for inst in node.instances:
            leaf_from = self.getInstanceLeaf(inst)
            assert leaf_from == node  # make sure from leaf is current node
            leaf_to = self.getInstanceLeaf(inst, previous=1)  # get the to node
            if leaf_to == "exceed":
                continue
            if inst.nextObs[0] == -1:
                node.updateModel(new_state=leaf_from.idx, action=inst.action,
                                 reward=inst.reward)
            else:
                node.updateModel(new_state=leaf_to.idx, action=inst.action,
                                 reward=inst.reward)  # update the node, add action reward, action count and transition states

    def getLeaf(self):
        """ Get leaf corresponding to current history """
        if len(self.history) == 0:
            return self.root

        idx = len(self.history) - 1

        if idx < 0:
            return self.root

        node = self.root
        if self.history[idx].currentObs[0] == -1:
            # print "terminal state"
            return self.term

        while node.nodeType != NodeLeaf:
            assert node.nodeType == NodeSplit
            child = node.applyDistinction(self.history, idx)  # ??? how do it get its child?
            node = node.children[child]  # go the children node
        return node

    def getAbsInstanceLeaf(self, inst, ntype=NodeLeaf):
        node = self.root
        while node.nodeType != ntype:
            child = node.applyInstanceDistinction(inst)
            node = node.children[child]

        return node
    # def getInstanceLeaf(self, inst, ntype=NodeLeaf, previous=1, check_term=True):
    #     """
    #     Get leaf that inst records a transition from
    #     previous=0 indicates transition_to, previous=1 indicates transition_from
    #     :param inst: target instance
    #     :param ntype: target node type
    #     :param previous: how many steps back
    #     :return:
    #     """
    #
    #     idx = inst.timestep
    #
    #     if len(self.history) > 0 and self.history[idx].currentObs[0] == -1 and idx >= 0:
    #         # print "terminal instance"
    #         return self.term
    #
    #     node = self.root
    #     while node.nodeType != ntype:  # iteratively find children
    #         if previous == 1:
    #             child = node.applyDistinction(self.history,
    #                                           idx)  # keep applying node's distinction until we find ntype node, where the instance should belong
    #         else:
    #             child = node.applyDistinction(self.history,
    #                                           idx)  # keep applying node's distinction until we find ntype node, where the instance should belong
    #         try:
    #             node = node.children[child]
    #         except:
    #             print node.idx
    #             print inst.timestep
    #             break
    #
    #     return node

    def getAbsInstanceLeaf(self, inst, ntype=NodeLeaf):

        node = self.root
        while node.nodeType != ntype:  # iteratively find children
            # keep applying node's distinction until we find ntype node, where the instance should belong
            try:
                child = node.applyInstanceDistinction(inst)
            except:
                ValueError("stop")
                print "ntype is {0}".format(ntype)
                print "nodeType is {0}".format(node.nodeType)
                return node

            node = node.children[child]

        return node

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
            if inst.nextObs[0] == -1:
                idx = inst.timestep
            if idx >= len(self.history):
                return "exceed"

        node = self.root
        while node.nodeType != ntype:  # iteratively find children
            # keep applying node's distinction until we find ntype node, where the instance should belong
            try:
                child = node.applyDistinction(self.history, idx)
            except:
                ValueError("stop")
                print "ntype is {0}".format(ntype)
                print "nodeType is {0}".format(node.nodeType)
                return node

            node = node.children[child]

        return node

    def genId(self):
        """
        :return: a new ID for node
        """
        self.node_id_count += 1
        return self.node_id_count

    # def split(self, node, distinction):
    #     """
    #     split decision tree on nodes
    #     :param node: node to split
    #     :param distinction: ???
    #     :return:
    #     """
    #     assert node.nodeType == NodeLeaf
    #     assert distinction.back_idx >= 0
    #
    #     node.nodeType = NodeSplit
    #     node.distinction = distinction
    #
    #     # Add children
    #     if distinction.dimension == ActionDimension:
    #         for i in range(self.n_actions):
    #             idx = self.genId()
    #             n = UNode(idx, NodeLeaf, node, self.n_actions)
    #             n.qValues = np.copy(node.qValues)
    #             self.nodes[idx] = n
    #             node.children.append(n)
    #
    #     else:
    #         for i in range(self.dim_sizes[distinction.dimension]):
    #             idx = self.genId()
    #             n = UNode(idx, NodeLeaf, node, self.n_actions)
    #             n.qValues = np.copy(node.qValues)
    #             self.nodes[idx] = n
    #             node.children.append(n)
    #
    #     # Add instances to children
    #     for inst in node.instances:
    #         n = self.getInstanceLeaf(inst)
    #         assert n.parent.idx == node.idx, "node={}, par={}, n={}".format(node.idx, n.parent.idx, n.idx)
    #         n.addInstance(inst, self.max_hist)
    #
    #     # Re-build model for all nodes
    #     for i, n in self.nodes.items():  # why rebuild model? because we have added new instances
    #         if n.nodeType == NodeLeaf:
    #             self.modelFromInstances(n)
    #
    #     # Update Q-values for children
    #     for n in node.children:
    #         self.sweepRecursive(n, self.gamma)

    def print_tree(self, file_directory='tree-structure.txt'):
        """
        print U tree
        :return:
        """
        if not os.path.isfile(file_directory):
            f = open(file_directory, 'w')
            f.close()

        self.print_tree_recursive("", self.root, file_directory=file_directory)

    def print_tree_recursive(self, blank, node, file_directory):
        """
        recursively print tree from root to leaves
        :param node: the node to be expand
        :return:
        """

        if node.nodeType == NodeSplit:
            with open(file_directory, 'a') as f:
                print >> f, blank + "idx={}, dis={}, par={}".format(node.idx,
                                                                    node.distinction.dimension,
                                                                    node.parent.idx if node.parent else None)

            for child in node.children:
                self.print_tree_recursive(blank + " ", child, file_directory)
        else:
            with open(file_directory, 'a') as f:
                print >> f, blank + 'idx={}'.format(node.idx)
                print >> f, blank + blank + 'r={}'.format(node.rewards)
                print >> f, blank + blank + 't={}'.format(node.transitions)
                print >> f, blank + blank + 'q={}'.format(['{0:.2f}'.format(q) for q in node.qValues])
                print >> f, blank + blank + 'par={}'.format(node.parent.idx if node.parent else None)

    def split(self, node, distinction):
        """
        split decision tree on nodes
        :param node: node to split
        :param distinction: distinction to split
        :return:
        """
        assert node.nodeType == NodeLeaf
        assert distinction.back_idx >= 0

        node.nodeType = NodeSplit
        node.distinction = distinction

        # Add children
        if distinction.dimension == ActionDimension:
            for i in range(self.n_actions):
                idx = self.genId()
                n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
                # n.qValues = np.copy(node.qValues)  # is it right to copy?
                self.nodes[idx] = n
                node.children.append(n)

        elif not distinction.iscontinuous:
            for i in range(self.dim_sizes[distinction.dimension]):
                idx = self.genId()
                n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
                # n.qValues = np.copy(node.qValues)
                self.nodes[idx] = n
                node.children.append(n)
        else:
            for i in range(2):
                idx = self.genId()
                n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
                # n.qValues = np.copy(node.qValues)
                self.nodes[idx] = n
                node.children.append(n)

        # Add instances to children
        for inst in node.instances:
            n = self.getInstanceLeaf(inst)
            assert n.parent.idx == node.idx, "node={}, par={}, n={}".format(node.idx, n.parent.idx, n.idx)
            n.addInstance(inst, self.max_hist)

        # Re-build model for all nodes
        # Rebuild is essential, but maybe should only rebuild all when we build really ill-conditioned model!
        # for i, n in self.nodes.items():
        #     if n.nodeType == NodeLeaf:
        #         self.modelFromInstances(n)
        for n in node.children:
            self.modelFromInstances(n)

        # update Q-values for children
        for n in node.children:
            self.sweepRecursive(n, self.gamma)

    def splitToFringe(self, node, distinction):
        """
        Create fringe nodes instead of leaf nodes after splitting; these nodes
        aren't used in the agent's model
        :param node: node to split
        :param distinction: distinction used for splitting
        :return: None
        """
        assert distinction.back_idx >= 0

        node.distinction = distinction

        # Add children
        if distinction.dimension == ActionDimension:  # ActionDimension = -1, means use action to split
            for i in range(self.n_actions):
                idx = self.genId()  # generate new id for new node
                fringe_node = UNode(idx, NodeFringe, node,
                                    self.n_actions,
                                    node.depth + 1)  # idx = idx, nodeType = NodeFringe, parent = node, n_actions = self.n_actions
                node.children.append(fringe_node)  # append new children to node
        elif not distinction.iscontinuous:
            for i in range(self.dim_sizes[distinction.dimension]):
                idx = self.genId()
                fringe_node = UNode(idx, NodeFringe, node, self.n_actions, node.depth + 1)
                node.children.append(fringe_node)
        else:
            for i in range(2):
                idx = self.genId()
                fringe_node = UNode(idx, NodeFringe, node, self.n_actions, node.depth + 1)
                node.children.append(fringe_node)

        # Add instances to children
        for inst in node.instances:
            # print node.idx
            n = self.getInstanceLeaf(inst, ntype=NodeFringe)  # Get fringe node that inst records a transition from
            assert n.parent.idx == node.idx, "idx={}".format(n.idx)
            n.addInstance(inst, self.max_hist)  # add instance to children

    def unsplit(self, node):
        """
        Undo split operation; can delete leaf or fringe nodes.
        """
        if node.nodeType == NodeSplit:
            assert len(node.children) > 0

            node.nodeType = NodeLeaf
            node.distinction = None

            for c in node.children:
                del self.nodes[c.idx]

            # Re-build model for all nodes
            for i, n in self.nodes.items():
                if n.nodeType == NodeLeaf:
                    self.modelFromInstances(n)

        node.children = []
        node.distinction = None

    def testFringe(self):
        """
        Tests fringe nodes for viable splits, splits nodes if they're found
        :return:
        """
        return self.testFringeRecursive(self.root)  # starting from root
        # TODO: count it takes how many times before it return

    def testFringeRecursive(self, node):
        """
        recursively perform test in fringe, until return total number of split
        :param node: node to test
        :return:
        """

        if len(node.instances) < self.minSplitInstances:  # haven't reach split criterion
            return 0

        if node.nodeType == NodeLeaf or node.nodeType == NodeFringe:  # NodeSplit = 0 NodeLeaf = 1 NodeFringe = 2

            if self.pre_split_flag:
                print 'pre-split on action'
                self.pre_split_flag = False
                d = Distinction(dimension=-1, back_idx=0, iscontinuous=False,
                                dimension_name='actions')
            else:
                d = self.getUtileDistinction_KS(node)  # KS test is performed here
            # d = self.getUtileDistinction_MSE(node)  # MSE test is performed here

            if d:  # if find distinction, use distinction to split
                self.split(node, d)
                return 1 + self.testFringeRecursive(node)
            else:
                return 0

        assert node.nodeType == NodeSplit
        total = 0

        for c in node.children:
            if c.depth >= self.max_tree_depth:
                print "exceeding max depth, stop split"
                return 0
            else:
                total += self.testFringeRecursive(c)

        return total

    def getUtileDistinction_KS(self, node, p_significanceLevel=float(0.01)):
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

        root_utils = self.getEFDRs(node)  # Get all expected future discounted returns for all instances in a node

        cds = self.getCandidateDistinctions(node)  # Get all the candidate distinctions

        p_min = float(1)
        cd_min = None

        for cd in cds:  # test all possible distinctions until find the one satisfy KS test
            child_utils = []
            self.splitToFringe(node, cd)  # split to fringe node with split candidate
            # self.split(node,cd)
            for c in node.children:
                if len(c.instances) < self.minSplitInstances:  # if not enough instance in a node, stop split
                    continue
                child_utils.append(
                    self.getEFDRs(c))  # Get all expected future discounted returns for all instances in a children
            # self.unsplit(node)  # delete split fringe node

            for i, cu in enumerate(child_utils):
                k, p = ks_2samp(root_utils, cu)
                # print p
                if p < p_significanceLevel and p < p_min:  # significance_level=0.00005, if p below it, this distinction is significant
                    p_min = p
                    cd_min = cd

                    if len(root_utils) == len(cu):
                        print 'find you'

                    print("KS passed, p={}, d = {}, back={}".format(p, cd.dimension, cd.back_idx))
                    # print(root_utils)
                    # print(cu)
                    # elif p< 0.1:
                    #    print("KS failed, p={}. d= {}, back={}".format(p,cd.dimension,cd.back_idx))
            self.unsplit(node)  # delete split fringe node

        if cd_min:
            print("Will be split, p={}, d={}, back={}".format(p_min, cd_min.dimension_name, cd_min.back_idx))
            # else:
            # print "Not found"
            return cd_min
        else:
            return None

    # def getUtileDistinction_MSE(self, node):
    #     assert node.nodeType == NodeLeaf
    #     root_utils = self.getEFDRs(node)  # Get all expected future discounted returns for all instances in a node
    #     # calculate MSE of root
    #     root_predict = sum(root_utils) / len(root_utils)
    #     root_mse = sum([(root_val - root_predict) ** 2 for root_val in root_utils]) / len(root_utils)
    #
    #     # get the best distinction
    #     dist_min = self.significanceLevel / len(root_utils) ** 2
    #     cd_min = None
    #
    #     cds = self.getCandidateDistinctions(node)  # Get all the candidate distinctions
    #
    #     for cd in cds:  # test all possible distinctions until find the one satisfy KS test
    #         self.splitToFringe(node, cd)  # split to fringe node with split candidate
    #         # record
    #         stop_test_flag = False
    #         child_mse = []
    #         for c in node.children:
    #             if cd.dimension == ActionDimension and len(c.instances) < 1:  # test for action
    #                 break
    #             if len(c.instances) < self.minSplitInstances and cd.dimension != ActionDimension:  # test for others
    #                 stop_test_flag = True
    #                 break
    #             # Get all expected future discounted returns for all instances in a children
    #             child_util = self.getEFDRs(c)
    #             # calculate MSE(weighted) of child
    #
    #             child_predict = sum(child_util) / len(child_util)
    #             child_weight = float(len(child_util)) / len(root_utils)
    #             child_mse.append(sum([(child_val - child_predict) ** 2 for child_val in child_util])
    #                              / len(child_util) * child_weight)
    #         self.unsplit(node)  # delete split fringe node
    #         # if not enough instance in a node, stop split
    #         if stop_test_flag:
    #             continue
    #
    #         # calculate difference between parents and students
    #         p = abs(sum(child_mse) - root_mse)
    #         if p > dist_min:
    #             print("MSE passed, p={}, d={}, back={}".format(p, cd.dimension_name, cd.back_idx))
    #             dist_min = p
    #             cd_min = cd
    #     if cd_min:
    #         print("Will be split, p={}, d={}, back={}".format(dist_min, cd_min.dimension_name, cd_min.back_idx))
    #         # else:
    #         # print "Not found"
    #     return cd_min

    def getEFDRs(self, node):
        """
        Get all expected future discounted returns for all instances in a node
        (q-value is just the average EFDRs)
        """
        efdrs = np.zeros(len(node.instances))
        for i, inst in enumerate(node.instances):
            efdrs[i] = inst.reward
            # next_state = self.getInstanceLeaf(inst, previous=1)  # Get leaf that inst records a transition to
            # if next_state == "exceed":
            #     efdrs[i] = inst.reward
            # else:
            #     if inst.action == 5:
            #         efdrs[i] = inst.reward
            #     else:
            #         if node.parent is None:
            #             efdrs[i] = inst.reward
            #         elif node.parent.idx == next_state.idx:
            #             efdrs[i] = inst.reward
            #         else:
            #             next_state_util = next_state.utility()  # maximum Q value
            #             efdrs[i] = inst.reward + self.gamma * next_state_util  # r + gamma * maxQ

        return efdrs

    def print_tree_structure(self, file_directory="./print_tree_record/print_oracle_tree_split_home_away.txt"):
        root = self.root
        with open(file_directory, 'w') as f:
            tree_structure = self.recursive_print_tree_structure(root, 0)

            print >> f, tree_structure

    def recursive_print_tree_structure(self, node, layer):
        tree_structure = ""
        for i in range(0, layer):
            tree_structure += "\t"
        if node.nodeType == NodeSplit:
            tree_structure += 'idx{2}(Q:{0}, distinct:{1}, value:{4}, par:{3})'.format(
                format(node.utility(), '.2f'),
                node.distinction.dimension_name,
                node.idx,
                node.parent.idx if node.parent is not None else None,
                node.distinction.continuous_divide_value if node.distinction.continuous_divide_value else 'Action'
            )
            child_string = ""
            for child in node.children:
                child_string += "\n" + self.recursive_print_tree_structure(child, layer + 1)
            tree_structure += child_string
        elif node.nodeType == NodeLeaf:
            tree_structure += 'idx{1}(Q:{0}, par{2})'.format(format(node.utility(), '.3f'),
                                                             node.idx,
                                                             node.parent.idx if node.parent is not None else None)
        else:
            raise ValueError("Unsupported tree nodeType:{0}".format(node.nodeType))
        return tree_structure


        # def getCandidateDistinctions(self, node):

    # """
    #     construct all candidate distinctions
    #     :param node: target nodes
    #     :return: all candidate distinctions
    #     """
    #
    #     p = node.parent
    #     anc_distinctions = []
    #
    #     while p:
    #         assert p.nodeType == NodeSplit
    #         anc_distinctions.append(p.distinction)
    #         p = p.parent  # append all the parent nodes' distinction to anc_distinctions list
    #
    #     candidates = []
    #     for i in range(self.max_back_depth):  # what does back_idx mean?
    #         for j in range(-1, self.n_dim):  # -1 is for actions and 0 for dimension 1?
    #
    #             if j > -1 and self.dim_sizes[j] == 'continuous':
    #                 d = "continuous division"
    #                 # TODO: implement CART division
    #             else:
    #
    #                 d = Distinction(dimension=j, back_idx=i,
    #                                 dimension_name=self.dim_names[j] if j > -1 else 'actions')  # j = dimension, i = back_idx
    #                 if d in anc_distinctions:
    #                     continue  # if new built distinction belongs to parent, jump it, we don't need duplicate distinction
    #                 candidates.append(d)
    #
    #     return candidates

    def getHistoryInstance(self, node, back_idx):
        """
        get all instances in node <back_idx> steps before
        :param node:
        :return:
        """
        historyinst = []
        for instance in node.instances:
            idx = instance.timestep - back_idx
            historyinst.append(self.history[idx])
        return historyinst

    def judge_distinction(self, distinct2j, anc_distinctions):
        for distinct in anc_distinctions:
            if distinct.continuous_divide_value == distinct2j.continuous_divide_value and distinct.dimension_name == distinct2j.dimension_name:
                return True
        return False


    def getCandidateDistinctions(self, node, select_interval=100):
        """
        construct all candidate distinctions
        :param select_interval:
        :param node: target nodes
        :return: all candidate distinctions
        """

        p = node.parent
        anc_distinctions = []

        while p:
            assert p.nodeType == NodeSplit
            anc_distinctions.append(p.distinction)
            p = p.parent  # append all the parent nodes' distinction to anc_distinctions list

        candidates = []
        history_instance = []  # history instances for instances in the node some time steps before
        for i in range(self.max_back_depth):
            history_instance = self.getHistoryInstance(node, i)

            for j in range(-1, self.n_dim):  # -1 is for actions and 0 for dimension 1?
                if j > -1 and self.dim_sizes[j] == 'continuous':
                    # d = "continuous division"

                    instance_values = []

                    for inst in history_instance:
                        # if inst.currentObs[j] in distinctions_values:  # duplicate split criterion
                        #     continue
                        # else:
                        instance_values.append(inst.currentObs[j])

                    instance_values.sort()

                    for instance_value_index in range(0, len(instance_values)):

                        if instance_value_index % select_interval == 0:
                            d = Distinction(dimension=j, back_idx=i,
                                            continuous_divide_value=instance_values[instance_value_index],
                                            iscontinuous=True,
                                            dimension_name=self.dim_names[j] if j > -1 else 'actions')
                            if self.judge_distinction(d, anc_distinctions):
                                continue
                            candidates.append(d)
                else:
                    d = Distinction(dimension=j, back_idx=i, iscontinuous=False,
                                    dimension_name=self.dim_names[
                                        j] if j > -1 else 'actions')  # j = dimension, i = back_idx
                    if self.judge_distinction(d, anc_distinctions):
                        continue
                    candidates.append(d)

        return candidates


class UNode:
    def __init__(self, idx, nodeType, parent, n_actions, depth):
        self.idx = idx
        self.nodeType = nodeType
        self.parent = parent
        self.depth = depth
        self.children = []

        self.rewards = [{} for i in range(n_actions)]  # r(s, a, s')
        self.count = np.zeros(n_actions)
        self.transitions = [{} for i in range(n_actions)]  # T(s, a, s')
        self.qValues = np.zeros(n_actions)

        self.instances = []

    def utility(self):
        """
        :return: maximum Q value
        """
        value = [x for x in self.qValues if x != float(0)]
        if len(value) == 0:
            return max(self.qValues)
        else:
            return max(value)

    def addInstance(self, instance, max_hist):
        """
        add new instance to node instance list
        if instance length exceed maximum history length, select most recent history
        :param instance:
        :param max_hist:
        :return:
        """
        assert (self.nodeType == NodeLeaf or self.nodeType == NodeFringe)
        self.instances.append(instance)
        if len(self.instances) > max_hist:
            self.instances = self.instances[1:]

    def updateModel(self, new_state, action, reward):
        """
        1. add action reward
        2. add action count
        3. record transition states
        :param new_state: new transition state
        :param action: new action
        :param home_reward: reward of action
        :return:
        """
        self.count[action] += 1
        if new_state not in self.transitions[action]:
            self.transitions[action][new_state] = 1  # record transition
            self.rewards[action][new_state] = reward
        else:
            self.transitions[action][new_state] += 1
            self.rewards[action][new_state] += reward

    # def applyDistinction(self, history, idx):
    #     """
    #     :param history:
    #     :param idx:
    #     :return:
    #     """
    #     assert self.nodeType != NodeFringe
    #     assert len(history) > self.distinction.back_idx
    #     assert len(history) > idx
    #     assert self.distinction.back_idx >= 0
    #
    #     if idx == -1 or idx == 0:
    #         return 0
    #
    #     # if back_idx is too far for idx, pick the first child
    #     if self.distinction.back_idx > idx:
    #         return 0
    #
    #     inst = history[
    #         idx - self.distinction.back_idx]  # find the instance from history, may back trace to former instance
    #
    #     if self.distinction.dimension == ActionDimension:
    #         return inst.action  # action distinction
    #
    #     assert self.distinction.dimension >= 0
    #
    #     return inst.observation[
    #         self.distinction.dimension]  # feature distinction, assumption: distinction.dimension indicates the feature used to split


    def applyInstanceDistinction(self, inst):

        if self.distinction.dimension == ActionDimension:
            return inst.action  # action distinction

        assert self.distinction.dimension >= 0

        if not self.distinction.iscontinuous:
            # if is_next:
            #     return inst.nextObs[  # TODO: check if is right to use the next observation
            #         self.distinction.dimension]  # feature distinction, assumption: distinction.dimension indicates the feature used to split
            # else:
            return inst.currentObs[
                self.distinction.dimension]

        else:
            # if is_next:  # the distinction is discrete
            #     if inst.nextObs[
            #         self.distinction.dimension] <= self.distinction.continuous_divide_value:  # TODO: check if is right to use the next observation
            #         return 0
            #     else:
            #         return 1
            # else:
            if inst.currentObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
                return 0
            else:
                return 1

    def applyDistinction(self, history, idx):
        """
        :param is_next: 
        :param history:
        :param idx:
        :return:
        """
        assert self.nodeType != NodeFringe
        assert len(history) > self.distinction.back_idx
        assert len(history) > idx
        assert self.distinction.back_idx >= 0

        # if idx == -1 or idx == 0:
        if idx == -1:
            return 0

        # if back_idx is too far for idx, pick the first child
        if self.distinction.back_idx > idx:
            return 0

        inst = history[idx - self.distinction.back_idx]

        if self.distinction.dimension == ActionDimension:
            return inst.action  # action distinction

        assert self.distinction.dimension >= 0

        if not self.distinction.iscontinuous:
            # if is_next:
            #     return inst.nextObs[  # TODO: check if is right to use the next observation
            #         self.distinction.dimension]  # feature distinction, assumption: distinction.dimension indicates the feature used to split
            # else:
            return inst.currentObs[
                self.distinction.dimension]

        else:
            # if is_next:  # the distinction is discrete
            #     if inst.nextObs[
            #         self.distinction.dimension] <= self.distinction.continuous_divide_value:  # TODO: check if is right to use the next observation
            #         return 0
            #     else:
            #         return 1
            # else:
            if inst.currentObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
                return 0
            else:
                return 1

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

    def __init__(self, timestep, currentObs, action, nextObs, reward):
        self.timestep = int(timestep)
        self.action = int(action)
        self.nextObs = map(float, nextObs)  # record the state data
        self.currentObs = map(float, currentObs)  # record the state data
        self.reward = reward


class Distinction:
    """
    For split node
    """

    def __init__(self, dimension, back_idx, dimension_name='unknown', iscontinuous=False, continuous_divide_value=None):
        """
        initialize distinction
        :param dimension: split of the node is based on the dimension
        :param back_idx: history index,how many time steps backward from the current time this feature will be examined
        :param dimension_name:
        :param iscontinuous:
        :param continuous_divide_value:
        """
        self.dimension = dimension
        self.back_idx = back_idx
        self.dimension_name = dimension_name
        self.iscontinuous = iscontinuous
        self.continuous_divide_value = continuous_divide_value

    def __eq__(self, distinction):
        return self.dimension == distinction.dimension and self.back_idx == distinction.back_idx
