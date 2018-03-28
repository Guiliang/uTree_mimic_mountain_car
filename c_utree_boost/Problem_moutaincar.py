# uncompyle6 version 2.14.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.4.3 (default, Nov 17 2016, 01:08:31) 
# [GCC 4.8.4]
# Embedded file name: /Local-Scratch/PycharmProjects/Sport-Analytic-U-Tree/continuous-U-Tree-ice-hockey/c_utree_oracle/Problem_moutaincar_control.py
# Compiled at: 2017-12-04 13:56:08
from datetime import datetime


class MoutainCar:
    """
    An MDP. Contains methods for initialisation, state transition.
    Can be aggregated or unaggregated.
    """

    def __init__(self, games_directory = '../save_all_transition/', gamma=1):
        assert games_directory is not None
        self.games_directory = games_directory
        self.actions = {'push_left': 0,
                        'no_push': 1,
                        'push_right': 2
                        }
        self.stateFeatures = {'position': 'continuous', '	velocity': 'continuous'}
        self.gamma = gamma
        self.reset = None
        self.isEpisodic = True
        self.nStates = len(self.stateFeatures)
        self.dimNames = ['position', 'velocity']
        self.dimSizes = ['continuous', 'continuous']
        d = datetime.today().strftime('%d-%m-%Y--%H:%M:%S')
        self.probName = ('{0}_gamma={1}_mode={2}').format(d, gamma,
                                                          'Action Feature States' if self.nStates > 12 else 'Feature States')
        self.games_directory = games_directory
        return
