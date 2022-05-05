# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # 可以使用util.Counter这个类创建一个增强版的字典，用来保存每个状态的Q值，初值为0
        # 特别注意一下，这个字典的键是二元组(state,action)
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # 如果对应state的某个action的Q值未曾计算过，则返回0
        if (state,action) not in self.q_values.keys():
            return 0
        # 初始化的时候已经约定使用self.q_values进行Q值的保存，所以直接返回对应的Q值即可
        return self.q_values[(state,action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # 对于每一个状态，先获取可行的动作集合
        legalActions = self.getLegalActions(state)
        # 建立一个临时的util.Counter对象，用来存放每个动作对应的Q值
        tmp = util.Counter()
        # 对每个action进行遍历，将其对应的Q值塞到tmp中
        for action in legalActions:
            tmp[action] = self.getQValue(state,action)
        # 最后只需将tmp中的最大值返回，即为当前状态的V值
        return tmp[tmp.argMax()]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # 对于每一个状态，先获取可行的动作集合
        legalActions = self.getLegalActions(state)
        # 建立一个临时的util.Counter对象，用来存放每个动作对应的Q值
        tmp = util.Counter()
        # 对每个action进行遍历，将其对应的Q值塞到tmp中
        for action in legalActions:
            tmp[action] = self.getQValue(state,action)
        # 最后只需将tmp中的Q值最大的元素返回，即可
        return tmp.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # 使用self.epsilon作为概率得到一个随机的action
        if util.flipCoin(self.epsilon):
            # 随机从当前状态可行的action中选一个
            action = random.choice(legalActions)
        else:
            # 从当前状态中选择Q值最大的action
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # 在更新Q值的时候需要判定是否存在nextState
        if nextState:
            qValue = (1-self.alpha)*self.getQValue(state,action) \
                     +self.alpha*(reward+self.discount*self.computeValueFromQValues(nextState))
        else:
            qValue = (1-self.aopha)*self.getQValue(state,action)+self.alpha*reward
        # 将计算完毕的qValue存储到self.q_values中
        self.q_values[(state,action)] = qValue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Approximate QLearning计算Q值的基本思想就是对特征值进行加权求和
        # 需要调用提供给我们的方法，把所有的特征值取出来
        features = self.featExtractor.getFeatures(state,action)
        # 按照权值进行矩阵的乘法，最后将乘积返回即可
        return sum([features[item]*self.weights[item] for item in features])

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # 根据幻灯片上的Approximate QLearning更新公式进行编程
        diff = (reward+self.discount*self.getValue(nextState))-self.getQValue(state,action)
        features = self.featExtractor.getFeatures(state,action)
        for item in features:
            self.weights[item] = self.weights[item]+self.alpha*diff*features[item]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # 这里可以放调试算法的输出语句
            # print(self.weights)
            pass
