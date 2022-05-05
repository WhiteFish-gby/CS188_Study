# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # 构造循环，根据指定的迭代次数进行反复更新
        for _ in range(self.iterations):
            # 在每一次迭代中，要计算当前策略中每一个可能的状态的评价值，
            # 所以我们先取出当前策略中的所有状态
            states = self.mdp.getStates()
            # 然后再使用旧策略的状态值去计算新策略中的状态值，
            # 为了存储新策略的V值，初始化一个空的Counter对象
            temp_counter = util.Counter()
            # 构造对所有状态的遍历
            for state in states:
                # 如果当前的状态是端点（即没有可行的动作），则Q值为0
                if len(self.mdp.getPossibleActions(state))==0:
                    maxVal = 0
                # 否则就遍历当前状态所有可行的动作，按照公式完成Q值的计算，
                # 并求其中的最大值作为当前的状态值
                else:
                    maxVal = -float('inf')
                    for action in self.mdp.getPossibleActions(state):
                        Q = self.computeQValueFromValues(state ,action)
                        if Q>maxVal:
                            maxVal = Q
                # 将得到的Q值更新到新的策略中，供下一次迭代使用
                temp_counter[state] = maxVal
            self.values = temp_counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # 实现计算Q-Value的功能
        # 基本算法就是把可能的下一步状态信息进行遍历，并求和
        total = 0
        # 按照Value Iteration算法的求值公式，进行代码编写
        for nextState,prob in self.mdp.getTransitionStatesAndProbs(state, action):
            total += prob * \
                (self.mdp.getReward(state, action, nextState) \
                 + self.discount * self.getValue(nextState))
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # 该函数实现的从策略中查询指定状态的最佳行动
        # 基本思想就是利用策略中各个状态的数据，计算指定状态中各个acion对应的Q值，取Q值最大的action
        maxVal = -float('inf')
        bestAction = None
        # 对当前状态所有的可能动作进行求Q值的遍历，从中选出最大的Q值，并返回其对应的action
        for action in self.mdp.getPossibleActions(state):
            Q = self.computeQValueFromValues(state ,action)
            if Q>maxVal:
                maxVal = Q
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # 异步值迭代是策略迭代算法的核心
        # 所谓的异步值迭代策略就是一次更新一个状态节点，而不像值迭代算法一次迭代中需要更新所有的状态
        # 保存当前MDP状态
        states = self.mdp.getStates()
        # 根据迭代次数构造相应次数的循环
        for index in range(self.iterations):
            # 首先要从states中获取一个state，下方代码中的求余数操作可以保证索引值不会超出边界
            state = states[index % len(states)]
            # 接下来只要更新这个state即可，但是题目上要求不更新Terminal节点
            if not self.mdp.isTerminal(state):
                # 按照与Value Iteration同样的方法求V值
                maxVal = -float('inf')
                for action in self.mdp.getPossibleActions(state):
                    Q = self.computeQValueFromValues(state ,action)
                    if Q>maxVal:
                        maxVal = Q
                # 这句话更新的是某个单独的节点，而非Value Iteration算法中更新所有节点的V值
                self.values[state] = maxVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # 计算所有状态的前驱节点，初始为空
        predecessors = {}
        # 为了完成上述任务，需要对MPD中的所有state进行遍历
        for state in self.mdp.getStates():
            # 终点状态没有后继节点，这也就意味着它不可能是其他节点的前驱节点，所以可以忽略
            if self.mdp.isTerminal(state):
                continue
            # 遍历状态state的所有可行的action，发现具有前驱-后继关系的节点对
            for action in self.mdp.getPossibleActions(state):
                # 调用getTransitionStatesAndProbs得到一系列节点和发生概率的组合
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    # 在predecessors保存的是前驱节点的信息，所以要以下一个节点作为键值创建字典元素
                    if nextState in predecessors:
                        predecessors[nextState].add(state)
                    else:
                        # 务必使用集合数据类型创建前驱节点容器，以防止将同一个节点多次添加到容器中
                        predecessors[nextState] = {state}
        # 初始化一个空的优先级队列
        pq = util.PriorityQueue()
        # 对所有非终点的状态s进行遍历
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            # 计算s的V值和最大Q值的差的绝对值，存入变量diff
            maxQ = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
            diff = abs(maxQ - self.values[s])
            # 把s按照-diff的值，推到优先级队列中，因为优先值越小的元素，会越先被PriorityQueue弹出来
            # 即：如果某个状态当前的V值和Q值相差很大，就会被优先进行更新
            pq.update(s, -diff)
        # 按照迭代次数构造循环
        for _ in range(self.iterations):
            # 如果优先队列为空，则终止循环
            if pq.isEmpty():
                break
            # 从队列中弹出一个状态s，从上面的步骤可知，该状态的V值和Q值相差最大
            s = pq.pop()
            # 更新状态s的V值
            if not self.mdp.isTerminal(s):
                self.values[s] = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
            # 随即遍历s的前驱节点并更新它们的状态
            for p in predecessors[s]:
                # 找到p的V值和从p计算的最大Q值的差的绝对值，存入变量diff
                maxQ = max([self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)])
                diff = abs(maxQ - self.values[p])
                # 如果diff大于theta的值，则将p以-diff的优先值放到队列中
                if diff>self.theta:
                    pq.update(p, -diff)
