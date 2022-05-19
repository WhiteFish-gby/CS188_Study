# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        # 归一化主要是为了数据处理方便提出来的，把数据映射到0～1范围之内处理，更加便捷快速。
        # 可以把归一化理解为计算某个样本在所有样本中出现的概率
        # 要进行归一化，首先需要求总体样本数量total
        total = self.total()
        # 根据注释的提醒，total为0的时候不要做任何事情
        if total!=0:
            # 如果total不为0，则进行遍历进行归一化
            for k,v in self.items():
                self[k] = v/total

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        # 所谓采样，就是根据样本出现的概率，生成单个样本
        # 根据幻灯的内容和上述注释文档，我们可以先取[0,1)之间的随机数u
        # 然后根据各个样本出现的概率，规定各个样本在[0,1)上的对应区间
        # 然后，只需判断随机数u落在哪一个区间内，即可返回对应的样本值
        u = random.random()
        # 使用变量alpha表示区间的下限
        alpha = 0.0
        for key,value in self.items():
            # 当随机数u落在了指定的离散分布区间内，返回对应的key即样本
            if alpha <= u < alpha+(value/self.total()):
                return key
            # 为了得到下一个分布区间的下限，需要不断地更新alpha
            alpha += value/self.total()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        # 本函数主要目的是计算P(noisyDistance | pacmanPosition, ghostPosition)
        # 根据题目描述，需要判定鬼怪是否已经被送到jail中
        if ghostPosition == jailPosition:
            # 该分支表示鬼怪在监狱中，即noisyDistance为None的概率为100%
            if noisyDistance == None:
                return 1.0
            # 该分支表示鬼怪在监狱中，即noisyDistance为None的概率为0%
            else:
                return 0.0
        else:
            # 该分支表示鬼怪不在监狱中，即noisyDistance为None的概率为0%
            if noisyDistance == None:
                return 0.0
            # 该分支表示鬼怪不在监狱中，返回在当前状态下，noisyDistance的概率
            else:
                trueDistance = manhattanDistance(pacmanPosition, ghostPosition)
                return busters.getObservationProbability(noisyDistance, trueDistance)


    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        # 本题的要求就是编写不断更新置信网络的函数
        # 先获取旧的置信网络数据(其本质就是一个离散分布)
        oldPD = self.beliefs
        # 获取吃豆人的位置
        pacmanPosition = gameState.getPacmanPosition()
        # 获得当前地图中监狱的位置
        jailPosition = self.getJailPosition()
        # 建立新的离散分布对象以存储新的置信网络数据
        newPD = DiscreteDistribution()
        # 遍历所有鬼怪可能出现的位置，并实现精确推理的核心步骤
        for ghostPosition in self.allPositions:
            newPD[ghostPosition] = self.getObservationProb(observation,
                                pacmanPosition, ghostPosition, jailPosition) * oldPD[ghostPosition]
        # 将新的置信网络更新到self.beliefs中
        self.beliefs = newPD
        # 最后，将置信网络进行归一化，这一步是最重要的，千万不能忘
        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        # 此题的测试用例1是随机走的Ghost，测试用例2和3是一直往下走的Ghost
        # 获取旧的置信网络数据(其实就是一个离散分布)
        oldPD = self.beliefs
        # 建立新的离散分布对象以存储新的置信网络数据
        newPD = DiscreteDistribution()
        # 遍历所有鬼怪所有可能出现的位置，将每一个可能出现的位置都作为初始位置进行计算
        for oldPos in self.allPositions:
            # 根据Time Elapse算法内容，我们得到接下来鬼怪可能存在位置的离散分布
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            # 新的置信网络中的数据需要将所有从oldPos出发达到newPos的可能性进行累加
            for newPos in self.allPositions:
                # 考虑到从oldPos到达newPos的概率可能为0，我们可以在这里做一些优化
                if newPosDist[newPos]>0:
                    newPD[newPos] += newPosDist[newPos]*oldPD[oldPos]
        # 将新的置信网络更新到self.beliefs中
        self.beliefs = newPD
        # 最后，将置信网络进行归一化
        self.beliefs.normalize()

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # 所谓样本的初始化，就是将一堆样本放到给定的空间中
        # 假设空间范围为[1,10]，现在有1000个样本，如果平均分布，那么1到10各取100次就是这个采样分布，形如：
        # [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,……,7,8,9,10,1,2,3,4,5,6,7,8,9,10]
        # 结合当前的实际问题，鬼怪可能存在的所有位置就是此题的样本空间
        self.particles += self.legalPositions*(self.numParticles//len(self.legalPositions))
        # 还要考虑特殊情况：样本总数不能整除空间尺寸，比如空间范围为[1,10]，但是样本数量为95
        self.particles += self.legalPositions[:(self.numParticles%len(self.legalPositions))]

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # 获取吃豆人的位置
        pacmanPosition = gameState.getPacmanPosition()
        # 获得当前地图中监狱的位置
        jailPosition = self.getJailPosition()
        # 建立新的离散分布对象以存储新的置信网络数据
        newPD = DiscreteDistribution()
        # 本算法通过观察样本出现的概率，更新置信网络中的数据
        # 与精确推理不同，计算中不需要用到oldPD，即旧的置信网络的值
        for ghostPosition in self.particles:
            newPD[ghostPosition] += \
                self.getObservationProb(observation, pacmanPosition, ghostPosition, jailPosition)
        # 根据注释的提示，需要考虑特殊情况，即置所有样本的概率总和为0，就调用初始化方法
        if newPD.total()==0:
            self.initializeUniformly(gameState)
        else:
            # 最后，将置信网络进行归一化
            newPD.normalize()
            # 再次生成新的样本(因为置信网络中的数据发生了变化，所以必须重新采样，以更新置信网络)
            self.particles = [newPD.sample() for _ in range(self.numParticles)]

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        PosDist = {}
        # 基本思想就是将下一步鬼怪可能出现位置作为样本按照给定的离散分布规律放在self.particles中
        # 与之前一样，此题的测试用例1是随机走的Ghost，测试用例2和3是一直往下走的Ghost
        # 更新样本数据的方法，就是从旧的样本数据中衍生出新的样本数据
        for index,particle in enumerate(self.particles):
            # 在项目说明中提示我们要降低对self.getPositionDistribution的调用次数
            # 为了达到上述目的，我们预先设置一个以particle为键的字典，值为该particle对应的PositionDistribution
            if particle not in PosDist.keys():
                newPosDist = self.getPositionDistribution(gameState, particle)
                PosDist[particle] = newPosDist
            # 接着，我们就可以直接使用字典PosDist中的已经计算好的PositionDistribution数据进行采样
            newParticle = PosDist[particle].sample()
            # 最后用采样出来的数据代替原始的样本
            self.particles[index] = newParticle

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        # 构造离散分布对象用于存放概率分布
        belief = DiscreteDistribution()
        # 接着只需要对以样本为键的字典对象进行+1计数，即可得到每一个样本出现的次数
        for particle in self.particles:
            belief[particle] += 1
        # 最后，千万不要忘记归一化，并返回
        belief.normalize()
        # 注意，不要返回上面这句话，归一化函数没有返回值
        return belief

class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # 所谓联合采样，就是采样空间中的数据，不再是一个单独的值，而是若干数据的组合
        # 假设有4个鬼怪，则采样数据中的每一个样本都将会是一个四元组
        # itertools.product的第一个参数表示取值范围，参数repeat表示元组中的元素数量
        permutations = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        # 题目要求打乱上述排列中的内容，以得到一个乱序的采样空间
        random.shuffle(permutations)

        # 依然按照Particle Filter算法的方法进行初始化，从给定的采样空间permutations中选取样本
        self.particles += permutations*(self.numParticles//len(permutations))
        # 依然要考虑特殊情况：样本总数不能整除空间尺寸，比如空间范围为[1,10]，但是样本数量为95
        self.particles += permutations[:(self.numParticles%len(permutations))]

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # 算法思想与Particle Filter类似，只不过概率的计算方法要考虑到多个鬼怪的组合
        # 获取吃豆人的位置
        pacmanPosition = gameState.getPacmanPosition()
        # 建立新的离散分布对象以存储新的置信网络数据
        newPD = DiscreteDistribution()
        # 本算法通过观察样本的概率，以更新置信网络中的数据
        for ghostPositions in self.particles:
            # 利用累乘算法，求出4个鬼怪均出现在样本所指位置的概率
            prob = 1
            for i in range(self.numGhosts):
                prob *= self.getObservationProb(observation[i], pacmanPosition, \
                                                ghostPositions[i], self.getJailPosition(i))
            newPD[ghostPositions] += prob
        # 根据注释的提示，需要考虑特殊情况，即置所有样本的概率总和为0，就调用初始化方法
        if newPD.total()==0:
            self.initializeUniformly(gameState)
        else:
            # 最后，将置信网络进行归一化
            newPD.normalize()
            # 再次生成新的样本(因为置信网络中的数据发生了变化，所以必须重新采样，以计算新的置信网络)
            self.particles = [newPD.sample() for _ in range(self.numParticles)]

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            # 基本思想就是将下一步鬼怪可能出现位置作为样本，按照给定的离散分布规律放在self.particles中
            # 与之前Particle Filter算法不一样的是，此处的样本是一个包含若干鬼怪位置的元组
            for i in range(self.numGhosts):
                newPosDist = self.getPositionDistribution(gameState, newParticle, i, self.ghostAgents[i])
                newParticle[i] = newPosDist.sample()
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
