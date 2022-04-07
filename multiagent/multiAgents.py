# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # 所谓Reflex Agent，就是根据对当前环境的感知，作出相应的行动，所以我们需要量化当前环境中的鬼怪和食物两种因素
        # 用正数表示食物带来的的正反馈，用负数表示鬼怪带来的的负反馈

        # 首先优先考虑吃掉最近的豆豆，即如果豆豆还没有吃光，用最近的豆豆的坐标计算出一个启发值
        # 这个程序从字面上看，是用曼哈顿距离计算启发值，所以如果吃豆人和豆豆之间有墙的话……吃豆人就卡在墙后面了
        # 但是，又因为有鬼怪的存在，只要鬼怪靠近，就会驱动吃豆人离开卡死在墙后面的状态，勉强算是通过测试了
        if len(newFood.asList()) > 0:
            nearestFood = (min([manhattanDistance(newPos, food)
                           for food in newFood.asList()]))
            # 为什么启发值是“9/距离”呢？因为按照我的设计，表示食物的启发值一定为正数
            # 同时，吃到隔壁的豆豆可以得9分(移动1格需要扣1分)，且上式满足距离越远启发值越小的特征
            # 举例说明：如果下一个豆豆就在隔壁，距离为1，那么启发值为9，如果距离为2，那么启发值为4.5
            foodScore = 9/nearestFood
        else:
            foodScore = 0

        # 找出最近的鬼怪，计算负反馈
        nearestGhost = min([manhattanDistance(
            newPos, ghostState.configuration.pos) for ghostState in newGhostStates])
        # 与计算食物的启发值的方法类似，我们用最近的鬼怪计算负反馈，距离吃豆人越远其值越接近零，表示影响越小
        # 为什么分子取-10？因为根据食物计算的启发值总是在10以内，只要两者相加为负数，就可以抵消豆豆对吃豆人的诱惑，起到让吃豆人远离鬼怪的作用
        # 举例说明：如果下一个豆豆和鬼怪都在隔壁，根据上述算法，食物的启发值为9，鬼怪的启发值为-10，两者的和为-1，吃豆人拒绝往这个方向行动
        dangerScore = -10/nearestGhost if nearestGhost != 0 else 0

        # 把计算好的各种启发值加在游戏得分上，并返回
        return successorGameState.getScore() + foodScore + dangerScore


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # 基本算法：从吃豆人开始，遍历所有可行的下一步，取效用最佳的action为bestAction
        maxVal = -float('inf')
        bestAction = None
        for action in gameState.getLegalActions(agentIndex=0):
            # 求出后续状态的评价值，并和maxVal比较，求出MAX值
            value = self.getValue(gameState.generateSuccessor(
                agentIndex=0, action=action), agentIndex=1, depth=0)
            # 如果当前的value比maxVal还要大，更新maxVal值，并记下bestAction
            if value is not None and value > maxVal:
                maxVal = value
                bestAction = action
        # 最后返回最佳选择
        return bestAction

    def getValue(self, gameState, agentIndex, depth):
        # 如果当前状态是一个终点状态（没有可行的下一步），则返回当前状态的评价值
        legalActions = gameState.getLegalActions(agentIndex)
        if len(legalActions) == 0:
            return self.evaluationFunction(gameState)
        # 根据agentIndex的值，遍历下一个状态的评价值
        # 如果下一个是吃豆人的行动使用max_value()函数，如果是鬼怪的行动使用min_value()函数
        if agentIndex == 0:
            # 如果agentIndex为0，表示所有的agent都已经轮询过一遍，此时depth加1
            depth += 1
            # 如果depth的值到达搜索深度限制，则返回当前状态的评价值，不再继续向下搜索
            if depth == self.depth:
                return self.evaluationFunction(gameState)
            else:
                return self.max_value(gameState, agentIndex, depth)
        elif agentIndex > 0:
            return self.min_value(gameState, agentIndex, depth)

    def max_value(self, gameState, agentIndex, depth):
        # 初始化v等于负无穷
        maxVal = -float('inf')
        # 获得当前Agent所有合法的下一步动作
        legalActions = gameState.getLegalActions(agentIndex)
        # 通过对合法动作遍历，轮询所有的下一个状态，取所有评价值和v的最大值
        for action in legalActions:
            value = self.getValue(gameState.generateSuccessor(
                agentIndex, action), (agentIndex+1) % gameState.getNumAgents(), depth)
            if value is not None and value > maxVal:
                maxVal = value
        return maxVal

    def min_value(self, gameState, agentIndex, depth):
        # 初始化v等于正无穷
        minVal = float('inf')
        # 获得当前Agent所有合法的下一步动作
        legalActions = gameState.getLegalActions(agentIndex)
        # 通过对合法动作遍历，轮询所有的下一个状态，取所有评价值和v的最小值
        for action in legalActions:
            value = self.getValue(gameState.generateSuccessor(
                agentIndex, action), (agentIndex+1) % gameState.getNumAgents(), depth)
            if value is not None and value < minVal:
                minVal = value
        return minVal


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # 基本算法：从吃豆人开始，遍历所有可行的下一步，取效用最佳的action为bestAction，这次是带剪枝的版本哟
        alpha = -float('inf')
        beta = float('inf')
        maxVal = -float('inf')
        bestAction = None
        for action in gameState.getLegalActions(agentIndex=0):
            # 求出后续状态的评价值，并和maxVal比较，求出MAX值
            value = self.getValue(gameState.generateSuccessor(agentIndex=0, action=action),
                                  agentIndex=1, depth=0, alpha=alpha, beta=beta)
            # 如果当前的value比maxVal还要大，更新maxVal值，并记下bestAction
            if value is not None and value > maxVal:
                maxVal = value
                bestAction = action
            # 按照α-β剪枝算法，这里还需要更新α的值
            if maxVal > alpha:
                alpha = maxVal
        # 最后返回最佳选择
        return bestAction

    def getValue(self, gameState, agentIndex, depth, alpha, beta):
        # 如果当前状态是一个终点状态（没有可行的下一步），则返回当前状态的评价值
        legalActions = gameState.getLegalActions(agentIndex)
        if len(legalActions) == 0:
            return self.evaluationFunction(gameState)
        # 根据agentIndex的值，遍历下一个状态的评价值
        # 如果下一个是吃豆人的行动使用max_value()函数，如果是鬼怪的行动使用min_value()函数
        if agentIndex == 0:
            # 如果agentIndex为0，表示所有的agent都已经轮询过一遍，此时depth加1
            depth += 1
            # 如果depth的值到达搜索深度限制，则返回当前状态的评价值，不再继续向下搜索
            if depth == self.depth:
                return self.evaluationFunction(gameState)
            else:
                return self.max_value(gameState, agentIndex, depth, alpha, beta)
        elif agentIndex > 0:
            return self.min_value(gameState, agentIndex, depth, alpha, beta)

    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        # 初始化v等于负无穷
        maxVal = -float('inf')
        # 获得当前Agent所有合法的下一步动作
        legalActions = gameState.getLegalActions(agentIndex)
        # 通过对合法动作遍历，轮询所有的下一个状态，取所有评价值和v的最大值
        for action in legalActions:
            value = self.getValue(gameState.generateSuccessor(agentIndex, action),
                                  (agentIndex+1) % gameState.getNumAgents(), depth, alpha, beta)
            if value is not None and value > maxVal:
                maxVal = value
            # 按照α-β剪枝算法，如果v>β，则直接返回maxVal
            if maxVal > beta:
                return maxVal
            # 按照α-β剪枝算法，这里还需要更新α的值
            if maxVal > alpha:
                alpha = maxVal
        return maxVal

    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        # 初始化v等于正无穷
        minVal = float('inf')
        # 获得当前Agent所有合法的下一步动作
        legalActions = gameState.getLegalActions(agentIndex)
        # 通过对合法动作遍历，轮询所有的下一个状态，取所有评价值和v的最小值
        for action in legalActions:
            value = self.getValue(gameState.generateSuccessor(agentIndex, action),
                                  (agentIndex+1) % gameState.getNumAgents(), depth, alpha, beta)
            if value is not None and value < minVal:
                minVal = value
            # 按照α-β剪枝算法，如果v<α，则直接返回minVal
            if minVal < alpha:
                return minVal
            # 按照α-β剪枝算法，这里还需要更新β的值
            if minVal < beta:
                beta = minVal
        return minVal


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # 基本算法：从吃豆人开始，遍历所有可行的下一步，取效用最佳的action为bestAction
        maxVal = -float('inf')
        bestAction = None
        for action in gameState.getLegalActions(agentIndex=0):
            # 求出后续状态的评价值，并和maxVal比较，求出MAX值
            value = self.getValue(gameState.generateSuccessor(
                agentIndex=0, action=action), agentIndex=1, depth=0)
            # 如果当前的value比maxVal还要大，更新maxVal值，并记下bestAction
            if value is not None and value > maxVal:
                maxVal = value
                bestAction = action
        # 最后返回最佳选择
        return bestAction

    def getValue(self, gameState, agentIndex, depth):
        # 如果当前状态是一个终点状态（没有可行的下一步），则返回当前状态的评价值
        legalActions = gameState.getLegalActions(agentIndex)
        if len(legalActions) == 0:
            return self.evaluationFunction(gameState)
        # 根据agentIndex的值，遍历下一个状态的评价值
        # 如果下一个是吃豆人的行动使用max_value()函数，如果是鬼怪的行动使用exp_value()函数
        if agentIndex == 0:
            # 如果agentIndex为0，表示所有的agent都已经轮询过一遍，此时depth加1
            depth += 1
            # 如果depth的值到达搜索深度限制，则返回当前状态的评价值，不再继续向下搜索
            if depth == self.depth:
                return self.evaluationFunction(gameState)
            else:
                return self.max_value(gameState, agentIndex, depth)
        elif agentIndex > 0:
            return self.exp_value(gameState, agentIndex, depth)

    def max_value(self, gameState, agentIndex, depth):
        # 初始化v等于负无穷
        maxVal = -float('inf')
        # 获得当前Agent所有合法的下一步动作
        legalActions = gameState.getLegalActions(agentIndex)
        # 通过对合法动作遍历，轮询所有的下一个状态，取所有评价值和v的最大值
        for action in legalActions:
            value = self.getValue(gameState.generateSuccessor(
                agentIndex, action), (agentIndex+1) % gameState.getNumAgents(), depth)
            if value is not None and value > maxVal:
                maxVal = value
        return maxVal

    def exp_value(self, gameState, agentIndex, depth):
        # 获得当前Agent所有合法的下一步动作
        legalActions = gameState.getLegalActions(agentIndex)
        # 初始化评价值总计为0
        totalValue = 0
        # 通过对合法动作遍历，轮询所有的下一个状态，取所有评价值的平均值作为计算结果返回
        for action in legalActions:
            value = self.getValue(gameState.generateSuccessor(agentIndex, action),
                                  (agentIndex+1) % gameState.getNumAgents(), depth)
            if value is not None:
                totalValue += value
        # 求评价值的平均结果，并返回
        return totalValue/(len(legalActions))


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # 此题要求对Reflex Agent的代码进行改进
    # 特别注意函数的参数发生了变化，此时我们只能观察到当前的状态，无法得知下一个状态的信息
    # 初始化可能需要的信息
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    # 找出最近的食物，计算正反馈
    if len(Food.asList()) > 0:
        nearestFood = (min([manhattanDistance(Pos, food)
                       for food in Food.asList()]))
        foodScore = 9/nearestFood
    else:
        foodScore = 0

    # 找出最近的鬼怪，计算负反馈
    nearestGhost = min([manhattanDistance(
        Pos, ghostState.configuration.pos) for ghostState in GhostStates])
    dangerScore = -10/nearestGhost if nearestGhost != 0 else 0

    # 这个值表示鬼怪保持可以被吃掉状态的剩余时间，其值为正
    # 由于通过吃掉地图上的大豆豆可以得到这个正反馈，所以吃豆人会考虑吃掉附近的大豆豆
    totalScaredTimes = sum(ScaredTimes)

    # 把计算好的各种启发值加在游戏得分上，并返回
    return currentGameState.getScore() + foodScore + dangerScore + totalScaredTimes


# Abbreviation
better = betterEvaluationFunction
