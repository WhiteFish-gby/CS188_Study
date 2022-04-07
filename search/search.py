# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from ast import NodeTransformer
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # 创建空集合closed 用于存放已经搜索过的节点
    closed = set()
    # 创建待搜索的节点集合fringe，根据深度优先算法的特点，应该使用栈作为其数据结构
    fringe = util.Stack()  # Stack到底是个什么类型的container,取决于传入的变量
    # 并将搜索问题的初始状态作为第一个待搜索节点添加到fringe集合中
    node = {"state": problem.getStartState(), "path": []}
    # "Push node onto the stack"初始化
    fringe.push(node)
    # 构建循环进行搜索
    while True:
        # 如果fringe集合为空，表示已经把所有待搜索的节点都搜索过了，但依然没有找到可行的行动序列，则搜索失败
        if fringe.isEmpty():
            return None
        # 如果上面的步骤不成立，表示fringe中还存在待搜索的节点，则从fringe取出下一个待搜索节点
        node = fringe.pop()
        # 判断取出的节点是不是目标节点，如果是的话，表示搜索成功，返回到达该节点的行动序列即可
        if problem.isGoalState(node["state"]):
            return node["path"]
        # 如果当前节点不是目标节点，且节点信息不在已经搜索过的节点
        elif node["state"] not in closed:
            # 将当前节点添加到已搜索的节点集合closed中
            closed.add(node["state"])
            # 展开当前节点的后续节点，并对这些节点进行搜索
            for nextnode in problem.getSuccessors(node["state"]):
                # 构造后续节点的相关信息，并将其添加到待搜索节点集合fringe中
                nextnode = {"state": nextnode[0],
                            "path": node["path"]+[nextnode[1]]}
                fringe.push(nextnode)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # 创建空集合closed用于存放已经搜索过的节点
    closed = set()
    # 创建待搜索的节点集合fringe，根据广度优先搜索算法的特点，应该使用队列作为其数据结构
    fringe = util.Queue()
    # 并将搜索问题的初始状态作为第一个待搜索节点添加到fringe集合中
    # 每一个节点中除了当前的状态，还需要保存到达此节点所需执行的行动序列，即变量path中存放的内容
    node = {"state": problem.getStartState(), "path": []}
    fringe.push(node)
    # 构建循环进行搜索
    while True:
        # 如果fringe集合为空，表示已经把所有待搜索的节点都搜索过了，但依然没有找到可行的行动序列，则搜索失败
        if fringe.isEmpty():
            return None
        # 如果上面的步骤不成立，表示fringe中还存在待搜索的节点，则从fringe中取出下一个待搜索节点
        node = fringe.pop()
        # 判断取出的节点是不是目标节点，如果是的话，表示搜索成功，返回到达该节点的行动序列即可
        if problem.isGoalState(node["state"]):
            return node["path"]
        # 如果当前节点不是目标节点，且节点信息不在已经搜索过的节点集合closed中
        elif node["state"] not in closed:
            # 将当前节点添加到已搜索的节点集合closed中
            closed.add(node["state"])
            # 展开当前节点的后续节点，并对这些节点进行遍历搜索
            for nextnode in problem.getSuccessors(node["state"]):
                # 构造后续节点的相关信息，并将其添加到待搜索节点集合fringe中
                nextnode = {"state": nextnode[0],
                            "path": node["path"]+[nextnode[1]]}
                fringe.push(nextnode)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # 创建空集合closed用于存放已经搜索过的节点
    closed = set()
    # 创建待搜索的节点集合fringe，根据一致代价搜索算法的特点，应该使用优先级队列作为其数据结构
    fringe = util.PriorityQueue()
    # 并将搜索问题的初始状态作为第一个待搜索节点添加到fringe集合中
    # 每一个节点中除了当前的状态，还需要保存到达此节点所需执行的行动序列和代价，即变量path和cost中存放的内容
    node = {"state": problem.getStartState(), "path": [], "cost": 0}
    fringe.push(node, node["cost"])
    # 构建循环进行搜索
    while True:
        # 如果fringe集合为空，表示已经把所有待搜索的节点都搜索过了，但依然没有找到可行的行动序列，则搜索失败
        if fringe.isEmpty():
            return None
        # 如果上面的步骤不成立，表示fringe中还存在待搜索的节点，则从fringe中取出下一个待搜索节点
        node = fringe.pop()
        # 判断取出的节点是不是目标节点，如果是的话，表示搜索成功，返回到达该节点的行动序列即可
        if problem.isGoalState(node["state"]):
            return node["path"]
        # 如果当前节点不是目标节点，且节点信息不在已经搜索过的节点集合closed中
        elif node["state"] not in closed:
            # 将当前节点添加到已搜索的节点集合closed中
            closed.add(node["state"])
            # 展开当前节点的后续节点，并对这些节点进行遍历搜索
            for nextnode in problem.getSuccessors(node["state"]):
                # 构造后续节点的相关信息，并将其添加到待搜索节点集合fringe中
                nextnode = {"state": nextnode[0],
                            "path": node["path"]+[nextnode[1]],
                            "cost": node["cost"]+nextnode[2]}
                fringe.update(nextnode, nextnode["cost"])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # 创建空集合closed用于存放已经搜索过的节点
    closed = set()
    # 创建待搜索的节点集合fringe，根据统一代价搜索算法的特点，应该使用优先级队列作为其数据结构
    fringe = util.PriorityQueue()
    # 并将搜索问题的初始状态作为第一个待搜索节点添加到fringe集合中
    # 每一个节点中除了当前的状态，还需要保存到达此节点所需执行的行动序列和代价，即变量path和cost中存放的内容
    node = {"state": problem.getStartState(), "path": [], "cost": 0}
    fringe.push(node, node["cost"]+heuristic(node["state"], problem))
    # 构建循环进行搜索
    while True:
        # 如果fringe集合为空，表示已经把所有待搜索的节点都搜索过了，但依然没有找到可行的行动序列，则搜索失败
        if fringe.isEmpty():
            return None
        # 如果上面的步骤不成立，表示fringe中还存在待搜索的节点，则从fringe中取出下一个待搜索节点
        node = fringe.pop()
        # 判断取出的节点是不是目标节点，如果是的话，表示搜索成功，返回到达该节点的行动序列即可
        if problem.isGoalState(node["state"]):
            return node["path"]
        # 如果当前节点不是目标节点，且节点信息不在已经搜索过的节点集合closed中
        elif node["state"] not in closed:
            # 将当前节点添加到已搜索的节点集合closed中
            closed.add(node["state"])
            # 展开当前节点的后续节点，并对这些节点进行遍历搜索
            for nextnode in problem.getSuccessors(node["state"]):
                # 构造后续节点的相关信息，并将其添加到待搜索节点集合fringe中
                nextnode = {"state": nextnode[0],
                            "path": node["path"]+[nextnode[1]],
                            "cost": node["cost"]+nextnode[2]}
                fringe.update(
                    nextnode, nextnode["cost"]+heuristic(nextnode["state"], problem))


bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
