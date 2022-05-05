# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # 这个题目告诉我们Noise的值表示执行action后到达其他状态的概率
    # 所以，设置为0的话，就不可能到达其他的状态了，即Agent不会掉到桥两边的区域中
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

def question3a():
    # 折扣系数，表示目的地距离带来的折扣
    # 其值越小，表示距离越远的状态对当前Agent的影响就越小
    answerDiscount = 0.3
    # 在q2中已经解释过，表示以某个概率执行意外的行动
    answerNoise = 0
    # 生存奖励，其值越低，则Agent更愿意冒风险
    answerLivingReward = 0
    # 综上所述，此题的目的是降低奖励值高的节点对Agent的吸引力，并让Agent愿意沿着悬崖前进
    # python gridworld.py -a value -i 100 -g DiscountGrid -d 0.3 -n 0 -r 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # 此题中，我们适当提高掉入悬崖的概率，Agent就会避开悬崖
    # 但是不能太大，否则会导致以较大的概率到达意外的状态
    # python gridworld.py -a value -i 100 -g DiscountGrid -d 0.3 -n 0.1 -r 0
    answerDiscount = 0.3
    answerNoise = 0.1
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # 此题中，我们增大最右边奖励值高的节点的吸引力，并且降低noise，让Agent愿意沿着悬崖前进
    # python gridworld.py -a value -i 100 -g DiscountGrid -d 0.9 -n 0 -r 0
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # 此题同样是增大奖励值最高节点的吸引力,但是增大Noise，让Agent避开可能调入的悬崖
    # python gridworld.py -a value -i 100 -g DiscountGrid -d 0.9 -n 0.1 -r 0
    answerDiscount = 0.9
    answerNoise = 0.1
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # 此题需要降低所有奖励值节点对Agent的吸引力，同时要避开悬崖边，所以适当提高Noise值
    # 同时适当调高生存奖励，让Agent不想去终点
    # python gridworld.py -a value -i 100 -g DiscountGrid -d 0 -n 0.8 -r 1
    answerDiscount = 0
    answerNoise = 0.1
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    # 通过实验，我们可以发现根本不可能在50次迭代内训练出一个可以到达高奖励值的模型
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
