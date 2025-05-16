import math
import random
import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''
        # print(state)
        # print(action)
        # print(self.mdp.R[action][state])

        # print(self.mdp.R.shape)
        # print(action, state)
        reward = self.sampleReward(self.mdp.R[action][state])
        # 按照当前R值为均值根据高斯密度函数得到随机奖励
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        # 把T转移概率由概率密度加为概率
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        # 使用随机数来比较T累加密度来获得下一个状态
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''
        qLearning算法，需要将Epsilon exploration和 Boltzmann exploration 相结合。
        以epsilon的概率随机取一个动作，否则采用 Boltzmann exploration取动作。
        当epsilon和temperature都为0时，将不进行探索。

        Boltzmann exploration:softmax 对于自己有把握的状态，就应该采取exploitation；而对于自己没有把握的状态，由于训练中的输赢不重要，
        所以可以多去尝试exploration，但同时也不是盲目地乱走，而是尝试一些自己认为或许还不错的走法。
        https://zhuanlan.zhihu.com/p/166410986

        Inputs:
        s0 -- 初始状态
        initialQ -- 初始化Q函数 (|A|x|S| array)
        nEpisodes -- 回合（episodes）的数量 (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- 每个回合的步数(steps)
        epsilon -- 随机选取一个动作的概率
        temperature -- 调节 Boltzmann exploration 的参数

        Outputs: 
        Q -- 最终的 Q函数 (|A|x|S| array)
        policy -- 最终的策略
        rewardList -- 每个episode的累计奖励（|nEpisodes| array）
        '''
        temperature_delta = 1
        epsilon_rate = 0.99
        episode = 0
        Qtable = initialQ
        learning_rate = 0.05
        rewardList = []
        policy_history = []

        while episode < nEpisodes:
            s = s0
            episode += 1
            step = 0
            rewardSum = 0
            temperature += temperature_delta
            epsilon *= epsilon_rate
            while step < nSteps:
                step += 1
                # 获得s下a的各q值

                A_choose = Qtable[:, s]
                A_choose = A_choose.squeeze()
                # print(A_choose)

                # 生成备选的a列表
                actions = np.arange(0, 1, Qtable.shape[0])

                # linspase会出现float

                # 取a
                # a = 0
                eps = random.uniform(0,1)
                # print(eps)
                if eps >= epsilon:
                    # exploitation
                    # softmax忘记除了
                    softmax = [temperature * math.exp(A_choose[i]) for i in range(Qtable.shape[0])]
                    softSum = sum(softmax)
                    softmax = [softmax[i]/softSum for i in range(Qtable.shape[0])]
                    # print(softmax)
                    cumProb = np.cumsum(softmax)  # 把
                    # print(cumProb)
                    # print(np.where(cumProb >= np.random.rand(1)))
                    a = np.where(cumProb >= np.random.rand(1))[0][0]
                else:
                    a = np.random.choice(actions)
                # print(a)
                vector = self.sampleRewardAndNextState(s, a)
                # (s,a,r,s')
                reward = vector[0]
                # print(reward)
                nextState = vector[1]

                # qtable_nextS = Qtable[:, nextState]
                # qtable_nextS = qtable_nextS.squeeze()
                qtable_nextS = Qtable[:, nextState]
                qtable_nextS = qtable_nextS.squeeze()

                current_q = Qtable[a][s]
                # 贝尔曼方程更新
                new_q = reward + self.mdp.discount * max(qtable_nextS)
                # print(current_q, new_q)
                Qtable[a][s] += learning_rate * (new_q - current_q)
                # print(reward)

                rewardSum += reward
                s = nextState
            # print(rewardSum)
            rewardList.append(rewardSum)

        Q = Qtable
        policy = [np.argmax(Qtable[:, i].squeeze()) for i in range(Qtable.shape[1])]
        # rewardList = np.zeros(nEpisodes)

        return [Q,policy,rewardList]