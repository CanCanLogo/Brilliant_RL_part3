import numpy as np

class MDP:
    '''一个简单的MDP类，它包含如下成员'''
    '''
    __init__是构造函数，初始化python类中的变量；
    valueIteration是值迭代函数；
    extractPolicy是根据状态价值提取策略的函数；
    evaluetePolicy是策略评估函数，通过求解线性方程组来实现；
    policyIteration是策略迭代函数；
    evaluatePolicyPartially是部分的策略评估函数，执行多次值迭代来代替精确评估；
    modifiedPolicyIteration是修改后的策略迭代函数，其中调用了 evaluatePolicyPartially函数进行策略评估。
    '''

    def __init__(self,T,R,discount):
        '''构建MDP类

        输入:
        T -- 转移函数: |A| x |S| x |S'| array
        R -- 奖励函数: |A| x |S| array
        V -- 值函数: 大小为|S|的array
        discount -- 折扣因子: scalar in [0,1)

        构造函数检验输入是否有效，并在MDP对象中构建相应变量'''

        assert T.ndim == 3, "转移函数无效，应该有3个维度"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "无效的转换函数：它具有维度 " + repr(T.shape) + ", 但它应该是(nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "无效的转移函数：某些转移概率不等于1"
        self.T = T
        assert R.ndim == 2, "奖励功能无效：应该有2个维度"
        assert R.shape == (self.nActions,self.nStates), "奖励函数无效：它具有维度 " + repr(R.shape) + ", 但它应该是 (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "折扣系数无效：它应该在[0,1]中"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''值迭代法
        V <-- max_a R^a + gamma T^a V

        输入:
        initialV -- 初始的值函数: 大小为|S|的array   initialV=np.zeros(mdp.nStates)
        nIterations -- 迭代次数的限制：标量 (默认值: infinity)
        tolerance -- ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''
        iterId = 0
        epsilon = 0
        V = initialV
        while iterId < nIterations:
            iterId += 1
            V_new = np.zeros_like(V)
            for state in range(self.nStates):
                max_value = float("-inf")
                for action in range(self.nActions):
                    value_temp = self.R[action] + np.dot(self.T[action][state], self.discount * V)
                    max_value = max(max_value, value_temp[state])
                V_new[state] = max_value
            epsilon = np.max(np.abs(V-V_new))
            V = V_new
            if epsilon < tolerance:
                break
            else:
                continue
        #填空部分
        return [V, iterId, epsilon]

    def extractPolicy(self,V):
        '''从值函数中提取具体策略的程序
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- 值函数: 大小为|S|的array

        Output:
        policy -- 策略: 大小为|S|的array'''
        #填空部分

        # V  'list' object has no attribute 'shape'
        policy = []
        # print(V)
        V = np.float64(V)
        for state in range(self.nStates):
            value_a_iter = []
            for action in range(self.nActions):
                value_temp = self.R[action] + np.dot(self.T[action][state], np.multiply(self.discount, V))
                value_a_iter.append(value_temp[state])
            a = np.argmax(value_a_iter)
            policy.append(a)
        return policy 

    def evaluatePolicy(self,policy):
        '''通过求解线性方程组来评估政策
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- 策略: 大小为|S|的array

        Ouput:
        V -- 值函数: 大小为|S|的array'''
        #填空部分
        T_pi = []
        for state in range(self.nStates):
            action = policy[state]
            T_pi.append(self.T[action][state])
        # AX = B
        A = np.eye(self.nStates) - np.multiply(T_pi, self.discount)
        B = self.R[0]
        V = np.linalg.solve(A, B)
        # V = np.around(V, 8)
        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''策略迭代程序:  在策略评估(solve V^pi = R^pi + gamma T^pi V^pi) 和
        策略改进 (pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: inf)

        Outputs: 
        policy -- 策略: 大小为|S|的array
        V -- 值函数: 大小为|S|的array
        iterId --策略迭代执行的次数 : 标量'''
        tolerance = 0.01

        V = np.zeros(self.nStates)
        policy = initialPolicy
        iterId = 0
        #填空部分
        while iterId < nIterations:
            iterId += 1
            V_prev = self.evaluatePolicy(policy)  # 策略评估
            policy_new = self.extractPolicy(V_prev)  # 策略改进

            V = self.evaluatePolicy(policy_new)
            policy = policy_new

            epsilon = np.max(np.abs(V - V_prev))
            if epsilon < tolerance:
                break
            else:
                continue
        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''部分的策略评估:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- 策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: infinity)
        tolerance --  ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 迭代执行的次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        V = np.float64(initialV)
        iterId = 0
        epsilon = 0
        #填空部分
        while iterId < nIterations:
            iterId += 1
            V_new = np.zeros_like(V)
            for state in range(self.nStates):
                action = policy[state]
                value_temp = self.R[action] + np.dot(self.T[action][state], self.discount * V)
                V_new[state] = value_temp[state]
            epsilon = np.max(np.abs(V - V_new))
            V = V_new
            if epsilon < tolerance:
                break
            else:
                pass

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''修改的策略迭代程序: 在部分策略评估 (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        和策略改进(pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nEvalIterations -- 每次部分策略评估时迭代次数的限制: 标量 (默认值: 5)
        nIterations -- 修改的策略迭代中迭代次数的限制: 标量 (默认值: inf)
        tolerance -- ||V^n-V^n+1||_inf的阈值: scalar (默认值: 0.01)

        Outputs: 
        policy -- 策略: 大小为|S|的array
        V --值函数: 大小为|S|的array
        iterId -- 修改后策略迭代执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        iterId = 0
        epsilon = 0
        policy = initialPolicy
        V = np.float64(initialV)

        #填空部分
        while iterId < nIterations:
            iterId += 1
            V_prev, _, epsilon = self.evaluatePolicyPartially(policy, V, nEvalIterations)  # 策略评估
            # 注意这个函数输出的是元组
            policy_new = self.extractPolicy(V_prev)  # 策略改进

            # V = self.evaluatePolicy(policy_new)
            policy = policy_new

            # epsilon = np.max(np.abs(V - V_prev))
            V = V_prev
            if epsilon < tolerance:
                break
            else:
                continue


        return [policy,V,iterId,epsilon]
        