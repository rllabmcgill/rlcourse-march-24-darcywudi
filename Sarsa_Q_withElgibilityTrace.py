
# coding: utf-8

# In this code,Expected Sarsa, Two Step Sarsa, double sarsa, and double expected sarsa are implemented
# I use the previous implementation of q, sarsa and double q (By weiwei Zhang) as comparison baslines 
# I use the same problem setting as the baselines, and this implementation uses some of the code from weiwei's implemtion on q,double q, and sarsa
# In[80]:



import numpy as np
from Baseline import CliffWalking
from Baseline import qLearning
from Baseline import dqLearning
from Baseline import Sarsa

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
gamma = 0.8


# ## Sarsa with accumulating trace
# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def SarsaEga(cw, width, height, avgR, iterator, max_iter, lmbdaeg):
    q = np.zeros((width, height, 4))
    eg = np.zeros((width, height, 4))
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = q[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
        a1 = q[s1[0], s1[1]].argmax()
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
        delta = r + gamma * q[s1[0], s1[1], a1] - q[s0[0], s0[1], a]
        
        eg[s0[0], s0[1], a] = eg[s0[0], s0[1], a] + 1   
        
        q[s0[0], s0[1], a] +=  0.1 * (delta)*eg[s0[0], s0[1],a]
        
        eg[s0[0], s0[1]] = gamma*lmbdaeg*eg[s0[0], s0[1], a]
        
    iterator.append(i)

    

# ## Sarsa with dutch trace
# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
alpha = 0.1
#Sarsa
def SarsaEgd(cw, width, height, avgR, iterator, max_iter, lmbdaeg):
    q = np.zeros((width, height, 4))
    eg = np.zeros((width, height, 4))
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = q[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
        a1 = q[s1[0], s1[1]].argmax()
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
        delta = r + gamma * q[s1[0], s1[1], a1] - q[s0[0], s0[1], a]
        
        eg[s0[0], s0[1], a] = (1-alpha)*eg[s0[0], s0[1], a] + 1   
        
        q[s0[0], s0[1], a] +=  alpha * (delta)*eg[s0[0], s0[1],a]
        
        eg[s0[0], s0[1]] = gamma*lmbdaeg*eg[s0[0], s0[1], a]
        
    iterator.append(i)

#Q Learning with accumulating trace
alpha = 0.1
def qLearningEga(cw, width, height, avgR, iterator, max_iter, lmbdaeg):
    q = np.zeros((width, height, 4))
    eg = np.zeros((width, height, 4))
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = q[s0[0], s0[1]].argmax()
        astar = a
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
        delta = r + gamma * q[s1[0], s1[1]].max() - q[s0[0], s0[1], a]
        eg[s0[0], s0[1], a] = eg[s0[0], s0[1], a] + 1   
        q[s0[0], s0[1], a] +=  alpha * delta * eg[s0[0], s0[1],a]
        if a == astar:
            eg[s0[0], s0[1]] = gamma*lmbdaeg*eg[s0[0], s0[1], a]
        else:
			eg[s0[0], s0[1]] = 0
    iterator.append(i)


#Q Learning with dutch trace
alpha = 0.1
def qLearningEgd(cw, width, height, avgR, iterator, max_iter, lmbdaeg):
    q = np.zeros((width, height, 4))
    eg = np.zeros((width, height, 4))
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = q[s0[0], s0[1]].argmax()
        astar = a
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
        delta = r + gamma * q[s1[0], s1[1]].max() - q[s0[0], s0[1], a]
        eg[s0[0], s0[1], a] = (1-alpha)*eg[s0[0], s0[1], a] + 1   
        q[s0[0], s0[1], a] +=  alpha * delta * eg[s0[0], s0[1],a]
        if a == astar:
            eg[s0[0], s0[1]] = gamma*lmbdaeg*eg[s0[0], s0[1], a]
        else:
			eg[s0[0], s0[1]] = 0
    iterator.append(i)
    

# ## Expected Sarsa

# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def ExpSarsa(cw, width, height, avgR, iterator, max_iter):
    q = np.zeros((width, height, 4))
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = q[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
        a1 = q[s1[0], s1[1]].argmax()
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
        q[s0[0], s0[1], a] +=  0.1 * (r + gamma * 0.25*(q[s1[0], s1[1], 0] + q[s1[0], s1[1], 1] + q[s1[0], s1[1], 2] + q[s1[0], s1[1], 3] ) - q[s0[0], s0[1], a])
    iterator.append(i)


# ## Double Sarsa

# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def dSarsa(cw, width, height, avgR, iterator, max_iter):
#    q = np.zeros((width, height, 4))
    qA = np.zeros((width, height, 4))
    qB = np.zeros((width, height, 4))
    qC = np.zeros((width, height, 4))
    
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = qA[s0[0], s0[1]].argmax()
        if np.random.random() < 0.5:
            a = qB[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
         
        
        qC = 0.5*(qA + qB)
        a1 = qC[s1[0], s1[1]].argmax()
        
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
        qA[s0[0], s0[1], a] +=  0.1 * (r + gamma * qB[s1[0], s1[1], a1] - qA[s0[0], s0[1], a])
        
        if np.random.random() <= 0.5:
                temp = qA
                qA = qB
                qB = temp
    iterator.append(i)


# ## Sarsa

# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def dExpSarsa(cw, width, height, avgR, iterator, max_iter):
    qA = np.zeros((width, height, 4))
    qB = np.zeros((width, height, 4))
    qC = np.zeros((width, height, 4))
    
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = qA[s0[0], s0[1]].argmax()
        if np.random.random() < 0.5:
            a = qB[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
         
        
        qC = 0.5*(qA + qB)
        a1 = qC[s1[0], s1[1]].argmax()
        
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
            qA[s0[0], s0[1], a] +=  0.1 * (r + gamma * 0.25*(qB[s1[0], s1[1], 0] + qB[s1[0], s1[1], 1] + qB[s1[0], s1[1], 2] + qB[s1[0], s1[1], 3] ) - qA[s0[0], s0[1], a])
        if np.random.random() <= 0.5:
                temp = qA
                qA = qB
                qB = temp
    iterator.append(i)



# ## Multi Step Sarsa

# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def TwoSarsa(cw, width, height, avgR, iterator, max_iter):
    q = np.zeros((width, height, 4))
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a0 = q[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a0 = np.random.choice(range(4))
        r0 = cw.move(actions[a0])
        G += r0
        i += 1
        
        s1 = cw.getPosition()
        a1 = q[s1[0], s1[1]].argmax()
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
        r1 = cw.move(actions[a1])

        s2 = cw.getPosition()
        a2 = q[s2[0], s2[1]].argmax()
        if np.random.random() < epsilon:
            a2 = np.random.choice(range(4))
        G += r1 - gamma*gamma*q[s2[0], s2[1], a2]
        if i < max_iter:
            avgR[i] = G / i
        q[s0[0], s0[1], a0] +=  0.1 * (r0 +   gamma*gamma * q[s2[0], s2[1], a2] - q[s0[0], s0[1], a0])
    iterator.append(i)




# ## Sarsa, Q-learning and Double Q-learning on CliffWalking

# In[90]:

import matplotlib.pyplot as plt
#initialize CliffWalking
cw = CliffWalking(variance = 1)
ite1 = []
avgR1 = [[0.0]*500] * 500
ite2 = []
avgR2 = [[0.0]*500] * 500
for i in range(500):
    cw.resetPosition()
    qLearning(cw, 12, 4, avgR1[i], ite1, 499)
    cw.resetPosition()
    dqLearning(cw, 12, 4, avgR2[i], ite2, 499)
plt.plot(np.mean(np.asarray(avgR1, dtype=np.float32), axis=0), label = 'QLearning')
print "Average iteration of Q-learning: " + str(np.mean(ite1))
plt.plot(np.mean(np.asarray(avgR2, dtype=np.float32), axis=0), label = 'Double QLearning')
print "Average iteration of Double Q-learning: " + str(np.mean(ite2))
plt.ylabel('Reward per episode')
plt.xlabel('Episodes')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ## Sarsa, Q-learning and Double Q-learning on randomized CliffWalking

# In[92]:

#initialize Random CliffWalking
variances = [0, 0.1, 0.5, 1, 5, 10]
ite1_a = []
ite2_a = []
ite3_a = []
ite4_a = []
ite5_a = []
ite6_a = []
ite7_a = []
ite8_a = []
ite9_a = []
ite10_a = []
ite11_a = []


for variance in variances:
    cw = CliffWalking(variance = variance)
    ite1 = []
    avgR1 = [[0.0]*200] * 100
    ite2 = []
    avgR2 = [[0.0]*200] * 100
    ite3 = []
    avgR3 = [[0.0]*200] * 100
    ite4 = []
    avgR4 = [[0.0]*200] * 100
    ite5 = []
    avgR5 = [[0.0]*200] * 100
    ite6 = []
    avgR6 = [[0.0]*200] * 100
    ite7 = []
    avgR7 = [[0.0]*200] * 100
    ite8 = []
    avgR8 = [[0.0]*200] * 100
    ite9 = []
    avgR9 = [[0.0]*200] * 100
    ite10 = []
    avgR10 = [[0.0]*200] * 100
    ite11 = []
    avgR11 = [[0.0]*200] * 100
    
	
    for i in range(100):
        cw.resetPosition()
        qLearning(cw, 12, 4, avgR1[i], ite1, 199)
        cw.resetPosition()
        dqLearning(cw, 12, 4, avgR2[i], ite2, 199)
        cw.resetPosition()
        Sarsa(cw, 12, 4, avgR3[i], ite3, 199)
        cw.resetPosition()
        ExpSarsa(cw, 12, 4, avgR4[i], ite4, 199)
        cw.resetPosition()
        TwoSarsa(cw, 12, 4, avgR5[i], ite5, 199)
        cw.resetPosition()
        dSarsa(cw, 12, 4, avgR6[i], ite6, 199)
        cw.resetPosition()
        dExpSarsa(cw, 12, 4, avgR7[i], ite7, 199)
        cw.resetPosition()
        SarsaEga(cw, 12, 4, avgR8[i], ite8, 199, 0.4)
        cw.resetPosition()
        SarsaEgd(cw, 12, 4, avgR9[i], ite9, 199, 0.4)
        cw.resetPosition()
        qLearningEga(cw, 12, 4, avgR10[i], ite10, 199, 0.4)
        cw.resetPosition()
        qLearningEgd(cw, 12, 4, avgR11[i], ite11, 199, 0.4)
        
		
    ite1_a.append(np.mean(ite1))
    ite2_a.append(np.mean(ite2))
    ite3_a.append(np.mean(ite3))
    ite4_a.append(np.mean(ite4))
    ite5_a.append(np.mean(ite5))
    ite6_a.append(np.mean(ite6))
    ite7_a.append(np.mean(ite7))
    ite8_a.append(np.mean(ite8))
    ite9_a.append(np.mean(ite9))
    ite10_a.append(np.mean(ite10))
    ite11_a.append(np.mean(ite11))
	
	
    print "Variance:" + str(variance)
    print "Average iteration of Q-learning: " + str(np.mean(ite1))
    print "Average iteration of Double Q-learning: " + str(np.mean(ite2))
    print "Average iteration of Sarsa: " + str(np.mean(ite3))
    print "Average iteration of Expected Sarsa: " + str(np.mean(ite4))
    print "Average iteration of Two Step Sarsa: " + str(np.mean(ite5))
    print "Average iteration of Double Sarsa: " + str(np.mean(ite6))
    print "Average iteration of Double Expected Sarsa: " + str(np.mean(ite7))
    print "Average iteration of Sarsa with Accumulating Eligibility Trace, 0.4: " + str(np.mean(ite8))
    print "Average iteration of Sarsa with Dutch Eligibility Trace, 0.4: " + str(np.mean(ite9))
    print "Average iteration of Q Learning with Accumulating Eligibility Trace, 0.9: " + str(np.mean(ite10))
    print "Average iteration of Q Learning with Accumulating Eligibility Trace, 0.9: " + str(np.mean(ite11))
    
	


plt.plot(ite1_a, label = 'QLearning')
plt.plot(ite2_a, label = 'Double QLearning')
plt.plot(ite3_a, label = 'Sarsa')
plt.plot(ite4_a, label = 'Expected Sarsa')
plt.plot(ite5_a, label = 'Two-Step Sarsa')
plt.plot(ite6_a, label = 'Double Sarsa')
plt.plot(ite7_a, label = 'Double Expected Sarsa')
plt.plot(ite8_a, label = 'Sarsa Eligibility Trace')


plt.xticks(range(6), variances, rotation=0)  
plt.ylabel('Number of steps')
plt.xlabel('Variances')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



# Plot the average return 11
plt.plot(ite1_a, label = 'QLearning')
plt.plot(ite3_a, label = 'Sarsa')
plt.plot(ite4_a, label = 'Expected Sarsa')
plt.plot(ite5_a, label = 'Two-Step Sarsa')

plt.xticks(range(6), variances, rotation=0)  
plt.ylabel('Number of steps')
plt.xlabel('Variances')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
plt.savefig('./Figure/f1.jpg')

# Plot the average return for sarsa
plt.plot(ite3_a, label = 'Sarsa') 
plt.plot(ite8_a, label = 'Sarsa Accumulating Eligibility Trace, lambda = 0.4')
plt.plot(ite9_a, label = 'Sarsa Dutch Eligibility Trace, lambda = 0.4')


plt.xticks(range(6), variances, rotation=0)  
plt.ylabel('Number of steps')
plt.xlabel('Variances')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
plt.savefig('./Figure/sarsa.jpg') 

#plot the average return for q learning


plt.plot(ite1_a,  markersize=8, label = 'QLearning') 
plt.plot(ite10_a,  markersize=8, label = 'QLearning Accumulating Trace, lambda = 0.4')
plt.plot(ite11_a,  markersize=6, label = 'QLearning Dutch  Trace, lambda = 0.4')

plt.xticks(range(6), variances, rotation=0)  
plt.ylabel('Number of steps')
plt.xlabel('Variances')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


plt.savefig('./Figure/Qlearning.jpg') 




# ## Discussion
# We create a randomized CliffWalking environment and compare these methods based on different vairance ratio. From the experiments,Double Q-learning is less sensitive to the variances (the higer variance the more uncertain environment) and needs much fewer steps to reach the goal when the variance is big.

# In[ ]:



