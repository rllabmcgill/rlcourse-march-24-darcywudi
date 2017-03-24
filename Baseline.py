import numpy as np

class CliffWalking:
    def __init__(self, width = 12, height = 4, variance = 0.1):
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0
        self.variance = variance
    def resetPosition(self):
        self.x = 0
        self.y = 0
    def getReward(self):
        rew = -1
        if self.y == 0 and self.x == self.width - 1:
            if self.variance > 0:
                rew = np.random.normal(100, 100 * self.variance)
            else:
                rew = 100
        elif self.y == 0 and self.x > 0 and self.x < self.width - 1:
            #cliff
            self.resetPosition()
            if self.variance > 0:
                rew = np.random.normal(-100, 100 * self.variance)
            else:
                rew = -100
        else:
            if self.variance > 0:
                rew = np.random.normal(-1, self.variance)
            else:
                rew = -1
        return rew
    def move(self, moveType):
        if moveType == 'u':
            if self.y < self.height - 1:
                self.y += 1
        elif moveType == 'd':
            if self.y > 0:
                self.y -= 1
        elif moveType == 'r':
            if self.x < self.width - 1:
                self.x += 1
        elif moveType == 'l':
            if self.x > 0:
                self.x -= 1
        return self.getReward()
    def isEnd(self):
        return self.x > 0 and self.y == 0
    def getPosition(self):
        return [self.x, self.y]


# ## Q-learning

# In[81]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
gamma = 0.8
#Q Learning
def qLearning(cw, width, height, avgR, iterator, max_iter):
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
        q[s0[0], s0[1], a] +=  0.1 * (r + gamma * q[s1[0], s1[1]].max() - q[s0[0], s0[1], a])
    iterator.append(i)

# ## Double Q-learning

# In[82]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Q Learning
def dqLearning(cw, width, height, avgR, iterator, max_iter):
    q1 = np.zeros((width, height, 4))
    q2 = np.zeros((width, height, 4))
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = np.argmax(q1[s0[0], s0[1]] + q2[s0[0], s0[1]])
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
        if np.random.random() < 0.5:
            q1[s0[0], s0[1], a] +=  0.1 * (r + gamma * q2[s1[0], s1[1], q1[s1[0], s1[1]].argmax()] - q1[s0[0], s0[1], a])
        else:
            q2[s0[0], s0[1], a] +=  0.1 * (r + gamma * q1[s1[0], s1[1], q2[s1[0], s1[1]].argmax()] - q2[s0[0], s0[1], a])
    iterator.append(i)


# ## Sarsa

# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def Sarsa(cw, width, height, avgR, iterator, max_iter):
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
        q[s0[0], s0[1], a] +=  0.1 * (r + gamma * q[s1[0], s1[1], a1] - q[s0[0], s0[1], a])
    iterator.append(i)
