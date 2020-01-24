import numpy as np
import random as random
from random import randrange

SAMPLES = 100
SEPARATION = 0.33

def linearly_separable_data(mA, sigmaA, mB, sigmaB):
    classA_x1 = np.zeros(SAMPLES)
    classA_x2 = np.zeros(SAMPLES)
    classB_x1 = np.zeros(SAMPLES)
    classB_x2 = np.zeros(SAMPLES)
    for i in range(SAMPLES):
        classA_x1[i] = np.random.normal() * sigmaA + mA[0] + SEPARATION
        classA_x2[i] = np.random.normal() * sigmaA + mA[1] + SEPARATION
        classB_x1[i] = np.random.normal() * sigmaB + mB[0] - SEPARATION
        classB_x2[i] = np.random.normal() * sigmaB + mB[1] - SEPARATION
    classA = np.vstack([classA_x1, classA_x2])
    labelsA = np.ones([1,SAMPLES])
    classB = np.vstack([classB_x1, classB_x2])
    labelsB = -np.ones([1,SAMPLES])
    
    data = np.hstack([classA, classB])
    targets = np.hstack([labelsA, labelsB])
    
    X = np.zeros([3, 2 * SAMPLES])
    t = np.zeros([1, 2 * SAMPLES])
    shuffle = np.random.permutation(2 * SAMPLES)
    
    for i in shuffle:
        X[:2,i] = data[:2, shuffle[i]] # shuffle data
        X[2,i] = 1 # add bias
        t[0,i] = targets[0, shuffle[i]]
    
    return X, t

def new_data_generation(m_a, m_b, sigma_a, sigma_b):
    classA_x1 = np.zeros(SAMPLES)
    classA_x2 = np.zeros(SAMPLES)
    classB_x1 = np.zeros(SAMPLES)
    classB_x2 = np.zeros(SAMPLES)
    for i in range(round(SAMPLES/2)):
        classA_x1[i] = np.random.normal() * sigma_a - m_a[0]
    for i in range(round(SAMPLES/2)):
        classA_x1[i + round(SAMPLES/2)] = np.random.normal() * sigma_a + m_a[0]
    for i in range(SAMPLES):
        classA_x2[i] = np.random.normal() * sigma_a + m_a[1]
        classB_x1[i] = np.random.normal() * sigma_b + m_b[0]
        classB_x2[i] = np.random.normal() * sigma_b + m_b[1]
    classA = np.vstack([classA_x1, classA_x2])
    labelsA = np.ones([1,SAMPLES])
    classB = np.vstack([classB_x1, classB_x2])
    labelsB = -np.ones([1,SAMPLES])
    
    data = np.hstack([classA, classB])
    targets = np.hstack([labelsA, labelsB])
    
    X = np.zeros([3, 2 * SAMPLES])
    t = np.zeros([1, 2 * SAMPLES])
    shuffle = np.random.permutation(2 * SAMPLES)
    
    for i in shuffle:
        X[:2,i] = data[:2, shuffle[i]] # shuffle data
        X[2,i] = 1 # add bias
        t[0,i] = targets[0, shuffle[i]]
    
    return X, t

def non_linearly_separable_data(m, sigma):
    classA_x1 = np.zeros(SAMPLES)
    classA_x2 = np.zeros(SAMPLES)
    classB_x1 = np.zeros(SAMPLES)
    classB_x2 = np.zeros(SAMPLES)
    for i in range(SAMPLES):
        classA_x1[i] = np.random.normal() * sigma + m[0]
        classA_x2[i] = np.random.normal() * sigma + m[1]
        classB_x1[i] = np.random.normal() * sigma + m[0]
        classB_x2[i] = np.random.normal() * sigma + m[1]
    classA = np.vstack([classA_x1, classA_x2])
    labelsA = np.ones([1,SAMPLES])  
    classB = np.vstack([classB_x1, classB_x2])
    labelsB = -np.ones([1,SAMPLES])
    
    data = np.hstack([classA, classB])
    targets = np.hstack([labelsA, labelsB])
    
    X = np.zeros([3, 2 * SAMPLES])
    t = np.zeros([1, 2 * SAMPLES])
    shuffle = np.random.permutation(2 * SAMPLES)
    
    for i in shuffle:
        X[:2,i] = data[:2, shuffle[i]] # shuffle data
        X[2,i] = 1 # add bias
        t[0,i] = targets[0, shuffle[i]]
    
    return X, t

"""
Data removal condition:
- random 50% from classA
"""
def generate_training_a(x, t, percentage):
    x_training = x
    t_training = t
    x_test = list()
    t_test = list()

    # Class A
    removal_pos_a = list()
    positions = range(x_training.shape[1])
    while (len(removal_pos_a) < round(x_training.shape[1] * percentage)):
        aux = np.random.randint(low=0, high=len(positions) - 1)
        if (t_training[0,positions[aux]] > 0):
            removal_pos_a.append(positions[aux])
            x_test.append(x_training[:,positions[aux]])
            t_test.append(t_training[:,positions[aux]])

    x_training = np.delete(x_training, removal_pos_a, axis=1)
    t_training = np.delete(t_training, removal_pos_a, axis=1)
    
    return x_training, t_training, np.transpose(np.array(x_test)), np.transpose(np.array(t_test))

"""
Data removal condition:
- random 50% from classB
"""
def generate_training_b(x, t, percentage):
    x_training = x
    t_training = t
    x_test = list()
    t_test = list()

    # Class B
    removal_pos_b = list()
    while (len(removal_pos_b) < round(x_training.shape[1] * percentage)):
        aux = np.random.randint(low=0, high=x_training.shape[1] - 1)
        if (t_training[0,aux] < 0):
            if (aux not in removal_pos_b):
                removal_pos_b.append(aux)
                x_test.append(x_training[:,aux])
                t_test.append(t_training[:,aux])

    x_training = np.delete(x_training, removal_pos_b, axis=1)
    t_training = np.delete(t_training, removal_pos_b, axis=1)
    
    return x_training, t_training, np.transpose(np.array(x_test)), np.transpose(np.array(t_test))

"""
Data removal condition:
- random 25% from each class
"""
def generate_training_a_b(x, t, percentage):
    x_training = x
    t_training = t
    x_training, t_training, x_test_1, t_test_1 = generate_training_a(x_training, t_training, percentage)
    x_training, t_training, x_test_2, t_test_2 = generate_training_b(x_training, t_training, percentage)    
    
    return x_training, t_training, np.concatenate((x_test_1, x_test_2), axis=1), np.concatenate((t_test_1, t_test_2), axis=1)

"""
Data removal condition:
- 20% from a subset of classA for which classA(1,:)<0 
    and 80% from a subset of classA for which classA(1,:)>0
"""
def generate_training_a_subsets(x, t, percentage1, percentage2):
    x_training = x
    t_training = t
    x_test = list()
    t_test = list()

    # Removing negative values

    cnt = list()
    for i in range(x_training.shape[1]):
        if (x_training[1,i] < 0 and t_training[0,i] > 0):
            cnt.append(i) # Position of a negative value

    cnt_neg = round(len(cnt) * percentage1) # Number of negative values to remove
    removal = list()
    for i in range(cnt_neg):
        pos = randrange(len(cnt)) # Get random position in the array
        removal.append(cnt[pos]) # Append the value stored in the position
        x_test.append(x_training[:,cnt[pos]])
        t_test.append(t_training[:,cnt[pos]])
        cnt = np.delete(cnt, pos) # Delete appended value

    x_training = np.delete(x_training, removal, axis=1)
    t_training = np.delete(t_training, removal, axis=1)

    # Removing positive values

    cnt = list()
    for i in range(x_training.shape[1]):
        if (x_training[1,i] > 0 and t_training[0,i] > 0):
            cnt.append(i) # Position of a positive value

    cnt_pos = round(len(cnt) * percentage2) # Number of positive values to remove
    removal = list()
    for i in range(cnt_pos):
        pos = randrange(len(cnt)) # Get random position in the array
        removal.append(cnt[pos]) # Append the value stored in the position
        x_test.append(x_training[:,cnt[pos]])
        t_test.append(t_training[:,cnt[pos]])
        cnt = np.delete(cnt, pos) # Delete appended value

    x_training = np.delete(x_training, removal, axis=1)
    t_training = np.delete(t_training, removal, axis=1)

    return x_training, t_training, np.transpose(np.array(x_test)), np.transpose(np.array(t_test))




# ---------------------------- DATA QUESTION 3.2.2. ---------------------------- #
"""
hour-glass shaped topology
simple autoencoder with 8–3–8 feed-forward architecture
only one input variable is active (=1)
"""

def enconder_data():
    X = -np.ones((8, 8))
    index = random.sample(range(X.shape[0]), 8) 
    for i in range(X.shape[0]):
        X[index[i],i] = 1     
        #X[i,i] = 1
    t = X.copy()
    X = np.vstack((X, np.ones((X.shape[1]))))
    return X, t    


# ---------------------------- DATA QUESTION 3.2.3. ---------------------------- #
"""
Bell-shaped gaussian 
"""
def gaussian_data(percentage):
    data = np.arange(-5, 5, 0.5)
    x = np.transpose(data.reshape((1,len(data))))
    y = np.transpose(data.reshape((1,len(data))))

    elements = round(x.shape[0] * percentage)
    x_training = x.copy()
    y_training = y.copy()    
    for i in range(elements):
        aux = np.random.randint(low=0, high=x_training.shape[0] - 1)
        x_training = np.delete(x_training, aux)
        y_training = np.delete(y_training, aux)
    x = np.transpose(np.reshape(x_training, (1, len(x_training))))
    y = np.transpose(np.reshape(y_training, (1, len(y_training))))
    z = np.dot(np.exp(-x*x*0.1), np.transpose(np.exp(-y*y*0.1))) - 0.5
    xx, yy = np.meshgrid(x, y)
    size = len(x)*len(y)
    xx_ = np.reshape(xx, (1, size))
    yy_ = np.reshape(yy, (1, size))
    X = np.vstack((xx_, yy_, np.ones((size))))
    t = np.reshape(z, (1,size))
    
    return X, t, xx, yy