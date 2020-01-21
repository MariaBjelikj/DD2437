import numpy as np
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
    while (len(removal_pos_a) < round(SAMPLES * percentage)):
        aux = np.random.randint(low=0, high=x_training.shape[1] - 1)
        if (t_training[0,aux] == 1):
            if (aux not in removal_pos_a):
                removal_pos_a.append(aux)
                x_test.append(x_training[:,aux])
                t_test.append(t_training[:,aux])

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
    while (len(removal_pos_b) < round(SAMPLES * percentage)):
        aux = np.random.randint(low=0, high=x_training.shape[1] - 1)
        if (t_training[0,aux] == -1):
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
        if (x_training[1,i] < 0 and t_training[0,i] == 1):
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
        if (x_training[1,i] > 0 and t_training[0,i] == 1):
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