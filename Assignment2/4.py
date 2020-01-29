import numpy as np
import matplotlib.pyplot as plt


NEIGHBOR_DISTANCE = 3


'function to compute the similarity, it returns the position of the weight which has the minimal distance to x'
def similarity(x, weight):
    distance_aux = np.inf 
    winner_position = 0
    
    for i in range(weight.shape[0]):
        diff = x - weight
        distance = np.dot(np.transpose(diff), diff) # monotonic function => don't need the square root 
        if (distance < distance_aux):
            distance_aux = distance 
            winner_position = i
            
    return winner_position

def neighbourhood(weight, winner_position):
    dimension = weight.shape[1]
    index = list()
    
    
    if (dimension == 1):
        for i in range(NEIGHBOR_DISTANCE):
            
    elif (dimension == 2):
        
    else:
        print("no dimension match")
        break
    
        

    