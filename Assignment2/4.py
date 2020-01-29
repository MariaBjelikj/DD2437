import numpy as np

NEIGHBOR_DISTANCE = 3
ETA = 0.2

'function to compute the similarity, it returns the position of the weight which has the minimal distance to x'


def similarity(x, weight):
    distance_aux = np.inf
    winner_position = 0

    for i in range(weight.shape[0]):
        diff = x - weight
        distance = np.dot(np.transpose(diff), diff)  # monotonic function => don't need the square root
        if distance < distance_aux:
            distance_aux = distance
            winner_position = i

    return winner_position


def get_indeces_1dim(winner_position, length):
    negative_index = 0
    positive_index = 0

    if winner_position - NEIGHBOR_DISTANCE < 0:
        negative_index = 0
    if winner_position + NEIGHBOR_DISTANCE > length:
        positive_index = length

    return negative_index, positive_index


def get_indeces_2dim(winner_position, len_x, len_y):
    pos_x = winner_position[0]
    pos_y = winner_position[1]
    negative_index_row, positive_index_row = get_indeces_1dim(pos_x, len_x)
    negative_index_col, positive_index_col = get_indeces_1dim(pos_y, len_y)

    return negative_index_row, positive_index_row, negative_index_col, positive_index_col


def neighbourhood(weight, winner_position, len_x, len_y):
    dimension = weight.shape[1]
    neighbors = list()

    if dimension == 1:
        negative_index, positive_index = get_indeces_1dim(winner_position[0], len_x)
        for i in range(negative_index, positive_index + 1):
            neighbors.append([0, i])
    elif dimension == 2:
        negative_index_row, positive_index_row, negative_index_col, positive_index_col = get_indeces_2dim(
            winner_position, len_x, len_y)
        for i in range(negative_index_row, positive_index_row + 1):
            for j in range(negative_index_col, positive_index_col + 1):
                neighbors.append([i, j])
    else:
        print("no dimension match")
        return

    return neighbors


def weight_update(neighbors, x, weight):
    for i in range(len(neighbors)):
        for j in range(len(neighbors)):
            weight[neighbors[i, j]] += ETA * (x - weight[neighbors[i, j]])

    return weight
