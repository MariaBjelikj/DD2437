import numpy as np
import pandas as pd

ETA = 0.2
EPOCH = 20
NEIGHBOR_DISTANCE = 25


def similarity(x, weight):
    distance_aux = np.inf
    winner_position = 0

    for i in range(weight.shape[0]):
        diff = x - weight[i]
        distance = np.dot(np.transpose(diff), diff)  # monotonic function => don't need the square root
        if distance < distance_aux:
            distance_aux = distance
            winner_position = i

    return winner_position


def get_indeces_1dim(winner_position, length, neighbor_distance):
    negative_index = max(0, winner_position - neighbor_distance)
    positive_index = min(length, winner_position + neighbor_distance)

    return negative_index, positive_index


def get_indeces_2dim(winner_position, len_x, len_y, neighbor_distance):
    pos_x = winner_position[0]
    pos_y = winner_position[1]
    negative_index_row, positive_index_row = get_indeces_1dim(pos_x, len_x, neighbor_distance)
    negative_index_col, positive_index_col = get_indeces_1dim(pos_y, len_y, neighbor_distance)

    return negative_index_row, positive_index_row, negative_index_col, positive_index_col


def neighborhood(weight, winner_position, len_x, len_y, neighbor_distance):
    dimension = 1
    neighbors = list()

    if dimension == 1:
        negative_index, positive_index = get_indeces_1dim(winner_position, len_x, neighbor_distance)
        for i in range(negative_index, positive_index):
            neighbors.append([0, i])
    elif dimension == 2:
        negative_index_row, positive_index_row, negative_index_col, positive_index_col = get_indeces_2dim(
            winner_position, len_x, len_y, neighbor_distance)
        for i in range(negative_index_row, positive_index_row):
            for j in range(negative_index_col, positive_index_col):
                neighbors.append([i, j])
    else:
        print("Error: No dimension match!")
        return

    return neighbors


def weight_update(neighbors, x, weight):
    for i in range(len(neighbors)):
        weight[neighbors[i][1]] += ETA * (x - weight[neighbors[i][1]])

    return weight


def clean_names(names):
    for i in range(len(names)):
        names[i] = names[i].replace("\'", "")
        names[i] = names[i].replace("\t", "")
        names[i] = names[i].replace("\"", "")

    return names


def main():
    # Reading data
    filename = "data_lab2/animalnames.txt"
    animal_names = pd.read_csv(filename, header=None)
    animal_names = clean_names(pd.DataFrame(animal_names).iloc[:, 0].tolist())
    filename = "data_lab2/animals.dat"
    animals_dat = pd.read_csv(filename, header=None)
    animals_dat = pd.DataFrame(animals_dat).iloc[0, :].tolist()
    filename = "data_lab2/animalattributes.txt"
    animals_att = pd.read_csv(filename, header=None)
    animals_att = pd.DataFrame(animals_att).iloc[:, 0].tolist()
    animals_dat = np.array(animals_dat).reshape((len(animal_names), len(animals_att)))

    # Executing SOM algorithm
    weight = np.random.rand(100, 84)
    neighbor_distance = NEIGHBOR_DISTANCE
    real_distance = NEIGHBOR_DISTANCE
    for i in range(EPOCH):
        for j in range(len(animal_names)):
            win_pos = similarity(animals_dat[j, :], weight)
            neighbors = neighborhood(weight, win_pos, weight.shape[0], weight.shape[1], neighbor_distance)
            weight = weight_update(neighbors, animals_dat[j, :], weight)
        real_distance -= (NEIGHBOR_DISTANCE - 1) / EPOCH  # This is the real distance without roundning (always updates)
        neighbor_distance = round(real_distance)  # Rounding the real distance for iterations

    best_combination = list()
    for i in range(1):
        for j in range(len(animal_names)):
            win_pos = similarity(animals_dat[j, :], weight)
            best_combination.append(win_pos)

    data = {'col1': animal_names, 'col2': best_combination}  # Generate data structure
    data = pd.DataFrame(data=data)  # Generating Panda's DataFrame
    data = data.sort_values(by='col2')  # Sorting DataFrame by values of the best position ['col2']


if __name__ == "__main__":
    main()
