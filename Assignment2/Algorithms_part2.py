import numpy as np
import pandas as pd

ETA = 0.2
EPOCH = 20
NEIGHBOR_DISTANCE = 2


def read_data(filename):
    animal_names = pd.read_csv(filename[0], header=None)
    animal_names = clean_names(pd.DataFrame(animal_names).iloc[:, 0].tolist())
    animals_dat = pd.read_csv(filename[1], header=None)
    animals_dat = pd.DataFrame(animals_dat).iloc[0, :].tolist()
    animals_att = pd.read_csv(filename[2], header=None)
    animals_att = pd.DataFrame(animals_att).iloc[:, 0].tolist()
    animals_dat = np.array(animals_dat).reshape((len(animal_names), len(animals_att)))

    return animal_names, animals_dat, animals_att


def clean_names(names):
    for i in range(len(names)):
        names[i] = names[i].replace("\'", "")
        names[i] = names[i].replace("\t", "")
        names[i] = names[i].replace("\"", "")
        names[i] = names[i].replace(";", "")

    return names


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


def get_indeces_circular(winner_position, length, neighbor_distance):
    negative_index = (winner_position - neighbor_distance)
    positive_index = (winner_position + neighbor_distance)

    return negative_index, positive_index


def get_indeces_2dim(winner_position, length, neighbor_distance):
    pos = winner_position
    negative_index_row, positive_index_row = get_indeces_circular(pos, length, neighbor_distance)

    return negative_index_row, positive_index_row


def neighborhood(winner_position, length, neighbor_distance, task):
    neighbors = list()

    if task == "task4_1":
        negative_index, positive_index = get_indeces_1dim(winner_position, length, neighbor_distance)
        for i in range(negative_index, positive_index):
            neighbors.append([0, i])
    elif task == "task4_2":
        negative_index_row, positive_index_row = get_indeces_2dim(
            winner_position, length, neighbor_distance)
        for i in range(negative_index_row, positive_index_row + 1):
            # for j in range(negative_index_col, positive_index_col):
            neighbors.append([0, i])
        for i in range(len(neighbors)):
            neighbors[i][1] = neighbors[i][1] % length
    else:
        print("Error: No task match!")
        return

    return neighbors


def weight_update(neighbors, x, weight):
    for i in range(len(neighbors)):
        weight[neighbors[i][1]] = weight[neighbors[i][1]] + ETA * (x - weight[neighbors[i][1]])

    return weight


def som_task4_2(weight, data, task):
    neighbor_distance = NEIGHBOR_DISTANCE
    for i in range(EPOCH):
        for j in range(data.shape[0]):
            win_pos = similarity(data[j, :], weight)
            neighbors = neighborhood(win_pos, weight.shape[0], neighbor_distance, task)
            weight = weight_update(neighbors, data[j, :], weight)
        if i < EPOCH / 2:
            neighbor_distance = 1
            if i == EPOCH - 1:
                neighbor_distance = 0

    return weight


def som_algorithm(weight, animal_names, animals_dat, task):
    neighbor_distance = NEIGHBOR_DISTANCE
    real_distance = NEIGHBOR_DISTANCE
    for i in range(EPOCH):
        for j in range(len(animal_names)):
            win_pos = similarity(animals_dat[j, :], weight)
            neighbors = neighborhood(win_pos, weight.shape[0], neighbor_distance, task)
            weight = weight_update(neighbors, animals_dat[j, :], weight)
        real_distance -= (NEIGHBOR_DISTANCE - 1) / EPOCH  # This is the real distance without roundning (always updates)
        neighbor_distance = round(real_distance)  # Rounding the real distance for iterations

    return weight


def animals_sorting(weight, animals_dat, animal_names):
    best_combination = list()
    for j in range(len(animal_names)):
        win_pos = similarity(animals_dat[j, :], weight)
        best_combination.append(win_pos)

    data = {'Animal': animal_names, 'WeightIndex': best_combination}  # Generate data structure
    data = pd.DataFrame(data=data)  # Generating Panda's DataFrame
    data = data.sort_values(by='WeightIndex')  # Sorting DataFrame by values of the best position ['col2']

    return data


def cities_sorting(weight, cities):
    best_combination = list()
    for j in range(cities.shape[0]):
        win_pos = similarity(cities[j, :], weight)
        best_combination.append(win_pos)

    data = {'City': range(1, cities.shape[0] + 1), 'WeightIndex': best_combination}  # Generate data structure
    data = pd.DataFrame(data=data)  # Generating Panda's DataFrame
    data = data.sort_values(by='WeightIndex')  # Sorting DataFrame by values of the best position ['col2']

    return data


def task4_1(filename, weight, task):
    animal_names, animals_dat, animals_att = read_data(filename)
    weight_updated = som_algorithm(weight, animal_names, animals_dat, task)
    sorted_animals = animals_sorting(weight_updated, animals_dat, animal_names)
    print(sorted_animals)


def task4_2(filename, weight, task):
    data = pd.read_csv(filename[0], header=None)
    data[1].replace(regex=True, inplace=True, to_replace=r';', value=r'')
    cities = data.to_numpy()
    cities[:, 0] = [float(i) for i in cities[:, 0]]
    cities[:, 1] = [float(i) for i in cities[:, 1]]
    weight_updated = som_task4_2(weight, cities, task)
    sorted_cities = cities_sorting(weight_updated, cities)
    print(sorted_cities)
