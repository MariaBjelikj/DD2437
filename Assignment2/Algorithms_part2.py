import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from enum import Enum


class Attribute(Enum):
    Sex = 0  # Sex attribute for politicians
    Party = 1  # Party attribute for politicians
    District = 3  # District attribute for politicians


ETA = 0.2
EPOCH = 20
NEIGHBOR_DISTANCE_ANIMAL = 25
NEIGHBOR_DISTANCE_CYCLING = 2
NEIGHBOR_DISTANCE_POLITICS = 5

"""
##############################################################################
# Functions for both tasks
###############################################################################
"""


def read_data(filename, task):
    if task == "task4_1":
        animal_names = pd.read_csv(filename[0], header=None)
        animal_names = clean_names(pd.DataFrame(animal_names).iloc[:, 0].tolist())
        animals_dat = pd.read_csv(filename[1], header=None)
        animals_dat = pd.DataFrame(animals_dat).iloc[0, :].tolist()
        animals_att = pd.read_csv(filename[2], header=None)
        animals_att = pd.DataFrame(animals_att).iloc[:, 0].tolist()
        animals_dat = np.array(animals_dat).reshape((len(animal_names), len(animals_att)))

        return animal_names, animals_dat, animals_att

    elif task == "task4_2":
        data = pd.read_csv(filename[0], header=None)
        data[1].replace(regex=True, inplace=True, to_replace=r';', value=r'')
        cities = data.to_numpy()
        cities[:, 0] = [float(i) for i in cities[:, 0]]
        cities[:, 1] = [float(i) for i in cities[:, 1]]

        return cities, data

    else:
        print("Error: No task match!")
        sys.exit()


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


def get_indeces_circular(winner_position, neighbor_distance):
    negative_index = (winner_position - neighbor_distance)
    positive_index = (winner_position + neighbor_distance)

    return negative_index, positive_index


def neighborhood(winner_position, length, neighbor_distance, task):
    neighbors = list()

    if task == "task4_1":
        negative_index, positive_index = get_indeces_1dim(winner_position, length, neighbor_distance)
        for i in range(negative_index, positive_index):
            neighbors.append([0, i])
    elif task == "task4_2":
        negative_index_row, positive_index_row = get_indeces_circular(winner_position, neighbor_distance)
        for i in range(negative_index_row, positive_index_row + 1):
            # for j in range(negative_index_col, positive_index_col):
            neighbors.append([0, i])
        for i in range(len(neighbors)):
            neighbors[i][1] = neighbors[i][1] % length
    elif task == "task4_3":
        negative_index_row, positive_index_row, negative_index_col, positive_index_col = \
            get_indeces_2dim(winner_position, neighbor_distance, length)
        for i in range(negative_index_row, positive_index_row):
            for j in range(negative_index_col, positive_index_col):
                neighbors.append([i, j])
        neighbors = np.array(neighbors)
    else:
        print("Error: No task match!")
        sys.exit()

    return neighbors


def weight_update(neighbors, x, weight, task):
    if task != "task4_3":
        for i in range(len(neighbors)):
            weight[neighbors[i][1]] = weight[neighbors[i][1]] + ETA * (x - weight[neighbors[i][1]])
    else:
        for i in range(neighbors.shape[0]):
            weight[neighbors[i][0], neighbors[i][1]] = weight[neighbors[i][0], neighbors[i][1]] \
                                                       + ETA * (x - weight[neighbors[i][0], neighbors[i][1]])

    return weight


def som_algorithm(weight, data, task):
    real_distance = 0
    if task == "task4_1":
        neighbor_distance = NEIGHBOR_DISTANCE_ANIMAL
        real_distance = NEIGHBOR_DISTANCE_ANIMAL
    elif task == "task4_2":
        neighbor_distance = NEIGHBOR_DISTANCE_CYCLING
    else:
        print("Error: No task match!")
        sys.exit()

    for i in range(EPOCH):
        for j in range(data.shape[0]):
            win_pos = similarity(data[j, :], weight)
            neighbors = neighborhood(win_pos, weight.shape[0], neighbor_distance, task)
            weight = weight_update(neighbors, data[j, :], weight, task)
        if task == "task4_1":
            real_distance -= (NEIGHBOR_DISTANCE_ANIMAL - 1) / EPOCH  # This is the real distance without roundning (
            # always updates)
            neighbor_distance = round(real_distance)  # Rounding the real distance for iterations
        elif task == "task4_2":
            if i < EPOCH / 2:
                neighbor_distance = 1
                if i == EPOCH - 1:
                    neighbor_distance = 0

    return weight


def sorting(weight, data, task, names=None):
    if names is None:
        names = []
    best_combination = list()
    for j in range(data.shape[0]):
        win_pos = similarity(data[j, :], weight)
        best_combination.append(win_pos)

    if task == "task4_1":
        df = {'Animal': names, 'WeightIndex': best_combination}  # Generate data structure
    else:
        df = {'City': range(1, data.shape[0] + 1), 'WeightIndex': best_combination}

    df = pd.DataFrame(data=df)  # Generating Panda's DataFrame
    df = df.sort_values(by='WeightIndex')  # Sorting DataFrame by values of the best position ['col2']

    return df


"""
##############################################################################
# Task 1
###############################################################################
"""


def task4_1(filename, weight, task):
    animal_names, animals_dat, animals_att = read_data(filename, task)
    weight_updated = som_algorithm(weight, animals_dat, task)
    sorted_animals = sorting(weight_updated, animals_dat, task, animal_names)
    print(sorted_animals)


"""
##############################################################################
# Task 2
###############################################################################
"""


def connectpoints(x, y, p1, p2, i):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    x_mean = min(x1, x2) + (max(x1, x2) - min(x1, x2)) / 2
    y_mean = min(y1, y2) + (max(y1, y2) - min(y1, y2)) / 2
    plt.plot([x1, x2], [y1, y2], 'k-', c=np.random.rand(3, ))
    plt.annotate(str(i), xy=(x_mean, y_mean))


def plotting(dat, sorted_cities):
    data = dat.to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[:, 0] = [float(i) for i in data[:, 0]]
    data[:, 1] = [float(i) for i in data[:, 1]]
    plt.plot(data[:, 0], data[:, 1], 'ro')
    for i in range(data.shape[0]):
        ax.annotate("City" + str(i + 1), (data[i, 0], data[i, 1]))
    for i in range(sorted_cities.shape[0]):
        if i == sorted_cities.shape[0] - 1:
            connectpoints(data[:, 0], data[:, 1], sorted_cities[i, 0] - 1, sorted_cities[0, 0] - 1, i + 1)
        else:
            connectpoints(data[:, 0], data[:, 1], sorted_cities[i, 0] - 1, sorted_cities[i + 1, 0] - 1, i + 1)
    plt.xlabel("$X_{axis}$")
    plt.ylabel("$Y_{axis}$")
    plt.title("Path across cities")
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    plt.grid()
    plt.show()


def task4_2(filename, weight, task):
    cities, data = read_data(filename, task)
    weight_updated = som_algorithm(weight, cities, task)
    sorted_cities = sorting(weight_updated, cities, task)
    sorted_cities = sorted_cities.to_numpy()
    plotting(data, sorted_cities)


"""
##############################################################################
# Task 3
###############################################################################
"""


def manhattan(x, weight):
    distance_aux = np.inf
    winner_position = [0, 0]

    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            diff = abs(x - weight[i, j, :])
            distance = np.dot(np.transpose(diff), diff)
            if distance < distance_aux:
                distance_aux = distance
                winner_position = [i, j]

    return winner_position


def get_indeces_2dim(winner_position, neighbor_distance, length):
    negative_index_row = max(0, winner_position[0] - neighbor_distance)
    positive_index_row = min(length, winner_position[0] + neighbor_distance)
    negative_index_col = max(0, winner_position[1] - neighbor_distance)
    positive_index_col = min(length, winner_position[1] + neighbor_distance)

    return negative_index_row, positive_index_row, negative_index_col, positive_index_col


def sorting_task3(weight, votes, data, attribute):
    best_combiniation = list()
    for i in range(votes.shape[0]):
        best_combiniation.append(manhattan(votes[i, :], weight))
    for i in range(len(best_combiniation)):
        best_combiniation[i] = best_combiniation[i][0] * weight.shape[0] + best_combiniation[i][1]
    df = {'Attributes': data[:, attribute], 'WeightIndex': best_combiniation}
    df = pd.DataFrame(data=df)  # Generating Panda's DataFrame
    # df = df.sort_values(by='WeightIndex')  # Sorting DataFrame by values of the best position ['col2']

    return df


def task4_3(filename, weight, task):
    votes = pd.read_csv(filename[0], header=None)
    sex = pd.read_csv(filename[1], header=None)
    party = pd.read_csv(filename[2], header=None)
    names = pd.read_csv(filename[3], header=None, encoding='iso-8859-1')  # Special encoding Swedish names
    district = pd.read_csv(filename[4], header=None)
    frames = [sex, party, names, district]  # Defining the frames
    data = pd.concat(frames, axis=1)  # Generating Panda's DataFrame
    data = data.to_numpy()
    votes = np.array(votes).reshape(349, 31)

    neighbor_distance = NEIGHBOR_DISTANCE_POLITICS
    # real_distance = NEIGHBOR_DISTANCE_POLITICS
    for i in range(EPOCH):
        for j in range(votes.shape[0]):
            win_pos = manhattan(votes[j, :], weight)
            neighbors = neighborhood(win_pos, weight.shape[0], neighbor_distance, task)
            weight = weight_update(neighbors, votes[j, :], weight, task)
        # real_distance -= (NEIGHBOR_DISTANCE_POLITICS - 1) / EPOCH  # This is the real distance without roundning (
        # always updates)
        # neighbor_distance = round(real_distance)  # Rounding the real distance for iterationss

    # Sorting by different attributes
    for attribute in Attribute:
        unique_data_attributes = np.unique(data[:, attribute.value])
        dictionary = {}
        for i, k in enumerate(unique_data_attributes):
            dictionary[k] = i
        colors = np.random.rand(len(unique_data_attributes), 3)
        sorted_politics = sorting_task3(weight, votes, data, attribute.value)
        sorted_politics = sorted_politics.to_numpy()
        for i in range(sorted_politics.shape[0]):
            x = int(sorted_politics[i][1] / weight.shape[0])  # + np.random.normal(0, 0.1)  # Small noise to define cluster
            y = sorted_politics[i][1] % weight.shape[0]  # + np.random.normal(0, 0.1)  # Small noise to define cluster
            plt.scatter(x, y, c=[colors[dictionary[sorted_politics[i][0]]]])
        plt.title("Politicians classified by " + attribute.name)
        plt.show()
