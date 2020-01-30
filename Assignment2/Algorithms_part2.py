import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ETA = 0.2
EPOCH = 20
NEIGHBOR_DISTANCE_ANIMAL = 25
NEIGHBOR_DISTANCE_CYCLING = 2



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
    
    if task == "task4_2":
        data = pd.read_csv(filename[0], header=None)
        data[1].replace(regex=True, inplace=True, to_replace=r';', value=r'')
        cities = data.to_numpy()
        cities[:, 0] = [float(i) for i in cities[:, 0]]
        cities[:, 1] = [float(i) for i in cities[:, 1]]
        
        return cities, data


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


def som_algorithm(weight, data, task):
    if task == "task4_1":
        neighbor_distance = NEIGHBOR_DISTANCE_ANIMAL
        real_distance = NEIGHBOR_DISTANCE_ANIMAL
    elif task == "task4_2":
        neighbor_distance = NEIGHBOR_DISTANCE_CYCLING
    
    for i in range(EPOCH):
        for j in range(data.shape[0]):
            win_pos = similarity(data[j, :], weight)
            neighbors = neighborhood(win_pos, weight.shape[0], neighbor_distance, task)
            weight = weight_update(neighbors, data[j, :], weight)
        
        if task == "task4_1":
            real_distance -= (NEIGHBOR_DISTANCE_ANIMAL - 1) / EPOCH  # This is the real distance without roundning (always updates)
            neighbor_distance = round(real_distance)  # Rounding the real distance for iterations
        elif task == "task4_2":
            if i < EPOCH / 2:
                neighbor_distance = 1
                if i == EPOCH - 1:
                    neighbor_distance = 0

    return weight


def sorting(weight, data, task,names=[]):
    best_combination = list()
    for j in range(data.shape[0]):
        win_pos = similarity(data[j, :], weight)
        best_combination.append(win_pos)        

    if task == "task4_1":
        data = {'Animal': names, 'WeightIndex': best_combination}  # Generate data structure
    elif task == "task4_2":
        data = {'City': range(1, data.shape[0] + 1), 'WeightIndex': best_combination}
    
    data = pd.DataFrame(data=data)  # Generating Panda's DataFrame
    data = data.sort_values(by='WeightIndex')  # Sorting DataFrame by values of the best position ['col2']

    return data

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
    plt.plot([x1, x2], [y1, y2], 'k-', c=np.random.rand(3,))
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
    for i in range(sorted_cities.shape[0] - 1):
        connectpoints(data[:, 0], data[:, 1], sorted_cities[i, 0] - 1, sorted_cities[i + 1, 0] - 1, i + 1)
    plt.xlabel("$X_{axis}$")
    plt.ylabel("$Y_{axis}$")
    plt.title("Path across cities")
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


def task4_3(filename, weight, task):
    votes = pd.read_csv(filename[0], header=None)
    sex = pd.read_csv(filename[1], header=None)
    party = pd.read_csv(filename[2], header=None)
    names = pd.read_csv(filename[3], header=None, encoding='utf_8')
    district = pd.read_csv(filename[4], header=None)
    

