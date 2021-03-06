import numpy as np
import Algorithms_part2 as alg

# TASK 4.1 - animal sorting
filenames = ["data_lab2/animalnames.txt", "data_lab2/animals.dat", "data_lab2/animalattributes.txt"]
weight = np.random.rand(100, 84)
alg.task4_1(filenames, weight, task="task4_1")
input("Press Enter to continue...")

# TASK 4.2 - cyclic tour
filenames = ["data_lab2/cities.dat"]
weight = np.random.rand(10, 2)
alg.task4_2(filenames, weight, task="task4_2")
input("Press Enter to continue...")

# TASK 4.3 - politicians classification
filenames = ["data_lab2/votes.dat", "data_lab2/mpsex.dat", "data_lab2/mpparty.dat",
             "data_lab2/mpnames.txt", "data_lab2/mpdistrict.dat"]
weight = np.random.rand(10, 10, 31)
alg.task4_3(filenames, weight, task="task4_3")
