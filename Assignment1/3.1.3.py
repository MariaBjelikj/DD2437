import numpy as np
import matplotlib.pyplot as plt
import time
import DataGeneration as dg
import Algorithms as alg
import Constants as cte

def main():
    
    # -------------------- FIRST PART -> Linear Separation for non-linearly separable data -------------------- #
    m = [1.0, 0.5]
    sigma = 0.5
    x, t = dg.non_linearly_separable_data(m, sigma)
    x_grid = np.arange(min(x[0,:]), max(x[0,:]), (max(x[0,:]) - min(x[0,:])) / cte.SAMPLES)
    w = np.random.rand(t.shape[0], x.shape[0])
    print("Running algorithms for: Non-linearly separable data")
    alg.run_algorithms(x_grid, x, t, w)

    # -------------------- SECOND PART -> Separation of non-linearly separable data -------------------- #
    
    # Case: Original new data
    m_a = [1.0, 0.3]
    m_b = [0.0, -0.1]
    sigma_a = 0.2
    sigma_b = 0.3
    x, t = dg.new_data_generation(m_a, m_b, sigma_a, sigma_b)

    x_grid = np.arange(min(x[0,:]), max(x[0,:]), (max(x[0,:]) - min(x[0,:])) / cte.SAMPLES)
    w = np.random.rand(t.shape[0], x.shape[0])
    print("Running algorithms for: New generated non-linearly separable data")
    alg.run_algorithms(x_grid, x, t, w)

    # Case: Remove 25% from each class
    x_temp = x.copy()
    t_temp = t.copy()
    x_temp, t_temp, x_test, t_test = dg.generate_training_a_b(x_temp, t_temp, 0.25)
    x_grid = np.arange(min(x_temp[0,:]), max(x_temp[0,:]), (max(x_temp[0,:]) - min(x_temp[0,:])) / cte.SAMPLES)
    # w can be equal, since t.shape[0] and x.shape[0] will not change (only column removal)
    print("Running algorithms for: New generated non-linearly separable data --> 25% removed in each class")
    alg.run_algorithms(x_grid, x_temp, t_temp, w)

    # Case: Remove 50% from class A
    x_temp = x.copy()
    t_temp = t.copy()
    x_temp, t_temp, x_test, t_test = dg.generate_training_a(x_temp, t_temp, 0.5)
    x_grid = np.arange(min(x_temp[0,:]), max(x_temp[0,:]), (max(x_temp[0,:]) - min(x_temp[0,:])) / cte.SAMPLES)
    # w can be equal, since t.shape[0] and x.shape[0] will not change (only column removal)
    print("Running algorithms for: New generated non-linearly separable data --> 50% removed in class A")
    alg.run_algorithms(x_grid, x_temp, t_temp, w)

    # Case: Remove 50% from class B
    x_temp = x.copy()
    t_temp = t.copy()
    x_temp, t_temp, x_test, t_test = dg.generate_training_b(x_temp, t_temp, 0.5)
    x_grid = np.arange(min(x_temp[0,:]), max(x_temp[0,:]), (max(x_temp[0,:]) - min(x_temp[0,:])) / cte.SAMPLES)
    # w can be equal, since t.shape[0] and x.shape[0] will not change (only column removal)
    print("Running algorithms for: New generated non-linearly separable data --> 50% removed in class B")
    alg.run_algorithms(x_grid, x_temp, t_temp, w)

    # Case: Remove 20% from negative subset class A, 80% from positive subset class B
    x_temp = x.copy()
    t_temp = t.copy()
    x_temp, t_temp, x_test, t_test = dg.generate_training_a_subsets(x_temp, t_temp, 0.2, 0.8)
    x_grid = np.arange(min(x_temp[0,:]), max(x_temp[0,:]), (max(x_temp[0,:]) - min(x_temp[0,:])) / cte.SAMPLES)
    # w can be equal, since t.shape[0] and x.shape[0] will not change (only column removal)
    print("Running algorithms for: New generated non-linearly separable data --> subsets removed in class A")
    alg.run_algorithms(x_grid, x_temp, t_temp, w)

if __name__ == "__main__":
    main()