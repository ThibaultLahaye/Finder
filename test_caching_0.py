"""
Testing the effect of caching on the performance of the fitness function.
"""

import numpy as np
from numba import jit
import time

@jit(nopython=True)
def fitness_opt(permutation: np.ndarray, distance_matrix: np.ndarray) -> float:
    num_cities = distance_matrix.shape[0]

    inf_mask = np.isinf(distance_matrix[permutation[:-1], permutation[1]])

    if np.any(inf_mask):
        return np.inf

    cost = np.sum(distance_matrix[permutation[:-1], permutation[1]])
    cost += distance_matrix[permutation[-1], permutation[0]]

    return cost

@jit(nopython=True, parallel=False)
def fitness(permutation: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Calculate the cost of a permutation in the Traveling Salesman Problem.

    First, the cache is checked for the cost of the given permutation.
    If the cost is not cached, the cost is calculated, returned and cached.

    Checks for np.inf values during each iteration of the loop. 
    If an infinite distance is encountered, the cost is set to np.inf, 
    and the loop is broken out of. This avoids unnecessary computation if 
    a city in the permutation is not connected to the next city.

    The keyword argument 'parallel=True' was specified but no transformation 
    for parallel execution was possible.

    Parameters:
    - distance_matrix (numpy.ndarray): A 2D array representing the distances between cities.
    - permutation (numpy.ndarray): A 1D array representing a permutation of cities.

    Returns:
    - float: The cost of the given permutation.
    """
    

    num_cities = distance_matrix.shape[0]

    cost = 0.0
    for i in range(num_cities - 1):
        from_city = permutation[i]
        to_city = permutation[i + 1]

        if np.isinf(distance_matrix[from_city, to_city]):
            cost = np.inf
            break

        cost += distance_matrix[from_city, to_city]

    cost += distance_matrix[permutation[-1], permutation[0]]

    return cost

@jit(nopython=True)
def rotate_0_up_front(permutation: np.ndarray) -> np.ndarray:
    idx = np.argmax(permutation == 0)
    return np.roll(permutation, -idx)

fitness_cache = dict()
def fitness_cached(permutation: np.ndarray, distance_matrix: np.ndarray) -> float:
    fitness_cache_key = tuple(rotate_0_up_front(permutation))

    if fitness_cache_key in fitness_cache:
        print("hit")
        return fitness_cache[fitness_cache_key]

    cost = fitness(permutation, distance_matrix)

    fitness_cache[fitness_cache_key] = cost

    return fitness
    

def test_already_cached(distance_matrix):
    permutation = np.random.permutation(distance_matrix.shape[0])
    fitness(permutation, distance_matrix) # Warm up the function (JIT compilation)

    start = time.time()
    fitness_cached(permutation, distance_matrix)
    fitness_cached(permutation, distance_matrix)
    end = time.time()

    print(f"Time taken: {end - start} seconds")

    start = time.time()
    fitness(permutation, distance_matrix)
    fitness(permutation, distance_matrix)
    end = time.time()   

    print(f"Time taken: {end - start} seconds")

def test_cached(distance_matrix):
    permutation = np.random.permutation(distance_matrix.shape[0])
    fitness(permutation, distance_matrix) # Warm up the function (JIT compilation)

    start = time.time()
    for i in range(1000000):
        permutation = np.random.permutation(distance_matrix.shape[0])
        fitness_cached(permutation, distance_matrix)
    end = time.time()

    print(f"Time taken: {end - start} seconds")

    start = time.time()
    for i in range(1000000):
        permutation = np.random.permutation(distance_matrix.shape[0])
        fitness(permutation, distance_matrix)
    end = time.time()

    print(f"Time taken: {end - start} seconds")

def test_fitness_opt(distance_matrix):
    permutation = np.random.permutation(distance_matrix.shape[0])
    fitness(permutation, distance_matrix) # Warm up the function (JIT compilation)
    fitness_opt(permutation, distance_matrix) # Warm up the function (JIT compilation)

    start = time.time()
    for i in range(1000000):
        permutation = np.random.permutation(distance_matrix.shape[0])
        fitness_opt(permutation, distance_matrix)
    end = time.time()

    print(f"Time taken: {end - start} seconds")

    start = time.time()
    for i in range(1000000):
        permutation = np.random.permutation(distance_matrix.shape[0])
        fitness(permutation, distance_matrix)
    end = time.time()

    print(f"Time taken: {end - start} seconds")


if __name__ == "__main__":
    
    file = open("tours/tour1000.csv")
    distance_matrix = np.loadtxt(file, delimiter=",")
    file.close()

    test_fitness_opt(distance_matrix)


    



    