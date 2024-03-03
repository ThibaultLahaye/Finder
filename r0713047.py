""" General Problem Description
This file contains the implementation of an evolutionary algorithm for the Traveling Salesman Problem (TSP).
A distance matrix is read from a tour[length].cvs file, where length is the number of cities in the problem.
The distance matrix is a 2D array representing the distances between cities. Unreachable cities are represented
by np.inf values. 
"""

""" Still to Complete
TODO: Complete the implementation of the different recombination operators.
TODO: Complete the implementation of the main optimization loop.
"""

""" Notes

# Caching
Testing showed that fitness_cached() is slower than fitness() for the given tour files.
Even when using rotate_0_up_front() to reduce the number of permutations that need to be cached.
This was mainly due to a low hit rate of the cache, for larger tour files (1000).
Even when evaluating a 
"""
import Reporter
import numpy as np
from numba import jit
import time # For testing

np.set_printoptions(edgeitems=10,linewidth=180) # For testing

INF = 10000000000.0

# Utility Functions 

@jit(nopython=True)
def rotate_0_up_front(permutation: np.ndarray) -> np.ndarray:
    idx = np.argmax(permutation == 0)
    return np.roll(permutation, -idx)

@jit(nopython=True)
def generate_report_values(population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
    
    # Best fitness & Best permutation
    min_idx = np.argmin(fitness_values)
    best_fitness = fitness_values[min_idx]
    best_permutation = rotate_0_up_front(population[min_idx])

    # Mean fitness 
    # fitness_values may include np.inf values, which are not included in the means
    mean_fitness = np.mean(fitness_values[fitness_values != np.inf])

    return mean_fitness, best_fitness, best_permutation

# Permutation fitness function

@jit(nopython=True)
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
    
    length = permutation.size
    assert length == distance_matrix.shape[0]

    cost = 0.0
    for i in range(length-1):
        from_city = permutation[i]
        to_city   = permutation[i+1]

        distance = distance_matrix[from_city, to_city]
        if np.isinf(distance):
            cost = np.inf
            break

        cost += distance

    cost += distance_matrix[permutation[-1], permutation[0]]

    return cost

@jit(nopython=True)
def fitnesses(population: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    pop_size = population.shape[0]

    fitness_values = np.empty(pop_size, dtype=np.float64)

    for i in np.arange(pop_size):
        fitness_values[i] = fitness(population[i], distance_matrix)

    return fitness_values

# Distance Functions

@jit(nopython=True)
def cyclic_edge_distance(permutation_1: np.ndarray, permutation_2: np.ndarray) -> np.int64: # Testing
    """
    Calculates the cyclic edge distance between two permutations.

    Cyclic edge distance treats permutations as sets of undirected edges, considering the last element
    connected to the first. For example, the permutation [1, 5, 2, 4, 0, 3] is equivalent to the set of
    edges {(1,5), (5,2), (2,4), (4,0), (0,3), (3,1)}.

    The distance between two permutations is the count of edges that differ. It accounts for undirected
    edges, meaning (i, j) is considered the same as (j, i).

    :param permutation_1: First permutation.
    :param permutation_2: Second permutation.

    :return: Cyclic edge distance between the two permutations.

    Runtime: O(n), where n is the length of the permutations.
    """

    count_non_shared_edges = np.int64(0)
    
    # successors2 = np.empty_like(p2)
    # for i in range(successors2.size):
    #     successors2[p2[i]] = p2[(i + 1) % successors2.size]
    successors2 = np.roll(permutation_2, -1)

    for i in range(successors2.size):
        j = (i + 1) % successors2.size
        if permutation_1[j] != successors2[permutation_1[i]] and permutation_1[i] != successors2[permutation_1[j]]:
            count_non_shared_edges += np.int64(1)

    return count_non_shared_edges

@jit(nopython=True)
def cyclic_rtype_distance(permutation_1: np.ndarray, permutation_2: np.ndarray) -> np.int64:
    """
    Calculates the Cyclic RType distance between two permutations.

    Cyclic RType distance treats permutations as sets of directed edges, considering the last element
    connected to the first. For example, the permutation [1, 5, 2, 4, 0, 3] is equivalent to the set of
    directed edges: {(1,5), (5,2), (2,4), (4,0), (0,3), (3,1)}.

    The distance between two permutations is the count of directed edges that differ.

    :param permutation_1: First permutation.
    :param permutation_2: Second permutation.

    :return: Cyclic RType distance between the two permutations.

    Runtime: O(n), where n is the length of the permutations.
    """
    assert  permutation_1.size == permutation_2.size

    count_non_shared_edges = 0
    successors_2 = np.empty_like(permutation_2)

    for i in range(len(successors_2)):
        successors_2[permutation_2[i]] = permutation_2[(i + 1) % len(successors_2)]

    for i in range(len(successors_2)):
        if permutation_1[(i + 1) % len(successors_2)] != successors_2[permutation_1[i]]:
            count_non_shared_edges += 1

    return count_non_shared_edges

# Initialization Operators

@jit(nopython=True)
def random_permutation(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate a random permutation of cities.

    Parameters:
    - distance_matrix (np.ndarray): A 2D array representing the distances between cities.

    Returns:
    - np.ndarray: A 1D array representing a permutation of cities.
    """

    num_cities = distance_matrix.shape[0]
    permutation = np.random.permutation(num_cities)
    fitness_value = fitness(permutation, distance_matrix)

    return permutation, fitness_value

@jit(nopython=True)
def valid_permutation(distance_matrix: np.ndarray) -> np.ndarray:
    permutation = np.empty(distance_matrix.shape[0], dtype=np.int64)
    cities = np.arange(distance_matrix.shape[0])

    attempts = 0
    while attempts < 5:

        permutation[0] = np.random.randint(0, distance_matrix.shape[0])
        fitness_value = 0.0
        for i in range(1, distance_matrix.shape[0]):
            
            neighbours_mask = np.isfinite(distance_matrix[permutation[i-1]]) #
            neighbours_mask[permutation[:i]] = False # 
            neighbours = cities[neighbours_mask] # 

            # check if no neighbours left
            if neighbours.size == 0:
                print("no neighbours left")
                attempts += 1
                break

            permutation[i] = np.random.choice(neighbours) # select random neighbour
            fitness_value += distance_matrix[permutation[i-1], permutation[i]] # add distance to fitness

            # check if last city is connected to first city
            if i == distance_matrix.shape[0] - 1:
                distance = distance_matrix[permutation[i], permutation[0]]
                fitness_value += distance
                if distance == np.inf:
                    print("last city not connected to first city")
                    attempts += 1
                    break 
        
        else: 
            return permutation, fitness_value

    # Last resort: return random permutation
    print("last resort")
    permutation = np.random.permutation(distance_matrix.shape[0])
    fitness_value = fitness(permutation, distance_matrix)
    return permutation, fitness_value

@jit(nopython=True)
def greedy_permutation(distance_matrix: np.ndarray) -> np.ndarray:
    permutation = np.empty(distance_matrix.shape[0], dtype=np.int64)
    cities = np.arange(distance_matrix.shape[0])

    attempts = 0
    while attempts < 5:

        permutation[0] = np.random.randint(0, distance_matrix.shape[0])
        fitness_value = 0.0
        for i in range(1, distance_matrix.shape[0]):
            
            prev_city = permutation[i-1]

            neighbours_mask = np.isfinite(distance_matrix[prev_city])
            neighbours_mask[permutation[:i]] = False
            neighbours = cities[neighbours_mask]

            if neighbours.size == 0:
                print("no neighbours left")
                attempts += 1
                break

            neighbours_r_idx = np.argmin(distance_matrix[prev_city][neighbours])

            permutation[i] = neighbours[neighbours_r_idx]
            fitness_value += distance_matrix[prev_city, permutation[i]]

            if i == distance_matrix.shape[0] - 1:
                distance = distance_matrix[permutation[i], permutation[0]]
                fitness_value += distance
                if distance == np.inf:
                    print("last city not connected to first city")
                    attempts += 1
                    break
        
        else: 
            return permutation, fitness_value

    # Last resort: return random permutation
    print("last resort")
    permutation = np.random.permutation(distance_matrix.shape[0])
    fitness_value = fitness(permutation, distance_matrix)
    return permutation, fitness_value
        
@jit(nopython=True)
def initialize_population(distance_matrix: np.ndarray, lambda_: np.int64) -> np.int64: 
    
    random_number = int(lambda_*0.0)
    greedy_number = int(lambda_*0.20)
    valid_number  = int(lambda_*0.80)
    print(random_number, greedy_number, valid_number)
    assert(random_number + greedy_number + valid_number == lambda_)

    population = np.empty((lambda_, distance_matrix.shape[0]), dtype=np.int64)
    fitness_values = np.empty(lambda_, dtype=np.float64)
    
    idx = np.random.permutation(lambda_)

    for r in np.arange(0, random_number):
        population[idx[r]], fitness_values[idx[r]] = random_permutation(distance_matrix)
    
    for g in np.arange(random_number, greedy_number):
        population[idx[g]], fitness_values[idx[g]] = greedy_permutation(distance_matrix)

    for v in np.arange(greedy_number, lambda_):
        population[idx[v]], fitness_values[idx[v]] = valid_permutation(distance_matrix)

    return population, fitness_values

# Selection Operators

@jit(nopython=True, parallel=True)
def k_tournament_selection (population: np.ndarray, fitness_values: np.ndarray, k:np.int64) -> np.ndarray:
    """
    Perform k-tournament selection on a population based on their fitness values.

    Parameters:
    - population (np.ndarray): 2D array representing the population, where each row is an individual.
    - fitness_values (np.ndarray): 1D array containing the fitness values corresponding to each individual in the population.
    - k (np.int64): The size of the tournament.

    Returns:
    - np.ndarray: 1D array representing the selected individual with the highest fitness in the tournament.

    Tournament Selection Process:
    Randomly select 'k' individuals from the population.
    The individual with the highest fitness among the selected individuals is chosen as the winner.

    Example:
    Population:
    [[1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10],
     [11, 12, 13, 14, 15],
     [16, 17, 18, 19, 20]]

    Fitness Values: [20, 15, 18, 12]

    Tournament Selection with k=3:
    Randomly selected indices: [2, 0, 1]

    Selected Individuals:
    [[11, 12, 13, 14, 15],
     [1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10]]

    Winner: [11, 12, 13, 14, 15]
    """

    tournament_indices = np.random.choice(population.shape[0], k, replace=False)
    winner_index = np.argmin(fitness_values[tournament_indices])
    tournament_winner = population[tournament_indices][winner_index]
    
    return tournament_winner

# Mutation Operators

@jit(nopython=False)
def inversion_mutation(permutation: np.ndarray, alpha_: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    """
    Perform inversion mutation on a permutation array within a specified window.

    Parameters:
    - permutation (numpy.ndarray): The input permutation array.
    - start (numpy.int64): The starting index of the mutation window.
    - end (numpy.int64): The ending index of the mutation window.
    - alpha_ (numpy.float64): The probability of performing the inversion mutation.

    Returns:
    - numpy.ndarray: The mutated permutation array.

    ASCII Art Visualization:
    
    Original Permutation: [1, 2, 3, 4, 5, 6, 7, 8, 9]

    Mutation Window:      |-------[        ]--------|
                              start        end

    Mutation Process:
    If random probability (alpha_) is met, a subset within the window is inverted.

    Example:
    Original Permutation: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    Mutation Window:      |-------[        ]--------|
                            start           end

    Subset to be inverted:         [4, 5, 6]

    Mutated Permutation: [1, 2, 3, 6, 5, 4, 7, 8, 9]
    """
    
    if np.random.random() <= alpha_:
        num_cities = distance_matrix.shape[0]

        start_idx = np.random.randint(0, num_cities)
        size = np.random.randint(2, num_cities)

        idx = np.arange(start_idx, start_idx + size)
        subset = permutation[idx % num_cities]

        mutated_permutation = permutation.copy()
        mutated_permutation[idx % num_cities]  = subset[::-1]
                        
        return mutated_permutation
    
    return permutation

@jit(nopython=True)
def scramble_mutation(permutation: np.ndarray, alpha_: np.float64, distance_matrix: np.ndarray) -> np.ndarray:
    """Scramble mutation: randomly choose 2 indices and scramble that subsequence."""   

    if np.random.random() < alpha_:
        num_cities = distance_matrix.shape[0]

        start_idx = np.random.randint(0, num_cities)
        size = np.random.randint(2, num_cities-1)

        idx = np.arange(start_idx, start_idx + size)
        subset = permutation[idx % num_cities]
        np.random.shuffle(subset)

        new_order = permutation.copy()
        new_order[idx % num_cities] = subset

        return new_order

    return permutation

@jit(nopython=True)
def mutate_population(population: np.ndarray, alpha_: np.float64, distance_matrix: np.ndarray) -> np.ndarray:
    """Apply mutation to each individual in the population."""
    mutated_population = np.empty(population.shape, dtype=np.int64)

    for i in np.arange(0, population.shape[0]):
        mutated_population[i] = scramble_mutation(population[i], alpha_, distance_matrix)

    return mutated_population

# Crossover Operators

@jit(nopython=True)
def uniform_crossover(parent_1: np.ndarray, parent_2: np.ndarray) -> np.ndarray: # Testing
    """
    https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#:~:text=k%20crossover%20points.-,Uniform%20crossover,-%5Bedit%5D
    """
    assert len(parent1) == len(parent2)

    child_genome = np.empty_like(parent1)

    # Randomly select genes from each parent with equal probability
    for i in range(len(parent1)):
        child_genome[i] = np.random.choice([parent1[i], parent2[i]])

    return child_genome

@jit(nopython=True)
def ox_crossover(parent_1: np.ndarray, parent_2: np.ndarray) -> np.ndarray: 
    """
    A random subset of the parent_1 is copied to the child. The remaining positions are filled
    with the values from parent_2 that are not already in the child. The values are copied from
    parent_2 in the same order as they appear in parent_2.
    This inplementation can also handle wrap-around, meaning that the subset can start at the
    last index of parent_1 and wrap around to the first index.  
    """
    assert parent_1.size == parent_2.size

    num_cities = parent_1.size

    start_idx = np.random.randint(0, num_cities)
    size = np.random.randint(0, num_cities)

    idx = np.arange(start_idx, start_idx + size)

    child = np.full(num_cities , fill_value=-1)

    child[idx % num_cities] = parent_1[idx % num_cities]
    child[child == -1] = [i for i in parent_2 if i not in child]

    return child

# Local Search Operators

@jit(nopython=True)
def swap(permutation: np.ndarray, idx_1: np.int64, idx_2: np.int64 , distance_matrix: np.ndarray) -> np.float64:
    if idx_2-idx_1 == 1:
        return (  
            - distance_matrix[permutation[idx_1-1]][permutation[idx_1]]
            - distance_matrix[permutation[idx_1]][permutation[idx_2]]
            - distance_matrix[permutation[idx_2]][permutation[idx_2+1]]
            + distance_matrix[permutation[idx_1-1]][permutation[idx_2]]
            + distance_matrix[permutation[idx_2]][permutation[idx_1]]
            + distance_matrix[permutation[idx_1]][permutation[idx_2+1]]
        )
    else:
        return (
            - distance_matrix[permutation[idx_1-1]][permutation[idx_1]]
            - distance_matrix[permutation[idx_1]][permutation[idx_1+1]]
            - distance_matrix[permutation[idx_2-1]][permutation[idx_2]]
            - distance_matrix[permutation[idx_2]][permutation[idx_2+1]]
            + distance_matrix[permutation[idx_1-1]][permutation[idx_2]]
            + distance_matrix[permutation[idx_2]][permutation[idx_1+1]]
            + distance_matrix[permutation[idx_2-1]][permutation[idx_1]]
            + distance_matrix[permutation[idx_1]][permutation[idx_2+1]]
        )

@jit(nopython=True)
def one_opt_local_search(permutation: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:

    num_cities = distance_matrix.shape[0]
    permutation_fitness = fitness(permutation, distance_matrix)

    if permutation_fitness == np.inf: return permutation, permutation_fitness

    best_delta = 0
    best_indices = 0, 0
    for idx_1 in range(1, num_cities - 1):
        for idx_2 in range(idx_1 + 1, num_cities - 1):
            new_delta = swap(permutation, idx_1, idx_2, distance_matrix)

            if new_delta < best_delta:
                best_delta = new_delta
                best_indices = (idx_1, idx_2)

    permutation[best_indices[0]], permutation[best_indices[1]] = permutation[best_indices[1]],permutation[best_indices[0]]

    return permutation, permutation_fitness + best_delta

@jit(nopython=True)
def two_opt_local_search_fast(permutation: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    num_cities = distance_matrix.shape[0]
    old_fitness = fitness(permutation, distance_matrix)

    init_costs = np.empty(num_cities, dtype=np.float64) 
    init_costs[0] = 0
    for i in np.arange(1, num_cities-1):
        init_costs[i] = init_costs[i-1] + \
            distance_matrix[permutation[i-1], permutation[i]]

    tail_costs = np.empty(num_cities, dtype=np.float64)
    tail_costs[num_cities-1] = distance_matrix[permutation[-1], permutation[0]]
    for i in np.arange(1, num_cities-1):
        tail_costs[num_cities - i - 1] = tail_costs[num_cities - i] + \
            distance_matrix[permutation[num_cities - i -1], permutation[num_cities - i]]

    # Find the optimal start and end index
    opt_start_idx = 0
    opt_end_idx   = 0
    opt_fitness   = old_fitness
    for start_idx in np.arange(1, num_cities-2):
        init_fitness = init_costs[start_idx-1]

        sub_fitness = 0.0
        for end_idx in np.arange(start_idx+2, num_cities):

            sub_fitness += distance_matrix[permutation[end_idx-1], permutation[end_idx-2]]

            if sub_fitness >= opt_fitness: break

            tail_fitness = tail_costs[end_idx]

            if tail_fitness >= opt_fitness: continue

            init_sub_cost = distance_matrix[permutation[start_idx-1], permutation[end_idx-1]]
            sub_tail_cost = distance_matrix[permutation[start_idx], permutation[end_idx]]

            if (init_sub_cost >= opt_fitness) or (sub_tail_cost >= opt_fitness) : continue

            new_fitness = init_fitness + init_sub_cost + sub_fitness + sub_tail_cost + tail_fitness 

            if new_fitness < old_fitness:
                opt_start_idx = start_idx
                opt_end_idx = end_idx
                opt_fitness = new_fitness

    # Apply the optimal 2-opt move
    if opt_fitness < old_fitness:
        new_order = np.empty_like(permutation)
        new_order[:opt_start_idx] = permutation[:opt_start_idx]
        new_order[opt_start_idx:opt_end_idx] = permutation[opt_start_idx:opt_end_idx][::-1]
        new_order[opt_end_idx:] = permutation[opt_end_idx:]
        return new_order, opt_fitness
    else:
        return permutation, old_fitness


@jit(nopython=True)
def find_elites(population: np.ndarray, fitness_values: np.ndarray, num_elites: np.int64) -> np.ndarray:
    """
    Find the num_elites individuals with the best fitness values in the population.

    Parameters:
    - population (numpy.ndarray): A 2D matrix representing the population.
    - fitness_values (numpy.ndarray): A 1D array containing the fitness values of the population.
    - num_elites (numpy.int64): The number of elites to find.

    Returns:
    - numpy.ndarray: A 2D matrix representing the elites.
    """
    sorted_indices = np.argsort(fitness_values)

    elites = population[sorted_indices][:num_elites]
    elite_fitness_values = fitness_values[sorted_indices][:num_elites]

    return elites, elite_fitness_values

@jit(nopython=True)
def add_elites(population: np.ndarray, fitness_values: np.ndarray, elites: np.ndarray, elite_fitness_values: np.ndarray) -> np.ndarray:
    """
    First the population is sorted according to the fitness values. Then the elites are added to the population, only if they are not already in the population.
    Elites that are not already in the population are added to the population, by replacing the worst individuals in the population.
    """
    # Sort the population and fitness values
    sorted_indices = np.argsort(fitness_values)
    population = population[sorted_indices]
    fitness_values = fitness_values[sorted_indices]

    for i in np.arange(0, elites.shape[0]):
        elite_fitness_value = elite_fitness_values[i]
        elite_present = False

        for j in np.arange(0, population.shape[0]):

            for j in np.arange(0, population.shape[0]):
                if elite_fitness_value <= fitness_values[j] and np.array_equal(elites[i], population[j]):
                    elite_present = True
                    break

        if not elite_present:
            population[-i-1] = elites[i]
            fitness_values[-i-1] = elite_fitness_values[i]

    return population, fitness_values
    
# Fitness Sharing

@jit(nopython=True)
def fitness_sharing(fitness_values: np.ndarray, 
                    population: np.ndarray,
                    shape_: np.float64, 
                    sigma_: np.float64) -> np.ndarray: #Testing
    """
    This inplementation leverages the fact that the cyclic r-type distance between two permutations
    is symmetric. This means that the distance between permutation_1 and permutation_2 is the same and 
    dist_matrix[i][j] equals dist_matrix[j][i].

    population (numpy.ndarray): A 2D matrix representing the population and offspring.
    fitness_values (numpy.ndarray): A 1D array containing the fitness values of the population and offspring.
    """
    INF =  10000000000.0
    num_permutations = population.shape[0]

    dist_matrix = np.zeros((num_permutations, num_permutations), dtype=np.float64)
    shared_fitness_values = np.zeros(num_permutations, dtype=np.float64)

    for i in np.arange(0, num_permutations):
        for j in np.arange(0, num_permutations):

            if j >= i: 
                dist_matrix[i, j] = cyclic_rtype_distance(population[i], population[j])

                if dist_matrix[i, j] <= sigma_:
                    correction = 1 - (dist_matrix[i, j] / sigma_)**shape_
                    fitness = fitness_values[i]
                    if np.isinf(fitness):
                        fitness = INF

                    shared_fitness_values[i] += fitness*correction

            else:
                if dist_matrix[j, i] <= sigma_:
                    correction = 1 - (dist_matrix[j, i] / sigma_)**shape_
                    fitness = fitness_values[i]

                    if np.isinf(fitness):
                        fitness = INF

                    shared_fitness_values[i] += fitness*correction

    # print("dist_matrix: \n")
    # print(dist_matrix)

    return shared_fitness_values

# Elimination Operators

@jit(nopython=True)
def lambda_mu_elimination(population_and_offspring: np.ndarray, fitness_values: np.ndarray, shared_fitness_values: np.ndarray, lamda_: np.int64) -> np.ndarray:
    """
    Order the rows of the population_and_offspring matrix based on the corresponding values in the
    shared_fitness_values array and return the top lamda_ rows along with their shared fitness values.

    Parameters:
    - population_and_offspring (numpy.ndarray): A 2D matrix representing the population and offspring.
    - shared_fitness_values (numpy.ndarray): A 1D array containing shared fitness values corresponding to the rows.
    - lamda_ (numpy.int64): The number of best values to keep.

    Returns:
    tuple: A tuple containing two elements:
        - numpy.ndarray: The top lamda_ rows of the ordered population_and_offspring matrix.
        - numpy.ndarray: The corresponding shared fitness values for the selected population.
    """
    sorted_indices = np.argsort(shared_fitness_values)

    selected_population = population_and_offspring[sorted_indices][:lamda_]
    selected_fitness_values = fitness_values[sorted_indices][:lamda_]

    return selected_population, selected_fitness_values

@jit(nopython=True)
def k_tournament_elimination(population: np.ndarray, fitness_values: np.ndarray, shared_fitness_values: np.ndarray, k_: np.int64, lambda_) -> np.ndarray:
    """
    If an element is choses as a survivor, it is removed from the population and the fitness values.
    To make this more efficient a mask is used to keep track of the elements that are still in the population.
    """
    num_cities = population.shape[1]
    pop_size = population.shape[0]

    mask = np.ones(pop_size, dtype=np.bool_)

    survivors               = np.empty((lambda_, num_cities), dtype=np.int64)
    survivor_fitness_values = np.empty(lambda_, dtype=np.float64)

    for i in np.arange(0, lambda_):

        # k-tournament selection based on shared fitness values
        # tournament indices are chosen from the population, based on the mask of the population
        # The mask is used to calculate the probabilities of the individuals being chosen -> np.random.choice p parameter not supported by numba
        available_indices = np.arange(0, pop_size)[mask]
        tournament_indices = np.random.choice(available_indices, k_, replace=False)

        winner_index = np.argmin(shared_fitness_values[tournament_indices])

        tournament_winner_idx = tournament_indices[winner_index]

        tournament_winner = population[tournament_winner_idx]
        tournament_winner_fitness = fitness_values[tournament_winner_idx]

        # Remove the winner from the population and fitness values
        mask[tournament_winner_idx] = False

        # Add the winner to the survivors
        survivors[i] = tournament_winner
        survivor_fitness_values[i] = tournament_winner_fitness

    return survivors, survivor_fitness_values

@jit(nopython=True)
def numba_choice_indices(weights, k):
    # Get cumulative weights
    wc = np.cumsum(weights)
    # Total of weights
    m = wc[-1]
    # Array of sampled indices
    sample_idx = np.full(k, -1, np.int32)
    # Sampling loop
    i = 0
    tol = 1e-10  # Tolerance for numerical imprecision
    while i < k:
        # Pick random weight value
        r = m * np.random.rand()
        # Get corresponding index
        idx = np.searchsorted(wc, r + tol, side='right')
        # Check index was not selected before
        for j in range(i):
            if sample_idx[j] == idx:
                break
        else:
            # Save sampled index
            sample_idx[i] = idx
            i += 1
    return sample_idx

@jit(nopython=True)
def roulette_wheel_elimination(population_and_offspring: np.ndarray, fitness_values: np.ndarray, shared_fitness_values: np.ndarray, lamda_: np.int64):

    # print("----------")
    # print("shared_fitness_values: \n")
    # print(shared_fitness_values)

    weights = shared_fitness_values.copy()

    # weights[np.isinf(weights)] = 0
    # print("weights: \n")
    # print(weights)

    weights = np.sum(weights) / weights # Normalize the weights, such that 

    print("population fitness_values")
    print(fitness_values)
    print("distribution")
    print(weights)

    # Invert the weights 

    # print("normalized weights: \n")
    # print(weights)
    # print("----------")

    # selected_indices = np.random.choice(np.arange(0, population_and_offspring.shape[0]), lamda_, replace=False, p=probabilities)
    # the p parameter is not supported by numba

    selected_idx = numba_choice_indices(weights, lamda_)
    print("selected_idx: \n")
    print(selected_idx)

    # print("selected_idx: \n")
    # print(selected_idx)

    selected_population = population_and_offspring[selected_idx]
    selected_fitness_values = fitness_values[selected_idx]

    return selected_population, selected_fitness_values

@jit(nopython=True)
def eliminitaion(population: np.ndarray, fitness_values: np.ndarray, sigma_: np.float64, alpha_: np.float64, lambda_: np.int64, elimination_k_: np.int64) -> np.ndarray:
    num_cities = population.shape[1]
    pop_indices = np.arange(0, population.shape[0])

    mask = np.ones(population.shape[0], dtype=np.bool_) # True if the individual is still in the population

    penalties = np.ones(population.shape[0], dtype=np.float64)

    fitnesses = fitness_values.copy()
    for idx in np.arange(0, fitnesses.shape[0]):
        if fitnesses[idx] == np.inf:
            fitnesses[idx] = INF
    
    # shared_fitnesses = np.zeros(population.shape[0], dtype=np.float64)
    # shared_fitnesses = fitness_values.copy()

    survivors = np.empty((lambda_, num_cities), dtype=np.int64)
    survivors_fitnesses = np.empty(lambda_, dtype=np.float64)

    # Find the elite
    # elite_idx = np.argmin(fitness_values)
    # winner = population[elite_idx]
    # survivors[0] = winner
    # survivors_fitnesses[0] = fitness_values[elite_idx]
    # mask[elite_idx] = False

    def update_shared_fitnesses(shared_fitnesses: np.ndarray, fitnesses: np.ndarray, penalties: np.ndarray, winner: np.ndarray):
        random_sample_indices = np.random.choice(pop_indices[mask], pop_indices[mask].shape[0], replace=False)
        # random_sample_indices = pop_indices[mask]
        for j in np.arange(0, random_sample_indices.shape[0]):
            random_sample_idx = random_sample_indices[j]
            random_sample = population[random_sample_idx]

            distance = cyclic_rtype_distance(winner, random_sample)
            if distance < sigma_:
                penalties[random_sample_idx] += 1.0 - (distance / sigma_) ** alpha_

        # shared_fitnesses += fitnesses*penalties
    
    # update_shared_fitnesses(shared_fitnesses, fitnesses, penalties, winner)

    for i in np.arange(0, lambda_):

        # [ K Tournament Selection ]
        # k tournamenst selection based on fitness values
        tournament_indices = np.random.choice(pop_indices[mask], elimination_k_, replace=False)
        shared_fitnesses = fitnesses*penalties
        winner_index = np.argmin(shared_fitnesses[tournament_indices])
        tournament_winner_idx = tournament_indices[winner_index]
        winner = population[tournament_winner_idx]
        winner_fitness = fitness_values[tournament_winner_idx]

        # [ Best ]
        # Find best individual from population according to shared fitness values and mask
        # tournament_indices = pop_indices[mask]
        # shared_fitnesses = fitnesses*penalties
        # winner_index = np.argmin(shared_fitnesses[tournament_indices])
        # tournament_winner_idx = tournament_indices[winner_index]
        # winner = population[tournament_winner_idx]
        # winner_fitness = fitness_values[tournament_winner_idx]

        # Remove the winner from the population and fitness values
        mask[tournament_winner_idx] = False

        # Add the winner to the survivors
        survivors[i] = winner
        survivors_fitnesses[i] = winner_fitness

        # # Update penalties
        update_shared_fitnesses(shared_fitnesses, fitnesses, penalties, winner)

        # print("penalties")
        print(penalties)

        # print("sared fitnesses")
        print(shared_fitnesses)

    return survivors, survivors_fitnesses

# Generate Offspring
@jit(nopython=True)
def generate_offspring(population: np.ndarray, 
                       fitness_values: np.ndarray, 
                       offspring: np.ndarray, 
                       offspring_fitness_values: np.ndarray, 
                       mu_: np.int64, 
                       selection_k_: np.int64, 
                       off_mutation_alpha_: np.float64, 
                       distance_matrix: np.ndarray,
                       iteration: np.int64):

    # Generate offspring
    for i in np.arange(0, mu_):
        # Select parents from the population
        parent_1 = k_tournament_selection(population, fitness_values, selection_k_)
        parent_2 = k_tournament_selection(population, fitness_values, selection_k_)

        # Perform crossover on the parents
        child = ox_crossover(parent_1, parent_2)

        # Perform mutation on the child
        child = inversion_mutation(child, off_mutation_alpha_, distance_matrix)

        # Perform local search on the child
        if iteration % 7 == 0:
            child, child_fitness = two_opt_local_search_fast(child, distance_matrix)
        # elif iteration % 13 == 0:
        #     child, child_fitness = two_opt_local_search_slow(child, distance_matrix)
        else:
            child, child_fitness = one_opt_local_search(child, distance_matrix)
            
        # Add the child to the offspring
        offspring[i] = child

        # Evaluate the fitness of the child
        offspring_fitness_values[i] = child_fitness

    return offspring, offspring_fitness_values

class r0713047:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):

        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        num_cities = distance_matrix.shape[0]

        ################ PARAMETERS ################
        lambda_                   = np.int64(200)
        mu_                       = np.int64(200)
        selection_k_              = np.int64(lambda_//60)          
        elimination_k_            = np.int64((lambda_ + mu_)//200)  
        off_mutation_alpha_       = np.float64(0.2)               
        pop_mutation_alpha_       = np.float64(0.3)               
        num_elites_               = np.int64(1)
        fitness_sharing_sigma_    = np.float64(0.2*num_cities)     
        fitness_sharing_alpha_    = np.float64(2)                  
        ############################################

        # initialize population
        population, fitness_values = initialize_population(distance_matrix, lambda_)

        # print()
        # print("Initial Population")
        # print(population)
        # print("Initial Fitness Values")
        # print(fitness_values)
        # print()

        # Initialize the offspring
        offspring                = np.empty((mu_, num_cities), dtype=np.int64)
        offspring_fitness_values = np.empty(mu_, dtype=np.float64)

        # Initialize population and offspring
        population_and_offspring                = np.empty((lambda_ + mu_, num_cities), dtype=np.int64)
        population_and_offspring_fitness_values = np.empty(lambda_ + mu_, dtype=np.float64)

        # Mainloop
        iteration = 0
        print_iteration = 20
        while True: 

            # if iteration % print_iteration == 0:
            #     print("----------------------------------------")
            #     print("\033[95m new generation \n \033[0m" )
            #     print("gen: ", iteration)
            #     print("----------")
            #     print()
            #     print(" \033[94m fitness_values before finding elites \n \033[0m")
            #     print(fitness_values)
            #     print("population before finding elites")
            #     print(population)
            #     print()

            # Find the elites
            elites, elite_fitness_values = find_elites(population, fitness_values, num_elites_)

            # if iteration % print_iteration == 0:
            #     print("\033[94m Elites \n \033[0m")
            #     print("Elite Fitness Values")
            #     print(elite_fitness_values)
            #     print()

            # Generate offspring
            offspring, offspring_fitness_values = generate_offspring(population, 
                                                                     fitness_values, 
                                                                     offspring, 
                                                                     offspring_fitness_values, 
                                                                     mu_, 
                                                                     selection_k_, 
                                                                     off_mutation_alpha_, 
                                                                     distance_matrix,
                                                                     iteration)

            
            # if iteration % print_iteration == 0:
            #     print("\033[94m Population before mutation \n \033[0m")
            #     print("Population Fitness Values: \n")
            #     print(fitness_values)
            #     print("Population before mutation: \n")
            #     print(population)
            #     print()

            #     print("\033[94m Offspring before recombination \n \033[0m")
            #     print("Offspring Fitness Values: \n")
            #     print(offspring_fitness_values)
            #     print("Offspring: \n")
            #     print(offspring)
            #     print()

            # Mutate the population
            population = mutate_population(population, pop_mutation_alpha_, distance_matrix)
            fitness_values = fitnesses(population, distance_matrix)

            # Combine the population and offspring
            population_and_offspring[:lambda_] = population
            population_and_offspring[lambda_:] = offspring
            population_and_offspring_fitness_values[:lambda_] = fitness_values
            population_and_offspring_fitness_values[lambda_:] = offspring_fitness_values

            # if iteration % print_iteration == 0:
            #     print("\033[94m Population and Offspring \n \033[0m")
            #     print("fitness_values: \n")
            #     print(population_and_offspring_fitness_values)
            #     print("population: \n")
            #     print(population_and_offspring)
            #     print()

            # Perform fitness sharing on the combined population and offspring
            # shared_fitness_values = fitness_sharing(population_and_offspring_fitness_values, 
            #                                         population_and_offspring, 
            #                                         fitness_sharing_alpha_, 
            #                                         fitness_sharing_sigma_)

            # Select the best lambda_ individuals from the combined population and offspring according to their shared fitness values
            # population, fitness_values = lambda_mu_elimination(population_and_offspring,
            #                                                    population_and_offspring_fitness_values, 
            #                                                    shared_fitness_values, 
            #                                                    lambda_)

            # population, fitness_values = k_tournament_elimination(population_and_offspring,
            #                                                       population_and_offspring_fitness_values, 
            #                                                       shared_fitness_values, 
            #                                                       elimination_k_, 
            #                                                       lambda_)

            # population, fitness_values = roulette_wheel_elimination(population_and_offspring,
            #                                                         population_and_offspring_fitness_values, 
            #                                                         shared_fitness_values, 
            #                                                         lambda_)

            population, fitness_values = eliminitaion(population_and_offspring,
                                                      population_and_offspring_fitness_values, 
                                                      fitness_sharing_sigma_, 
                                                      fitness_sharing_alpha_, 
                                                      lambda_, 
                                                      elimination_k_)

            # Add the elites to the population
            population, fitness_values = add_elites(population, fitness_values, elites, elite_fitness_values)

            # if iteration % print_iteration == 0:
            #     print("\033[94m Survivors: \n \033[0m")
            #     print("fitness_values: \n")
            #     print(fitness_values)
            #     print("population: \n")
            #     print(population)
            #     print()

            #     print("####")
            #     print("recalc survivors fitness values: \n")
            #     test = fitnesses(population, distance_matrix)
            #     print(test)
            #     print()

            # Report results
            pop_mean_fitness, pop_best_fitness, pop_best_permutation = generate_report_values(population, fitness_values)
            timeLeft = self.reporter.report(pop_mean_fitness, pop_best_fitness, pop_best_permutation)

            if timeLeft < 5:
                break
            
            # End of Mainloop
            iteration += 1

        
        # The End
        pop_mean_fitness, pop_best_fitness, pop_best_permutation = generate_report_values(population, fitness_values)
        timeLeft = self.reporter.report(pop_mean_fitness, pop_best_fitness, pop_best_permutation)

        return  pop_mean_fitness, pop_best_fitness, pop_best_permutation

if __name__ == "__main__":
    problem = r0713047()
    problem.optimize("tours/tour1000.csv")