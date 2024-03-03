"""
Extened the numba library with custom to bytes function. 
How is this not yet inplemented in this library?!?
"""

from r0713047 import valid_permutation

import Reporter
import numpy as np
from numba import jit
import time # For testing

from numba import types
from numba.typed import Dict

from numba.core import types, typing, errors, cgutils, extending
from numba.cpython.charseq import _make_constant_bytes, bytes_type
from numba.core.imputils import (lower_builtin, lower_getattr,
                                 lower_getattr_generic,
                                 lower_setattr_generic,
                                 lower_cast, lower_constant,
                                 iternext_impl, impl_ret_borrowed,
                                 impl_ret_new_ref, impl_ret_untracked,
                                 RefType)
from numba.core.extending import (register_jitable, overload, overload_method,
                                  intrinsic)

np.set_printoptions(edgeitems=10,linewidth=200)

@lower_cast(types.Array, types.Bytes)
def array_to_bytes(context, builder, fromty, toty, val):
    arrty = make_array(fromty)
    arr = arrty(context, builder, val)

    itemsize = arr.itemsize
    nbytes = builder.mul(itemsize, arr.nitems)

    bstr = _make_constant_bytes(context, builder, nbytes)

    if (fromty.is_c_contig and fromty.layout == "C"):
        cgutils.raw_memcpy(builder, bstr.data, arr.data, arr.nitems, itemsize)
    else:
        shape = cgutils.unpack_tuple(builder, arr.shape)
        strides = cgutils.unpack_tuple(builder, arr.strides)
        layout = fromty.layout
        intp_t = context.get_value_type(types.intp)

        byteidx = cgutils.alloca_once(
            builder, intp_t, name="byteptr", zfill=True
        )
        with cgutils.loop_nest(builder, shape, intp_t) as indices:
            ptr = cgutils.get_item_pointer2(
                context, builder, arr.data, shape, strides, layout, indices
            )
            srcptr = builder.bitcast(ptr, bstr.data.type)

            idx = builder.load(byteidx)
            destptr = builder.gep(bstr.data, [idx])

            cgutils.memcpy(builder, destptr, srcptr, itemsize)
            builder.store(builder.add(idx, itemsize), byteidx)

    return bstr._getvalue()

@intrinsic
def _array_tobytes_intrinsic(typingctx, b):
    assert isinstance(b, types.Array)
    sig = bytes_type(b)

    def codegen(context, builder, sig, args):
        [arr] = args
        return array_to_bytes(context, builder, b, bytes_type, arr)
    return sig, codegen

@overload_method(types.Array, "tobytes")
def impl_array_tobytes(arr):
    if isinstance(arr, types.Array):
        def impl(arr):
            return _array_tobytes_intrinsic(arr)
        return impl

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

    Example:
    >>> permutation_1 = np.array([1, 5, 2, 4, 0, 3])
    >>> permutation_2 = np.array([5, 1, 4, 0, 3, 2])
    >>> cyclic_rtype_distance(permutation_1, permutation_2)
    4

    Runtime: O(n), where n is the length of the permutations.

    Cyclic RType distance was introduced in:
    V.A. Cicirello, "The Permutation in a Haystack Problem and the Calculus of Search Landscapes,"
    IEEE Transactions on Evolutionary Computation, 20(3):434-446, June 2016.

    Author: Vincent A. Cicirello, https://www.cicirello.org/
    """
    assert  permutation_1.size == permutation_2.size

    count_non_shared_edges = 0
    successors_2 = np.empty_like(permutation_2)

    # Roll
    for i in np.arange(successors_2.size):
        successors_2[permutation_2[i]] = permutation_2[(i + 1) % successors_2.size]

    for i in np.arange(successors_2.size):
        if permutation_1[(i + 1) % successors_2.size] != successors_2[permutation_1[i]]:
            count_non_shared_edges += 1

    return count_non_shared_edges

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
    
    # Dict with keys as string and values of type float
    cache = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

    num_permutations = population.shape[0]

    dist_matrix = np.zeros((num_permutations, num_permutations), dtype=np.float64)
    shared_fitness_values = np.zeros(num_permutations, dtype=np.float64)

    for i in np.arange(0, num_permutations):
        for j in np.arange(0, num_permutations):

            if j >= i: 


                if key in cache:
                    print("cache hit")
                    dist_matrix[i, j] = cache[key]
                else:
                    cache[key] = dist_matrix[i, j]

                if dist_matrix[i, j] <= sigma_:
                    shared_fitness_values[i] += fitness_values[i]*(1 - (dist_matrix[i, j] / sigma_)**shape_)

            else:
                if dist_matrix[j, i] <= sigma_:
                    shared_fitness_values[i] += fitness_values[i]*(1 - (dist_matrix[j, i] / sigma_)**shape_)


    # print("fits 1: \n", fitness_values)
    # print("distances 1: \n", dist_matrix)
    # print("shared fitness values: \n ", shared_fitness_values)

    return shared_fitness_values

# def fitness_sharing(distanceMatrix: np.ndarray, population: List[Individual], 
#                     survivors: np.ndarray, alpha: float, sigma_percentage: float, 
#                     original_fits: np.ndarray, all_distances_hashmap: dict) -> np.ndarray:
def fitness_sharing_2(population: np.ndarray, 
                      original_fits: np.ndarray,
                      alpha: float, 
                      sigma: float, 
                      ) -> np.ndarray:
                      
    
    distances = np.zeros((len(population), len(population)))
    for i in range(len(population)):
        for j in range(len(population)):
            # distance1 = all_distances_hashmap.get((population[i], survivors[j]), -1)
            # distance2 = all_distances_hashmap.get((survivors[j], population[i]), -1)
            # if distance1 == -1 and distance2 == -1:
            #     distance = distance_from_to(population[i], survivors[j])
            #     all_distances_hashmap[(population[i], survivors[j])] = distance
            # else:
            #     if distance1 != -1:
            #         distance = distance1
            #     else:
            #         distance = distance2
            distances[i][j] = cyclic_rtype_distance(population[i], population[j])

    shared_part = (1 - (distances / sigma) ** alpha)
    shared_part *= np.array(distances <= sigma)
    sum_shared_part = np.sum(shared_part, axis=1)
    shared_fitnesses = original_fits * sum_shared_part
    shared_fitnesses = np.where(np.isnan(shared_fitnesses), np.inf, 
                                shared_fitnesses)

    # print("fits 2: \n", original_fits)
    # print("distances 2: \n", distances)
    # print("shared fitness values 2: \n ", shared_fitnesses)

    return shared_fitnesses

if __name__ == "__main__":
    # Read distance matrix from file.
    filename = "tours/tour50.csv"
    file = open(filename)
    distance_matrix = np.loadtxt(file, delimiter=",")
    file.close()

    num_cities = distance_matrix.shape[0]

    # generate valid population of size 5
    population = np.zeros((5, num_cities), dtype=np.int64)
    fitness_values = np.zeros(5, dtype=np.float64)
    for i in range(5):
        population[i], fitness_values[i] = valid_permutation(distance_matrix)

    
    # warmup
    fitness_sharing(fitness_values, population, 1, 0.5*num_cities)

    print()
    print("--------------------")
    print()

    # Test fitness sharing
    start = time.time()
    shared_fitness_values = fitness_sharing(fitness_values, population, 1, 0.5*num_cities)
    end = time.time()
    print("fitness sharing: ", end - start)
    print("shared fitness values: \n ", shared_fitness_values)

    print()


    # Test fitness sharing 2
    start = time.time()
    shared_fitness_values_2 = fitness_sharing_2(population,  fitness_values, 1, 0.5*num_cities)
    end = time.time()
    print("fitness sharing 2: ", end - start)
    print("shared fitness values 2: \n ", shared_fitness_values_2)





    

